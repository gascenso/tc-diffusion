# tc_diffusion/train_loop.py
import base64
import glob
import json
import math
import os
import pickle
import random
import shutil
import time
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tensorflow as tf #type: ignore
from tensorflow import keras #type: ignore
from tqdm.auto import tqdm # type: ignore

from .data import _build_aug_policy, augment_batch_x_given_y, create_dataset, ss_class_midpoint_kt
from .model_unet import build_unet
from .diffusion import Diffusion
from .evaluation.evaluator import TCEvaluator
from .plotting import save_image_grid


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to lr_min.

    When warmup_steps=0 the warmup phase is skipped and cosine decay begins
    immediately.  When lr_min==lr_peak the schedule is effectively constant
    (zero-amplitude cosine).
    """

    def __init__(self, lr_peak: float, lr_min: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.lr_peak = float(lr_peak)
        self.lr_min = float(lr_min)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.constant(float(self.warmup_steps), tf.float32)
        total_steps = tf.constant(float(self.total_steps), tf.float32)
        lr_peak = tf.constant(self.lr_peak, tf.float32)
        lr_min = tf.constant(self.lr_min, tf.float32)

        # Linear warmup: 0 → lr_peak over warmup_steps
        warmup_lr = lr_peak * step / tf.maximum(warmup_steps, 1.0)

        # Cosine decay: lr_peak → lr_min over (total_steps - warmup_steps)
        decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value((step - warmup_steps) / decay_steps, 0.0, 1.0)
        cosine_lr = lr_min + 0.5 * (lr_peak - lr_min) * (
            1.0 + tf.cos(tf.constant(math.pi, tf.float32) * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "lr_peak": self.lr_peak,
            "lr_min": self.lr_min,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


class EMA(tf.Module):
    """Exponential Moving Average (EMA) over model weights.

    Updates after each optimizer step:
        ema = decay * ema + (1 - decay) * w
    """

    def __init__(self, model: tf.keras.Model, decay: float):
        super().__init__(name="ema")
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.model = model
        self.decay = float(decay)

        # Shadow variables mirror model.weights (trainable + non-trainable)
        self.shadow_vars = [
            tf.Variable(w, trainable=False, dtype=w.dtype, name=f"ema/{w.name.split(':')[0]}")
            for w in model.weights
        ]

    def update(self):
        d = tf.constant(self.decay, dtype=tf.float32)
        one_minus_d = 1.0 - d
        for s, w in zip(self.shadow_vars, self.model.weights):
            s.assign(d * s + one_minus_d * w)

    def copy_to_model(self):
        for s, w in zip(self.shadow_vars, self.model.weights):
            w.assign(s)

    def copy_from_model(self):
        for s, w in zip(self.shadow_vars, self.model.weights):
            s.assign(w)

    def get_model_weights_snapshot(self):
        return [w.numpy() for w in self.model.weights]

    def restore_model_weights_snapshot(self, snapshot):
        for arr, w in zip(snapshot, self.model.weights):
            w.assign(arr)

@contextmanager
def _ema_weights_applied(ema):
    """Temporarily swap the live model weights to EMA weights."""
    if ema is None:
        yield
        return

    snapshot = ema.get_model_weights_snapshot()
    ema.copy_to_model()
    try:
        yield
    finally:
        ema.restore_model_weights_snapshot(snapshot)

def _state_path(out_dir: Path) -> Path:
    return out_dir / "run_state.json"


def _checkpoint_dir(out_dir: Path) -> Path:
    return out_dir / "train_checkpoints"


def _find_latest_last_weights(out_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(out_dir / "weights_last.epoch_*.weights.h5")))
    if not files:
        return None
    return Path(files[-1])


def _delete_other_last_weights(out_dir: Path, keep: Path):
    for f in glob.glob(str(out_dir / "weights_last.epoch_*.weights.h5")):
        fp = Path(f)
        if fp != keep:
            try:
                fp.unlink()
            except Exception:
                pass

def _find_latest_ema_last_weights(out_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(out_dir / "weights_ema_last.epoch_*.weights.h5")))
    if not files:
        return None
    return Path(files[-1])

def _delete_other_ema_last_weights(out_dir: Path, keep: Path):
    for f in glob.glob(str(out_dir / "weights_ema_last.epoch_*.weights.h5")):
        fp = Path(f)
        if fp != keep:
            try:
                fp.unlink()
            except Exception:
                pass

def _load_state(out_dir: Path) -> dict | None:
    sp = _state_path(out_dir)
    if not sp.exists():
        return None
    with sp.open("r") as f:
        return json.load(f)


def _save_state(out_dir: Path, state: dict):
    sp = _state_path(out_dir)
    tmp = out_dir / (sp.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(sp)


def _encode_py_state(obj) -> str:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("ascii")


def _copy_file_atomic(src: Path, dst: Path) -> str:
    """Publish dst from src without re-serializing model weights."""
    src = Path(src)
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return "copied"
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _publish_weight_snapshot(
    source_path: Path | None,
    dest_path: Path,
    *,
    save_fn,
    label: str,
) -> Path:
    """Save once, then reuse the written artifact for sibling snapshots."""
    t0 = time.perf_counter()
    if source_path is None:
        save_fn(dest_path)
        dt = time.perf_counter() - t0
        print(f"  Saved {label} to {dest_path} ({dt:.1f}s)")
        return dest_path

    method = _copy_file_atomic(source_path, dest_path)
    dt = time.perf_counter() - t0
    print(
        f"  {method.capitalize()} {label} to {dest_path} "
        f"from {source_path.name} ({dt:.2f}s)"
    )
    return source_path


def _decode_py_state(encoded: str):
    return pickle.loads(base64.b64decode(encoded.encode("ascii")))


@dataclass
class InputPipelineSummary:
    profile: str
    loader: str
    backend: str
    prefetch_batches: int
    steps: int
    samples: int
    avg_batch_prep_time_sec: float
    avg_data_wait_time_sec: float
    avg_compute_time_sec: float
    avg_step_time_sec: float
    wait_fraction_pct: float
    throughput_samples_per_sec: float
    avg_backend_load_time_sec: float | None = None
    avg_preprocess_time_sec: float | None = None

    def to_json(self, epoch: int) -> dict:
        out = {
            "epoch": int(epoch),
            "profile": self.profile,
            "loader": self.loader,
            "backend": self.backend,
            "prefetch_batches": int(self.prefetch_batches),
            "steps": int(self.steps),
            "samples": int(self.samples),
            "avg_batch_prep_time_sec": float(self.avg_batch_prep_time_sec),
            "avg_data_wait_time_sec": float(self.avg_data_wait_time_sec),
            "avg_compute_time_sec": float(self.avg_compute_time_sec),
            "avg_step_time_sec": float(self.avg_step_time_sec),
            "wait_fraction_pct": float(self.wait_fraction_pct),
            "throughput_samples_per_sec": float(self.throughput_samples_per_sec),
        }
        if self.avg_backend_load_time_sec is not None:
            out["avg_backend_load_time_sec"] = float(self.avg_backend_load_time_sec)
        if self.avg_preprocess_time_sec is not None:
            out["avg_preprocess_time_sec"] = float(self.avg_preprocess_time_sec)
        return out


class InputPipelineProfiler:
    def __init__(self, enabled: bool, window_steps: int, out_dir: Path):
        self.enabled = bool(enabled)
        self.window_steps = max(1, int(window_steps))
        self.out_dir = Path(out_dir)
        self.jsonl_path = self.out_dir / "data_pipeline_profile.jsonl"
        self._epoch_reset()

    def _epoch_reset(self):
        self.step_count = 0
        self.sample_count = 0
        self.total_batch_prep_time = 0.0
        self.total_data_wait_time = 0.0
        self.total_compute_time = 0.0
        self.total_step_time = 0.0
        self.total_backend_load_time = 0.0
        self.total_preprocess_time = 0.0
        self.batch_info_count = 0
        self.wait_window = deque(maxlen=self.window_steps)
        self.step_window = deque(maxlen=self.window_steps)
        self.throughput_window = deque(maxlen=self.window_steps)
        self.batch_prep_window = deque(maxlen=self.window_steps)

    def start_epoch(self):
        self._epoch_reset()

    def record(
        self,
        *,
        batch_size: int,
        data_wait_time_sec: float,
        compute_time_sec: float,
        batch_info: dict | None = None,
    ):
        if not self.enabled:
            return

        batch_prep_time_sec = float(
            batch_info.get("batch_prep_time_sec", data_wait_time_sec)
            if batch_info is not None
            else data_wait_time_sec
        )
        step_time_sec = float(data_wait_time_sec + compute_time_sec)

        self.step_count += 1
        self.sample_count += int(batch_size)
        self.total_batch_prep_time += batch_prep_time_sec
        self.total_data_wait_time += float(data_wait_time_sec)
        self.total_compute_time += float(compute_time_sec)
        self.total_step_time += step_time_sec

        if batch_info is not None:
            self.total_backend_load_time += float(batch_info.get("load_time_sec", 0.0))
            self.total_preprocess_time += float(batch_info.get("preprocess_time_sec", 0.0))
            self.batch_info_count += 1

        throughput = float(batch_size) / max(step_time_sec, 1e-12)
        self.batch_prep_window.append(batch_prep_time_sec)
        self.wait_window.append(float(data_wait_time_sec))
        self.step_window.append(step_time_sec)
        self.throughput_window.append(throughput)

    def running(self) -> dict | None:
        if not self.enabled or self.step_count <= 0:
            return None
        return {
            "batch_prep_ms": 1000.0 * (sum(self.batch_prep_window) / max(1, len(self.batch_prep_window))),
            "wait_ms": 1000.0 * (sum(self.wait_window) / max(1, len(self.wait_window))),
            "step_ms": 1000.0 * (sum(self.step_window) / max(1, len(self.step_window))),
            "samples_per_sec": sum(self.throughput_window) / max(1, len(self.throughput_window)),
        }

    def finish_epoch(self, *, epoch: int, loader_info: dict) -> InputPipelineSummary | None:
        if not self.enabled or self.step_count <= 0:
            return None

        avg_step = self.total_step_time / max(1, self.step_count)
        summary = InputPipelineSummary(
            profile=str(loader_info.get("profile", "unknown")),
            loader=str(loader_info.get("loader", "unknown")),
            backend=str(loader_info.get("backend", "unknown")),
            prefetch_batches=int(loader_info.get("prefetch_batches", 0)),
            steps=int(self.step_count),
            samples=int(self.sample_count),
            avg_batch_prep_time_sec=self.total_batch_prep_time / max(1, self.step_count),
            avg_data_wait_time_sec=self.total_data_wait_time / max(1, self.step_count),
            avg_compute_time_sec=self.total_compute_time / max(1, self.step_count),
            avg_step_time_sec=avg_step,
            wait_fraction_pct=100.0 * self.total_data_wait_time / max(self.total_step_time, 1e-12),
            throughput_samples_per_sec=self.sample_count / max(self.total_step_time, 1e-12),
            avg_backend_load_time_sec=(
                self.total_backend_load_time / self.batch_info_count if self.batch_info_count > 0 else None
            ),
            avg_preprocess_time_sec=(
                self.total_preprocess_time / self.batch_info_count if self.batch_info_count > 0 else None
            ),
        )

        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(summary.to_json(epoch=epoch)) + "\n")
        return summary

def evaluate_loss(
    diffusion,
    model,
    ds_val,
    val_steps: int | None = None,
    *,
    base_seed: int = 42,
    split_seed_offset: int = 0,
):
    """Compute mean diffusion loss weighted by actual batch size.

    If val_steps is None or <=0, iterate over the *entire* ds_val.
    For validation we also use stateless per-batch timestep/noise draws, so the
    result is deterministic for a fixed model state and dataset.

    Weighting by sample count rather than batch count ensures the last
    (potentially partial) batch does not get inflated weight.
    """
    full_pass = (val_steps is None) or (int(val_steps) <= 0)
    loss_sum = 0.0
    n_samples = 0
    for batch, (x0, cond) in enumerate(ds_val):
        if (not full_pass) and batch >= int(val_steps):
            break
        batch_size = int(tf.shape(x0)[0])
        t_seed = tf.constant(
            [int(base_seed) + int(split_seed_offset), int(batch)],
            dtype=tf.int32,
        )
        noise_seed = tf.constant(
            [int(base_seed) + int(split_seed_offset) + 1, int(batch)],
            dtype=tf.int32,
        )
        t = tf.random.stateless_uniform(
            shape=(batch_size,),
            seed=t_seed,
            minval=0,
            maxval=diffusion.num_steps,
            dtype=tf.int32,
        )
        noise = tf.random.stateless_normal(
            shape=tf.shape(x0),
            seed=noise_seed,
            dtype=x0.dtype,
        )
        loss = diffusion.loss(model, x0, cond, training=False, t=t, noise=noise)
        loss_sum += float(loss.numpy()) * batch_size
        n_samples += batch_size
    return loss_sum / max(1, n_samples)


def _default_epoch_preview_cfg(cfg: dict) -> dict:
    ev = dict(cfg.get("evaluation", {}))
    preview = dict(ev.get("epoch_preview", {}))
    preview.setdefault("enabled", False)
    preview.setdefault("every_epochs", 20)
    preview.setdefault("n_per_class", 10)
    preview.setdefault("ncols", 5)
    preview.setdefault("use_ema", True)
    preview.setdefault("seed", ev.get("seed", 123))
    preview.setdefault("sampler", ev.get("sampler", "dpmpp_2m"))
    preview.setdefault("sampling_steps", ev.get("sampling_steps", ev.get("ddim_steps", None)))
    preview.setdefault("ddim_eta", ev.get("ddim_eta", 0.0))
    preview.setdefault("guidance_scale", ev.get("guidance_scale", 0.0))
    preview.setdefault("gen_batch_size", ev.get("gen_batch_size", None))
    return preview


def _resolve_epoch_preview_wind_target_kt(cfg: dict, ss_cat: int) -> float:
    cond_cfg = cfg.get("conditioning", {})
    targets = cond_cfg.get("eval_wind_targets_kt")
    if isinstance(targets, dict):
        key = str(int(ss_cat))
        if key in targets:
            try:
                return float(targets[key])
            except Exception:
                pass
    return ss_class_midpoint_kt(int(ss_cat))


def _save_epoch_preview_samples(
    *,
    cfg: dict,
    out_dir: Path,
    epoch: int,
    model,
    diffusion,
    preview_cfg: dict,
    using_ema_weights: bool,
) -> Path:
    n_per_class = int(preview_cfg["n_per_class"])
    if n_per_class <= 0:
        raise ValueError(f"evaluation.epoch_preview.n_per_class must be > 0, got {n_per_class}")

    ncols = max(1, min(int(preview_cfg["ncols"]), n_per_class))
    seed = int(preview_cfg["seed"])
    image_size = int(cfg["data"]["image_size"])
    bt_min_k = float(cfg["data"]["bt_min_k"])
    bt_max_k = float(cfg["data"]["bt_max_k"])
    num_classes = int(cfg["conditioning"]["num_ss_classes"])
    use_wind_speed = bool(cfg.get("conditioning", {}).get("use_wind_speed", False))

    sampling_steps = preview_cfg.get("sampling_steps")
    if sampling_steps is not None:
        sampling_steps = int(sampling_steps)

    gen_batch_size_cfg = preview_cfg.get("gen_batch_size")
    if gen_batch_size_cfg is None:
        gen_batch_size = int(cfg["data"].get("batch_size", n_per_class))
    else:
        gen_batch_size = int(gen_batch_size_cfg)
    if gen_batch_size <= 0:
        raise ValueError(
            f"evaluation.epoch_preview.gen_batch_size must be > 0 when set, got {gen_batch_size}"
        )
    gen_batch_size = min(gen_batch_size, n_per_class)

    preview_root = out_dir / "qualitative_samples" / f"epoch_{epoch+1:04d}"
    preview_root.mkdir(parents=True, exist_ok=True)

    wind_targets_kt = {}
    for c in range(num_classes):
        tf.random.set_seed(seed + c)
        raw_chunks = []
        remaining = n_per_class
        wind_target = None
        if use_wind_speed:
            wind_target = float(_resolve_epoch_preview_wind_target_kt(cfg, c))
            wind_targets_kt[str(c)] = wind_target

        while remaining > 0:
            bsz = min(gen_batch_size, remaining)
            sample_outputs = diffusion.sample(
                model=model,
                batch_size=bsz,
                image_size=image_size,
                cond_value=c,
                wind_value_kt=wind_target,
                show_progress=False,
                guidance_scale=float(preview_cfg["guidance_scale"]),
                sampler=str(preview_cfg["sampler"]),
                num_sampling_steps=sampling_steps,
                ddim_eta=float(preview_cfg["ddim_eta"]),
                return_both=True,
            )
            raw_chunks.append(sample_outputs["raw_final"].numpy())
            remaining -= bsz

        x_samples = np.concatenate(raw_chunks, axis=0)
        save_image_grid(
            x_samples,
            path=str(preview_root / f"class_{c}.png"),
            bt_min_k=bt_min_k,
            bt_max_k=bt_max_k,
            ncols=ncols,
        )

    meta_path = preview_root / "meta.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "epoch": int(epoch + 1),
                "using_ema_weights": bool(using_ema_weights),
                "n_per_class": int(n_per_class),
                "ncols": int(ncols),
                "seed": int(seed),
                "sampler": str(preview_cfg["sampler"]),
                "sampling_steps": sampling_steps,
                "ddim_eta": float(preview_cfg["ddim_eta"]),
                "guidance_scale": float(preview_cfg["guidance_scale"]),
                "gen_batch_size": int(gen_batch_size),
                "class_wind_targets_kt": wind_targets_kt,
            },
            f,
            indent=2,
        )

    return preview_root


def resolve_train_alpha(cfg, epoch_idx: int, num_epochs: int) -> float:
    data_cfg = cfg.get("data", {})
    default_alpha = float(data_cfg.get("class_balance_alpha", 1.0))
    curr_cfg = data_cfg.get("class_balance_curriculum", {})
    if not isinstance(curr_cfg, dict) or not bool(curr_cfg.get("enabled", False)):
        return default_alpha

    start_alpha = float(curr_cfg.get("start_alpha", 1.0))
    end_alpha = float(curr_cfg.get("end_alpha", default_alpha))
    mode = str(curr_cfg.get("mode", "linear")).strip().lower()
    start_epoch = int(curr_cfg.get("start_epoch", 0))
    ramp_epochs = int(curr_cfg.get("ramp_epochs", max(1, num_epochs // 2)))

    if epoch_idx < start_epoch:
        return start_alpha

    if ramp_epochs <= 1:
        return end_alpha

    progress = (epoch_idx - start_epoch) / float(ramp_epochs - 1)
    progress = max(0.0, min(1.0, progress))

    if mode == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif mode != "linear":
        raise ValueError(
            f"Unsupported data.class_balance_curriculum.mode='{mode}'. Expected 'linear' or 'cosine'."
        )

    return (1.0 - progress) * start_alpha + progress * end_alpha

def train(cfg, resume: bool = False):
    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # training.seed is the authoritative seed for data order, train-time
    # stochasticity, and resumable RNG streams. If it is omitted or null we fall
    # back to 42 so resume-equivalent training still has a stable base seed.
    seed_cfg = cfg["training"].get("seed", 42)
    train_seed = 42 if seed_cfg is None else int(seed_cfg)
    random.seed(train_seed)
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)
    print(f"[train] Global seed set to {train_seed}")

    # Mixed precision — must be set before any model or layer is built.
    # With "mixed_float16", Keras layers compute in float16 but store weights in
    # float32, cutting memory ~40% and boosting throughput on Tensor Core GPUs.
    use_mixed_precision = bool(cfg["training"].get("mixed_precision", False))
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[train] Mixed precision enabled (mixed_float16)")

    # --- build train/val datasets ---
    data_cfg = cfg.setdefault("data", {})

    # Preserve eval_mode if present
    orig_eval_mode = data_cfg.get("eval_mode", "full")

    ds_train, train_sampler = create_dataset(cfg, split="train", return_train_generator=True)
    train_loader_info = train_sampler.describe() if hasattr(train_sampler, "describe") else {}
    pipeline_bits = [
        f"profile={train_loader_info.get('profile', 'unknown')}",
        f"loader={train_loader_info.get('loader', 'unknown')}",
        f"backend={train_loader_info.get('backend', 'unknown')}",
        f"prefetch_batches={train_loader_info.get('prefetch_batches', 0)}",
    ]
    for key in ("sort_within_shard", "max_samples_per_read", "warmup_shards"):
        if key in train_loader_info:
            pipeline_bits.append(f"{key}={train_loader_info[key]}")
    print("[data] Train pipeline: " + ", ".join(pipeline_bits))

    # Validation: build two deterministic finite sets:
    #   1) full pass over the val split (micro average)
    #   2) fixed balanced subset (tail-sensitive but stable)
    # Validation loss itself is made deterministic later via stateless RNG.
    data_cfg["eval_mode"] = "full"
    ds_val_full = create_dataset(cfg, split="val")

    data_cfg["eval_mode"] = "balanced_fixed"
    ds_val_balanced = create_dataset(cfg, split="val")

    # restore (so cfg isn't left in a surprising state)
    data_cfg["eval_mode"] = orig_eval_mode
    
    model = build_unet(cfg)
    diffusion = Diffusion(cfg)

    ema_decay = float(cfg.get("training", {}).get("ema_decay", 0.0))
    use_ema = (ema_decay is not None) and (ema_decay > 0.0)
    ema = EMA(model, ema_decay) if use_ema else None

    lr = float(cfg["training"]["lr"])
    lr_min = float(cfg["training"].get("lr_min", 1e-6))
    warmup_steps = int(cfg["training"].get("warmup_steps", 500))
    num_epochs = int(cfg["training"]["num_epochs"])
    steps_per_epoch = int(cfg["training"].get("steps_per_epoch", 2000))
    total_steps = num_epochs * steps_per_epoch
    val_every_epochs = int(cfg["training"].get("val_every_epochs", 1))
    if val_every_epochs <= 0:
        raise ValueError(f"training.val_every_epochs must be >= 1, got {val_every_epochs}")
    # If val_steps <= 0, we will do a full deterministic pass with fixed
    # timestep/noise draws as well as a fixed sample order.
    val_steps = int(cfg["training"].get("val_steps", 0))
    val_base_seed = train_seed
    log_interval = int(cfg["training"]["log_interval_steps"])
    save_last_weights = bool(cfg["training"].get("save_last_weights", True))
    save_ema_last_cfg = cfg["training"].get("save_ema_last_weights")
    if save_ema_last_cfg is None:
        save_ema_last_weights = save_last_weights
    else:
        save_ema_last_weights = bool(save_ema_last_cfg)

    lr_schedule = WarmupCosineDecay(
        lr_peak=lr,
        lr_min=lr_min,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    if use_mixed_precision:
        # LossScaleOptimizer multiplies the loss by a dynamic scale factor before
        # the backward pass so float16 gradients don't underflow, then divides
        # back before apply_gradients.  The scale is halved on any NaN/Inf step
        # and doubled every 2000 finite steps.
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe_profile_cfg = cfg["training"].get("input_pipeline_profile", {})
    pipe_profiler = InputPipelineProfiler(
        enabled=bool(pipe_profile_cfg.get("enabled", False)),
        window_steps=int(pipe_profile_cfg.get("window_steps", 50)),
        out_dir=out_dir,
    )
    if pipe_profiler.enabled:
        print(
            "[profile] Input-pipeline timing enabled "
            f"(window_steps={pipe_profiler.window_steps}, jsonl={pipe_profiler.jsonl_path.name})"
        )
    epoch_preview_cfg = _default_epoch_preview_cfg(cfg)
    if bool(epoch_preview_cfg["enabled"]):
        preview_every_epochs = int(epoch_preview_cfg["every_epochs"])
        if preview_every_epochs <= 0:
            raise ValueError(
                "evaluation.epoch_preview.every_epochs must be >= 1 when enabled, "
                f"got {preview_every_epochs}"
            )
        print(
            "[eval] Epoch previews enabled "
            f"(every_epochs={preview_every_epochs}, n_per_class={int(epoch_preview_cfg['n_per_class'])}, "
            f"use_ema={bool(epoch_preview_cfg['use_ema'])})"
        )
    checkpoint_items = {
        "model": model,
        "optimizer": optimizer,
        "epoch": tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch"),
        "global_step": tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step"),
    }
    if use_ema and ema is not None:
        checkpoint_items["ema"] = ema
    cfg_dropout_layer = model.get_layer("cfg_cond_dropout")
    cfg_dropout_rng = getattr(cfg_dropout_layer, "rng", None)
    if cfg_dropout_rng is not None:
        checkpoint_items["cfg_dropout_rng"] = cfg_dropout_rng
    checkpoint = tf.train.Checkpoint(**checkpoint_items)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(_checkpoint_dir(out_dir)),
        max_to_keep=1,
    )

    # ----- evaluation -----
    evaluator = TCEvaluator(cfg)

    # ----- early stopping config -----
    es_cfg = cfg["training"].get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    patience_epochs = int(es_cfg.get("patience_epochs", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    early_stopping_monitor = str(
        es_cfg.get("monitor", "ema" if use_ema else "raw")
    ).strip().lower()
    if early_stopping_monitor not in {"raw", "ema"}:
        raise ValueError(
            "training.early_stopping.monitor must be 'raw' or 'ema', "
            f"got {early_stopping_monitor!r}"
        )
    if early_stopping_monitor == "ema" and not use_ema:
        raise ValueError(
            "training.early_stopping.monitor='ema' requires training.ema_decay > 0."
        )
    if es_enabled:
        print(
            "[train] Early stopping enabled "
            f"(monitor={early_stopping_monitor}, patience={patience_epochs}, min_delta={min_delta:g})"
        )
    else:
        print("[train] Early stopping disabled")

    # ----- resume state -----
    start_epoch = 0
    best_epoch_loss = float("inf")
    best_epoch_balanced = float("inf")
    best_epoch_idx = -1
    best_epoch_loss_ema = float("inf")
    best_epoch_balanced_ema = float("inf")
    best_epoch_idx_ema = -1
    epochs_without_improvement = 0

    epoch_indices = []
    epoch_losses = []
    val_losses = []
    val_losses_balanced = []
    val_losses_ema = []
    val_losses_balanced_ema = []
    state = _load_state(out_dir)

    if resume:
        latest_ckpt = checkpoint_manager.latest_checkpoint
        if latest_ckpt is None:
            print(f"[resume] No checkpoint found in {out_dir}. Starting fresh.")
        else:
            print(f"[resume] Restoring checkpoint: {latest_ckpt}")
            restore_status = checkpoint.restore(latest_ckpt)
            restore_status.assert_existing_objects_matched()
            start_epoch = int(checkpoint.epoch.numpy())

            if state is not None:
                state_epoch = int(state.get("last_epoch", start_epoch))
                if state_epoch != start_epoch:
                    print(
                        f"[resume] Warning: checkpoint epoch ({start_epoch}) and "
                        f"run_state last_epoch ({state_epoch}) differ; using checkpoint state."
                    )
                best_epoch_loss = float(state.get("best_epoch_loss", best_epoch_loss))
                best_epoch_balanced = float(state.get("best_epoch_balanced", best_epoch_balanced))
                best_epoch_idx = int(state.get("best_epoch", best_epoch_idx))
                best_epoch_loss_ema = float(state.get("best_epoch_loss_ema", best_epoch_loss_ema))
                best_epoch_balanced_ema = float(state.get("best_epoch_balanced_ema", best_epoch_balanced_ema))
                best_epoch_idx_ema = int(state.get("best_epoch_ema", best_epoch_idx_ema))
                epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
                epoch_indices = list(state.get("epoch_indices", []))
                epoch_losses = list(state.get("epoch_losses", []))
                val_losses = list(state.get("val_losses", []))
                val_losses_balanced = list(state.get("val_losses_balanced", []))
                val_losses_ema = list(state.get("val_losses_ema", []))
                val_losses_balanced_ema = list(state.get("val_losses_balanced_ema", []))

                py_state = state.get("python_random_state")
                if py_state is not None:
                    random.setstate(_decode_py_state(py_state))
                np_state = state.get("numpy_random_state")
                if np_state is not None:
                    np.random.set_state(_decode_py_state(np_state))
                sampler_state = state.get("train_sampler_state")
                if sampler_state is not None:
                    train_sampler.set_state(_decode_py_state(sampler_state))
            else:
                print(
                    "[resume] Checkpoint restored, but run_state.json is missing; "
                    "history and Python/NumPy/train-sampler RNG state start fresh."
                )

            print(
                f"[resume] start_epoch={start_epoch}, best_epoch={best_epoch_idx}, "
                f"best_loss={best_epoch_loss:.6f}, best_epoch_ema={best_epoch_idx_ema}, "
                f"best_loss_ema={best_epoch_loss_ema:.6f}, global_step={int(checkpoint.global_step.numpy())}"
            )

    active_alpha = None

    # ----- compile the train step -----
    # All TF ops (tape, grads, apply, EMA) go inside the compiled function.
    # Python bookkeeping (step counter, loss accumulation, tqdm) stays outside.
    # use_ema is a Python bool resolved at trace time → the EMA branch is either
    # always present or always absent in the compiled graph, with zero runtime overhead.
    jit_compile = bool(cfg["training"].get("jit_compile", False))
    grad_clip_norm_cfg = cfg["training"].get("grad_clip_norm", 1.0)
    if grad_clip_norm_cfg is None:
        grad_clip_norm = None
        clip_gradients = False
    else:
        grad_clip_norm = float(grad_clip_norm_cfg)
        clip_gradients = grad_clip_norm > 0.0
    use_wind_speed = bool(cfg.get("conditioning", {}).get("use_wind_speed", False))
    p_aug, max_shift = _build_aug_policy(int(cfg["conditioning"]["num_ss_classes"]))

    @tf.function(reduce_retracing=True, jit_compile=jit_compile)
    def train_step(x0, cond, step_index):
        step_index = tf.cast(step_index, tf.int32)
        aug_seed = tf.stack(
            [
                tf.cast(train_seed + 20_000, tf.int32),
                step_index,
            ]
        )
        loss_seed = tf.stack(
            [
                tf.cast(train_seed + 40_000, tf.int32),
                step_index,
            ]
        )
        cond_labels = cond["ss_cat"] if use_wind_speed else cond
        x0 = augment_batch_x_given_y(
            x0,
            cond_labels,
            p_aug=p_aug,
            max_shift_per_class=max_shift,
            base_seed=aug_seed,
        )
        with tf.GradientTape() as tape:
            loss = diffusion.loss(model, x0, cond, seed=loss_seed)
            # Loss scaling must happen inside the tape so that the scale factor
            # is included in the gradient computation.
            if use_mixed_precision:
                if hasattr(optimizer, "get_scaled_loss"):
                    scaled_loss = optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = optimizer.scale_loss(loss)
            else:
                scaled_loss = loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if use_mixed_precision:
            if hasattr(optimizer, "get_unscaled_gradients"):
                grads = optimizer.get_unscaled_gradients(grads)
        # Clip after unscaling so the norm is in the natural gradient space.
        # A non-positive or null grad_clip_norm disables clipping entirely.
        if clip_gradients:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        # Guard against non-finite loss (NaN/Inf): zero out all gradients so
        # the weights are unchanged on this step.  The optimizer's step counter
        # and the LR schedule still advance normally so one bad step doesn't
        # stall training.  We can't use a Python `if` here (inside tf.function),
        # so we use tf.where with a scalar finite flag broadcast over each tensor.
        finite = tf.math.is_finite(loss)
        grads = [tf.where(finite, g, tf.zeros_like(g)) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if use_ema and ema is not None:
            ema.update()
        return loss

    global_step = int(checkpoint.global_step.numpy())

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            epoch_alpha = resolve_train_alpha(cfg, epoch_idx=epoch, num_epochs=num_epochs)
            if active_alpha is None or abs(epoch_alpha - active_alpha) > 1e-12:
                train_sampler.set_alpha(epoch_alpha, verbose=True)
                active_alpha = epoch_alpha
                print(f"[data] Epoch {epoch+1}: class_balance_alpha={epoch_alpha:.4f}")

            epoch_loss_sum = 0.0
            epoch_batches = 0
            pipe_profiler.start_epoch()

            pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            train_iter = iter(ds_train)

            for batch in range(steps_per_epoch):
                t_fetch0 = time.perf_counter()
                try:
                    x0, cond = next(train_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Training input pipeline exhausted unexpectedly before steps_per_epoch."
                    ) from exc
                data_wait_time_sec = time.perf_counter() - t_fetch0
                batch_info = (
                    train_sampler.consume_last_batch_info()
                    if hasattr(train_sampler, "consume_last_batch_info")
                    else None
                )

                t_step0 = time.perf_counter()
                step_index = tf.convert_to_tensor(global_step, dtype=tf.int32)
                loss = train_step(x0, cond, step_index=step_index)
                global_step += 1
                checkpoint.global_step.assign(global_step)
                loss_value = float(loss.numpy())
                compute_time_sec = time.perf_counter() - t_step0
                batch_size = int(np.shape(x0)[0])

                pipe_profiler.record(
                    batch_size=batch_size,
                    data_wait_time_sec=data_wait_time_sec,
                    compute_time_sec=compute_time_sec,
                    batch_info=batch_info,
                )

                if not np.isfinite(loss_value):
                    print(f"\n[warn] step {global_step}: non-finite loss ({loss_value}), update skipped")
                epoch_loss_sum += loss_value if np.isfinite(loss_value) else 0.0
                epoch_batches += 1

                if global_step % log_interval == 0:
                    postfix = {"loss": f"{loss_value:.4f}"}
                    running = pipe_profiler.running()
                    if running is not None:
                        postfix.update(
                            {
                                "wait_ms": f"{running['wait_ms']:.1f}",
                                "step_ms": f"{running['step_ms']:.1f}",
                                "samples/s": f"{running['samples_per_sec']:.1f}",
                            }
                        )
                    pbar.set_postfix(postfix)
                pbar.update(1)

            pbar.close()
            if hasattr(train_sampler, "rewind_unconsumed"):
                train_sampler.rewind_unconsumed()

            epoch_loss = epoch_loss_sum / max(1, epoch_batches)
            epoch_indices.append(epoch + 1)
            epoch_losses.append(epoch_loss)

            print(f"Epoch {epoch+1} mean loss: {epoch_loss:.6f}")
            pipe_summary = pipe_profiler.finish_epoch(epoch=epoch + 1, loader_info=train_loader_info)
            if pipe_summary is not None:
                print(
                    "[profile] "
                    f"epoch {epoch+1} ({pipe_summary.profile}): "
                    f"avg_batch_prep={pipe_summary.avg_batch_prep_time_sec:.4f}s, "
                    f"avg_wait={pipe_summary.avg_data_wait_time_sec:.4f}s, "
                    f"avg_compute={pipe_summary.avg_compute_time_sec:.4f}s, "
                    f"avg_step={pipe_summary.avg_step_time_sec:.4f}s, "
                    f"wait={pipe_summary.wait_fraction_pct:.1f}%, "
                    f"throughput={pipe_summary.throughput_samples_per_sec:.2f} samples/s"
                )
                if pipe_summary.avg_backend_load_time_sec is not None:
                    print(
                        "[profile] "
                        f"loader breakdown: backend_load={pipe_summary.avg_backend_load_time_sec:.4f}s, "
                        f"preprocess={pipe_summary.avg_preprocess_time_sec:.4f}s"
                    )
            val_loss = None
            val_loss_balanced = None
            val_loss_ema = None
            val_loss_balanced_ema = None
            if (epoch + 1) % val_every_epochs == 0:
                val_loss = evaluate_loss(
                    diffusion,
                    model,
                    ds_val_full,
                    val_steps=val_steps,
                    base_seed=val_base_seed,
                    split_seed_offset=0,
                )
                val_loss_balanced = evaluate_loss(
                    diffusion,
                    model,
                    ds_val_balanced,
                    val_steps=val_steps,
                    base_seed=val_base_seed,
                    split_seed_offset=10_000,
                )
                val_losses.append(val_loss)
                val_losses_balanced.append(val_loss_balanced)
                if use_ema and ema is not None:
                    with _ema_weights_applied(ema):
                        val_loss_ema = evaluate_loss(
                            diffusion,
                            model,
                            ds_val_full,
                            val_steps=val_steps,
                            base_seed=val_base_seed,
                            split_seed_offset=0,
                        )
                        val_loss_balanced_ema = evaluate_loss(
                            diffusion,
                            model,
                            ds_val_balanced,
                            val_steps=val_steps,
                            base_seed=val_base_seed,
                            split_seed_offset=10_000,
                        )
                    val_losses_ema.append(val_loss_ema)
                    val_losses_balanced_ema.append(val_loss_balanced_ema)
                val_summary_bits = [
                    f"raw_full={val_loss:.6f}",
                    f"raw_balanced={val_loss_balanced:.6f}",
                ]
                if val_loss_ema is not None:
                    val_summary_bits.extend(
                        [
                            f"ema_full={val_loss_ema:.6f}",
                            f"ema_balanced={val_loss_balanced_ema:.6f}",
                        ]
                    )
                print(f"  Validation: {', '.join(val_summary_bits)}")

            # ----- check for improvement (best) on FULL validation only -----
            raw_improved = val_loss is not None and (best_epoch_loss - val_loss > min_delta)
            ema_improved = (
                val_loss_ema is not None
                and use_ema
                and ema is not None
                and (best_epoch_loss_ema - val_loss_ema > min_delta)
            )
            balanced_improved = (
                val_loss_balanced is not None
                and (best_epoch_balanced - val_loss_balanced > min_delta)
            )
            balanced_ema_improved = (
                val_loss_balanced_ema is not None
                and use_ema
                and ema is not None
                and (best_epoch_balanced_ema - val_loss_balanced_ema > min_delta)
            )

            raw_snapshot_source: Path | None = None
            ema_snapshot_source: Path | None = None

            def _save_raw_weights(path: Path):
                model.save_weights(path)

            def _save_ema_weights(path: Path):
                if ema is None:
                    raise RuntimeError("EMA weights requested, but EMA is not enabled.")
                with _ema_weights_applied(ema):
                    model.save_weights(path)

            if save_last_weights:
                last_path = out_dir / f"weights_last.epoch_{epoch+1}.weights.h5"
                raw_snapshot_source = _publish_weight_snapshot(
                    raw_snapshot_source,
                    last_path,
                    save_fn=_save_raw_weights,
                    label="last weights",
                )
                _delete_other_last_weights(out_dir, keep=last_path)

            if use_ema and ema is not None:
                if save_ema_last_weights:
                    ema_last_path = out_dir / f"weights_ema_last.epoch_{epoch+1}.weights.h5"
                    ema_snapshot_source = _publish_weight_snapshot(
                        ema_snapshot_source,
                        ema_last_path,
                        save_fn=_save_ema_weights,
                        label="EMA last weights",
                    )
                    _delete_other_ema_last_weights(out_dir, keep=ema_last_path)

            if val_loss is not None:
                if raw_improved:
                    best_epoch_loss = val_loss
                    best_epoch_idx = epoch + 1

                    best_path = out_dir / "weights_best_val.weights.h5"
                    raw_snapshot_source = _publish_weight_snapshot(
                        raw_snapshot_source,
                        best_path,
                        save_fn=_save_raw_weights,
                        label="best validation weights",
                    )
                    print(
                        f"  New best model (epoch {best_epoch_idx}), "
                        f"saved to {best_path} (metric={best_epoch_loss:.6f})"
                    )

            if val_loss_ema is not None and use_ema and ema is not None:
                if ema_improved:
                    best_epoch_loss_ema = val_loss_ema
                    best_epoch_idx_ema = epoch + 1
                    best_ema_path = out_dir / "weights_ema_best_val.weights.h5"
                    ema_snapshot_source = _publish_weight_snapshot(
                        ema_snapshot_source,
                        best_ema_path,
                        save_fn=_save_ema_weights,
                        label="EMA best validation weights",
                    )
                    print(
                        f"  New best EMA model (epoch {best_epoch_idx_ema}), "
                        f"saved to {best_ema_path} (metric={best_epoch_loss_ema:.6f})"
                    )

            if val_loss is not None:
                if early_stopping_monitor == "ema":
                    monitor_improved = ema_improved
                    monitor_metric = val_loss_ema
                    monitor_best = best_epoch_loss_ema
                else:
                    monitor_improved = raw_improved
                    monitor_metric = val_loss
                    monitor_best = best_epoch_loss

                if monitor_metric is None:
                    raise RuntimeError(
                        f"Early-stopping monitor '{early_stopping_monitor}' has no validation metric for epoch {epoch+1}."
                    )

                if monitor_improved:
                    epochs_without_improvement = 0
                    print(
                        f"  Early stopping ({early_stopping_monitor}): improved to "
                        f"{monitor_metric:.6f}; patience reset"
                    )
                else:
                    epochs_without_improvement += 1
                    print(
                        f"  Early stopping ({early_stopping_monitor}): no improvement "
                        f"(current={monitor_metric:.6f}, best={monitor_best:.6f}) "
                        f"({epochs_without_improvement}/{patience_epochs})"
                    )
            else:
                print("  Skipping early-stopping update (validation not run this epoch).")

            # Also track a best model on the balanced_fixed validation metric (does NOT drive early stopping).
            if balanced_improved:
                best_epoch_balanced = val_loss_balanced
                best_bal_path = out_dir / "weights_best_balanced_val.weights.h5"
                raw_snapshot_source = _publish_weight_snapshot(
                    raw_snapshot_source,
                    best_bal_path,
                    save_fn=_save_raw_weights,
                    label="best balanced-validation weights",
                )
                print(
                    f"  New best BALANCED model, saved to {best_bal_path} "
                    f"(metric={best_epoch_balanced:.6f})"
                )
            if balanced_ema_improved:
                best_epoch_balanced_ema = val_loss_balanced_ema
                best_bal_ema_path = out_dir / "weights_ema_best_balanced_val.weights.h5"
                ema_snapshot_source = _publish_weight_snapshot(
                    ema_snapshot_source,
                    best_bal_ema_path,
                    save_fn=_save_ema_weights,
                    label="EMA best balanced-validation weights",
                )
                print(
                    f"  New best BALANCED EMA model, saved to {best_bal_ema_path} "
                    f"(metric={best_epoch_balanced_ema:.6f})"
                )

            checkpoint.epoch.assign(epoch + 1)
            ckpt_t0 = time.perf_counter()
            ckpt_path = checkpoint_manager.save(checkpoint_number=epoch + 1)
            ckpt_dt = time.perf_counter() - ckpt_t0
            print(f"  Saved training checkpoint to {ckpt_path} ({ckpt_dt:.1f}s)")

            # ----- persist run state (for resuming) -----
            _save_state(
                out_dir,
                {
                    "last_epoch": epoch + 1,  # completed epochs
                    "global_step": global_step,
                    "best_epoch": best_epoch_idx,
                    "best_epoch_loss": best_epoch_loss,
                    "best_epoch_balanced": best_epoch_balanced,
                    "best_epoch_ema": best_epoch_idx_ema,
                    "best_epoch_loss_ema": best_epoch_loss_ema,
                    "best_epoch_balanced_ema": best_epoch_balanced_ema,
                    "epochs_without_improvement": epochs_without_improvement,
                    "epoch_indices": epoch_indices,
                    "epoch_losses": epoch_losses,
                    "val_losses": val_losses,
                    "val_losses_balanced": val_losses_balanced,
                    "val_losses_ema": val_losses_ema,
                    "val_losses_balanced_ema": val_losses_balanced_ema,
                    "python_random_state": _encode_py_state(random.getstate()),
                    "numpy_random_state": _encode_py_state(np.random.get_state()),
                    "train_sampler_state": _encode_py_state(train_sampler.get_state()),
                },
            )

            if bool(epoch_preview_cfg["enabled"]) and ((epoch + 1) % int(epoch_preview_cfg["every_epochs"]) == 0):
                preview_use_ema = bool(epoch_preview_cfg["use_ema"]) and use_ema and ema is not None
                preview_t0 = time.perf_counter()
                if preview_use_ema:
                    with _ema_weights_applied(ema):
                        preview_root = _save_epoch_preview_samples(
                            cfg=cfg,
                            out_dir=out_dir,
                            epoch=epoch,
                            model=model,
                            diffusion=diffusion,
                            preview_cfg=epoch_preview_cfg,
                            using_ema_weights=True,
                        )
                else:
                    preview_root = _save_epoch_preview_samples(
                        cfg=cfg,
                        out_dir=out_dir,
                        epoch=epoch,
                        model=model,
                        diffusion=diffusion,
                        preview_cfg=epoch_preview_cfg,
                        using_ema_weights=False,
                    )
                preview_dt = time.perf_counter() - preview_t0
                preview_label = "EMA" if preview_use_ema else "raw"
                print(
                    f"  Saved epoch preview samples to {preview_root} "
                    f"using {preview_label} weights ({preview_dt:.1f}s)"
                )

            # ----- early stopping -----
            if es_enabled and epochs_without_improvement >= patience_epochs:
                stop_best_epoch = best_epoch_idx_ema if early_stopping_monitor == "ema" else best_epoch_idx
                stop_best_loss = best_epoch_loss_ema if early_stopping_monitor == "ema" else best_epoch_loss
                print(
                    f"\n Early stopping at epoch {epoch+1} "
                    f"(monitor={early_stopping_monitor}, best epoch: {stop_best_epoch}, "
                    f"best loss {stop_best_loss:.6f})"
                )
                break
    finally:
        if hasattr(train_sampler, "close"):
            train_sampler.close()
    

    
    # ----- post-training physics evaluation (always) -----
    heavy = True
    tag = "post_training"
    eval_weights = out_dir / "weights_best_val.weights.h5"
    if use_ema and ema is not None:
        ema_best = out_dir / "weights_ema_best_val.weights.h5"
        if ema_best.exists():
            eval_weights = ema_best

    if not eval_weights.exists():
        raise FileNotFoundError(
            f"Post-training evaluation requires best weights, but none were found at {eval_weights}."
        )

    model.load_weights(str(eval_weights))
    evaluator.run(model=model, diffusion=diffusion, out_dir=out_dir, tag=tag, heavy=heavy)
    print(f"  [eval] wrote eval/{tag} (heavy={heavy}) using {eval_weights.name}")

    return {
        "epoch": epoch_indices,
        "epoch_loss": epoch_losses,
        "best_epoch": best_epoch_idx,
        "best_epoch_loss": best_epoch_loss,
        "best_epoch_ema": best_epoch_idx_ema,
        "best_epoch_loss_ema": best_epoch_loss_ema,
        "early_stopping_monitor": early_stopping_monitor,
        "stopped_early": es_enabled and epochs_without_improvement >= patience_epochs,
    }
