# tc_diffusion/train_loop.py
import json
import glob
import math
import random
import base64
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import tensorflow as tf #type: ignore
from tensorflow import keras #type: ignore
from tqdm.auto import tqdm # type: ignore

from .data import _build_aug_policy, augment_batch_x_given_y, create_dataset
from .model_unet import build_unet
from .diffusion import Diffusion
from .evaluation.evaluator import TCEvaluator


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


def _decode_py_state(encoded: str):
    return pickle.loads(base64.b64decode(encoded.encode("ascii")))

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
    print(f"[train] Early stopping monitor: {early_stopping_monitor}")

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

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        epoch_alpha = resolve_train_alpha(cfg, epoch_idx=epoch, num_epochs=num_epochs)
        if active_alpha is None or abs(epoch_alpha - active_alpha) > 1e-12:
            train_sampler.set_alpha(epoch_alpha, verbose=True)
            active_alpha = epoch_alpha
            print(f"[data] Epoch {epoch+1}: class_balance_alpha={epoch_alpha:.4f}")

        epoch_loss_sum = 0.0
        epoch_batches = 0

        pbar = tqdm(ds_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch, (x0, cond) in enumerate(pbar):
            if batch >= steps_per_epoch:
                break
            step_index = tf.convert_to_tensor(global_step, dtype=tf.int32)
            loss = train_step(x0, cond, step_index=step_index)
            global_step += 1
            checkpoint.global_step.assign(global_step)
            loss_value = float(loss.numpy())
            if not np.isfinite(loss_value):
                print(f"\n[warn] step {global_step}: non-finite loss ({loss_value}), update skipped")
            epoch_loss_sum += loss_value if np.isfinite(loss_value) else 0.0
            epoch_batches += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        pbar.close()

        epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        epoch_indices.append(epoch + 1)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch+1} mean loss: {epoch_loss:.6f}")
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
            print(f"Epoch {epoch+1} val mean loss (full): {val_loss:.6f}")
            print(f"Epoch {epoch+1} val mean loss (balanced_fixed): {val_loss_balanced:.6f}")
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
                print(f"Epoch {epoch+1} EMA val mean loss (full): {val_loss_ema:.6f}")
                print(f"Epoch {epoch+1} EMA val mean loss (balanced_fixed): {val_loss_balanced_ema:.6f}")

        # ----- ALWAYS save "last" -----
        last_path = out_dir / f"weights_last.epoch_{epoch+1}.weights.h5"
        model.save_weights(last_path)
        _delete_other_last_weights(out_dir, keep=last_path)
        if use_ema and ema is not None:
            ema_last_path = out_dir / f"weights_ema_last.epoch_{epoch+1}.weights.h5"
            with _ema_weights_applied(ema):
                model.save_weights(ema_last_path)
            _delete_other_ema_last_weights(out_dir, keep=ema_last_path)
            print(f"  Saved EMA last weights to {ema_last_path}")
        print(f"  Saved last weights to {last_path}")

        # ----- check for improvement (best) on FULL validation only -----
        raw_improved = val_loss is not None and (best_epoch_loss - val_loss > min_delta)
        ema_improved = (
            val_loss_ema is not None
            and use_ema
            and ema is not None
            and (best_epoch_loss_ema - val_loss_ema > min_delta)
        )

        if val_loss is not None:
            if raw_improved:
                best_epoch_loss = val_loss
                best_epoch_idx = epoch + 1

                best_path = out_dir / "weights_best_val.weights.h5"
                model.save_weights(best_path)
                print(f"  New best model (epoch {best_epoch_idx}), saved to {best_path} (metric={best_epoch_loss:.6f})")

        if val_loss_ema is not None and use_ema and ema is not None:
            if ema_improved:
                best_epoch_loss_ema = val_loss_ema
                best_epoch_idx_ema = epoch + 1
                best_ema_path = out_dir / "weights_ema_best_val.weights.h5"
                with _ema_weights_applied(ema):
                    model.save_weights(best_ema_path)
                print(
                    f"  New best EMA model (epoch {best_epoch_idx_ema}), "
                    f"saved to {best_ema_path} (metric={best_epoch_loss_ema:.6f})"
                )

        if val_loss is not None:
            if early_stopping_monitor == "ema":
                monitor_improved = ema_improved
                monitor_metric = val_loss_ema
            else:
                monitor_improved = raw_improved
                monitor_metric = val_loss

            if monitor_metric is None:
                raise RuntimeError(
                    f"Early-stopping monitor '{early_stopping_monitor}' has no validation metric for epoch {epoch+1}."
                )

            if monitor_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    f"  No significant improvement on early-stopping monitor "
                    f"({early_stopping_monitor}) ({epochs_without_improvement}/{patience_epochs})"
                )
        else:
            print("  Skipping early-stopping update (validation not run this epoch).")

        # Also track a best model on the balanced_fixed validation metric (does NOT drive early stopping).
        if val_loss_balanced is not None and (best_epoch_balanced - val_loss_balanced > min_delta):
            best_epoch_balanced = val_loss_balanced
            best_bal_path = out_dir / "weights_best_balanced_val.weights.h5"
            model.save_weights(best_bal_path)
            print(f"  New best BALANCED model, saved to {best_bal_path} (metric={best_epoch_balanced:.6f})")
        if (
            val_loss_balanced_ema is not None
            and use_ema
            and ema is not None
            and (best_epoch_balanced_ema - val_loss_balanced_ema > min_delta)
        ):
            best_epoch_balanced_ema = val_loss_balanced_ema
            best_bal_ema_path = out_dir / "weights_ema_best_balanced_val.weights.h5"
            with _ema_weights_applied(ema):
                model.save_weights(best_bal_ema_path)
            print(
                f"  New best BALANCED EMA model, saved to {best_bal_ema_path} "
                f"(metric={best_epoch_balanced_ema:.6f})"
            )

        checkpoint.epoch.assign(epoch + 1)
        ckpt_path = checkpoint_manager.save(checkpoint_number=epoch + 1)
        print(f"  Saved training checkpoint to {ckpt_path}")

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
