import json
import random
import time
from bisect import bisect_right
from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import xarray as xr

def _build_aug_policy(num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return classwise augmentation parameters sized to num_classes."""
    if int(num_classes) <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    # Anchor policy for the canonical 6 SS classes; resampled for other counts.
    p_anchor = np.array([0.15, 0.20, 0.35, 0.60, 0.75, 0.85], dtype=np.float32)
    shift_anchor = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)

    if int(num_classes) == int(p_anchor.shape[0]):
        p_aug = p_anchor
        max_shift = shift_anchor
    else:
        src = np.linspace(0.0, 1.0, num=p_anchor.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=int(num_classes), dtype=np.float32)
        p_aug = np.interp(dst, src, p_anchor).astype(np.float32)
        max_shift = np.rint(
            np.interp(dst, src, shift_anchor.astype(np.float32))
        ).astype(np.int32)

    max_shift = np.maximum(max_shift, 0)
    return (
        tf.constant(p_aug, dtype=tf.float32),
        tf.constant(max_shift, dtype=tf.int32),
    )


def _seed_with_offset(seed: tf.Tensor, offset: int) -> tf.Tensor:
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    return tf.stack([seed[0], seed[1] + tf.cast(offset, tf.int32)])

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def rot90_only(x, seed: tf.Tensor | None = None):
    # x: [H, W, C] or [H, W]
    if seed is None:
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    else:
        k = tf.random.stateless_uniform(
            [],
            seed=tf.convert_to_tensor(seed, dtype=tf.int32),
            minval=0,
            maxval=4,
            dtype=tf.int32,
        )
    return tf.image.rot90(x, k)

def random_shift_reflect(x, max_shift, seed: tf.Tensor | None = None):
    # max_shift: int scalar tensor
    max_shift = tf.cast(max_shift, tf.int32)

    def _do_shift():
        h = tf.shape(x)[0]
        w = tf.shape(x)[1]

        if seed is None:
            dy = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            dx = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
        else:
            seed_i = tf.convert_to_tensor(seed, dtype=tf.int32)
            dy = tf.random.stateless_uniform(
                [],
                seed=_seed_with_offset(seed_i, 0),
                minval=-max_shift,
                maxval=max_shift + 1,
                dtype=tf.int32,
            )
            dx = tf.random.stateless_uniform(
                [],
                seed=_seed_with_offset(seed_i, 1),
                minval=-max_shift,
                maxval=max_shift + 1,
                dtype=tf.int32,
            )

        # reflect pad enough to allow cropping after shift
        pad_y = max_shift
        pad_x = max_shift
        x_pad = tf.pad(
            x,
            paddings=[[pad_y, pad_y], [pad_x, pad_x], [0, 0]]
            if x.shape.rank == 3
            else [[pad_y, pad_y], [pad_x, pad_x]],
            mode="REFLECT",
        )

        # crop with offset
        y0 = pad_y + dy
        x0 = pad_x + dx
        if x.shape.rank == 3:
            return tf.image.crop_to_bounding_box(x_pad, y0, x0, h, w)

        x_pad3 = x_pad[..., None]
        out = tf.image.crop_to_bounding_box(x_pad3, y0, x0, h, w)
        return tf.squeeze(out, axis=-1)

    return tf.cond(max_shift <= 0, lambda: x, _do_shift)

def augment_x_given_y(
    x,
    y,
    p_aug: tf.Tensor,
    max_shift_per_class: tf.Tensor,
    seed: tf.Tensor | None = None,
):
    """
    x: [H,W,1] (or [H,W]) normalized already (e.g., to [-1,1])
    y: int label
    """
    y = tf.cast(y, tf.int32)

    p = tf.gather(p_aug, y)
    max_shift = tf.gather(max_shift_per_class, y)

    def do_aug():
        rot_seed = None if seed is None else _seed_with_offset(seed, 1)
        shift_seed = None if seed is None else _seed_with_offset(seed, 2)
        x1 = rot90_only(x, seed=rot_seed)
        x2 = random_shift_reflect(x1, max_shift=max_shift, seed=shift_seed)
        return x2

    if seed is None:
        apply_aug = tf.random.uniform([]) < p
    else:
        apply_aug = tf.random.stateless_uniform(
            [],
            seed=tf.convert_to_tensor(seed, dtype=tf.int32),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        ) < p
    return tf.cond(apply_aug, do_aug, lambda: x)


def augment_batch_x_given_y(
    x,
    y,
    p_aug: tf.Tensor,
    max_shift_per_class: tf.Tensor,
    base_seed: tf.Tensor,
):
    """Stateless batch augmentation keyed by an explicit per-step seed."""
    base_seed = tf.convert_to_tensor(base_seed, dtype=tf.int32)
    batch_size = tf.shape(x)[0]
    sample_idx = tf.range(batch_size, dtype=tf.int32)
    sample_seeds = tf.stack(
        [
            tf.fill([batch_size], base_seed[0]),
            base_seed[1] + sample_idx,
        ],
        axis=1,
    )

    def _augment_one(args):
        x_i, y_i, seed_i = args
        return augment_x_given_y(
            x_i,
            y_i,
            p_aug=p_aug,
            max_shift_per_class=max_shift_per_class,
            seed=seed_i,
        )

    channel_dim = x.shape[-1] if x.shape.rank == 4 else None
    out_spec = tf.TensorSpec(shape=(None, None, channel_dim), dtype=x.dtype)
    return tf.map_fn(
        _augment_one,
        (x, y, sample_seeds),
        fn_output_signature=out_spec,
        parallel_iterations=1,
    )

def preprocess_bt(bt: np.ndarray, bt_range: Tuple[float, float]) -> np.ndarray:
    bt = bt.astype(np.float32)
    bt = np.nan_to_num(bt, nan=0.0, posinf=0.0, neginf=0.0)

    bt_min, bt_max = bt_range
    bt = np.clip(bt, bt_min, bt_max)

    bt = (bt - bt_min) / (bt_max - bt_min)  # [0,1]
    bt = bt * 2.0 - 1.0                     # [-1,1]
    return bt


def _ensure_channel_axis(bt: np.ndarray) -> np.ndarray:
    """Normalize train/eval arrays to [..., H, W, 1] or [H, W, 1]."""
    if bt.ndim == 2:
        return bt[..., None]
    if bt.ndim == 3:
        if bt.shape[-1] == 1:
            return bt
        return bt[..., None]
    return bt

def ss_class_midpoint_kt(ss_cat: int) -> float:
    """Representative wind speed [kt] for each SS class."""
    cls = int(ss_cat)
    # Midpoints in the WMO bins used by scripts/create_dataset_index.py.
    if cls == 0:
        return 49.0   # 35-63
    if cls == 1:
        return 73.0   # 64-82
    if cls == 2:
        return 89.0   # 83-95
    if cls == 3:
        return 104.0  # 96-112
    if cls == 4:
        return 124.5  # 113-136
    return 145.0      # >136


def load_dataset_index(
    index_path: Path,
    return_sample_meta: bool = False,
) -> Dict[int, List[str]] | Tuple[Dict[int, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Load dataset_index.json and return class -> list of relative file paths.
    """
    with open(index_path, "r") as f:
        index = json.load(f)

    class_to_files = {
        int(k): v for k, v in index["classes"].items()
    }
    if not return_sample_meta:
        return class_to_files

    raw_samples = index.get("samples", {})
    sample_meta: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_samples, dict):
        for rel_path, meta in raw_samples.items():
            if not isinstance(meta, dict):
                continue
            entry: Dict[str, Any] = {}
            if "ss_cat" in meta:
                try:
                    entry["ss_cat"] = int(meta["ss_cat"])
                except Exception:
                    pass
            if "wmo_wind_kt" in meta:
                try:
                    w = float(meta["wmo_wind_kt"])
                    if np.isfinite(w):
                        entry["wmo_wind_kt"] = w
                except Exception:
                    pass
            if entry:
                sample_meta[str(rel_path)] = entry

    return class_to_files, sample_meta


def build_relpath_to_wind_kt(
    rel_paths: List[str],
    labels: List[int],
    sample_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], int]:
    """Build rel_path -> wind[kt] map; fall back to class midpoint when metadata is missing."""
    out: Dict[str, float] = {}
    n_fallback = 0
    for rel_path, y in zip(rel_paths, labels):
        meta = sample_meta.get(rel_path)
        if meta is not None and "wmo_wind_kt" in meta:
            wind_kt = float(meta["wmo_wind_kt"])
            if np.isfinite(wind_kt):
                out[rel_path] = wind_kt
                continue
        out[rel_path] = ss_class_midpoint_kt(int(y))
        n_fallback += 1
    return out, n_fallback


class BTDataBackend:
    """Backend that returns raw BT arrays by dataset-relative path."""

    name = "base"
    supports_parallel_reads = False
    supports_vectorized_batch_loads = False

    def load_bt(self, rel_path: str) -> np.ndarray:
        raise NotImplementedError

    def load_bt_batch(self, rel_paths: Sequence[str]) -> np.ndarray:
        if len(rel_paths) == 0:
            raise ValueError("load_bt_batch requires at least one rel_path.")
        rows = [self.load_bt(str(rel_path)) for rel_path in rel_paths]
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


class NetCDFBTBackend(BTDataBackend):
    name = "netcdf"
    supports_parallel_reads = False

    def __init__(self, data_root: Path, engine: str = "netcdf4"):
        self.data_root = Path(data_root)
        self.engine = str(engine)

    def load_bt(self, rel_path: str) -> np.ndarray:
        nc_path = self.data_root / rel_path
        with xr.open_dataset(nc_path, engine=self.engine) as ds:
            return ds["bt"].values.astype(np.float32)


@dataclass(frozen=True)
class PackedShardInfo:
    path: str
    start: int
    stop: int


class PackedMemmapBackend(BTDataBackend):
    """Read raw BT arrays from sharded .npy memmaps produced offline."""

    name = "packed_memmap"
    supports_parallel_reads = True
    supports_vectorized_batch_loads = True

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Packed manifest not found: {self.manifest_path}")

        with self.manifest_path.open("r") as f:
            manifest = json.load(f)

        backend = str(manifest.get("backend", "")).lower()
        if backend != self.name:
            raise ValueError(
                f"Packed manifest backend mismatch: expected '{self.name}', got '{backend or 'missing'}'"
            )

        self.root = self.manifest_path.parent
        self.sample_shape = tuple(int(v) for v in manifest["sample_shape"])
        self.dtype = np.dtype(manifest["dtype"])
        self.num_samples = int(manifest["num_samples"])

        rel_paths_file = self.root / manifest["rel_paths_file"]
        with rel_paths_file.open("r") as f:
            self.rel_paths = [line.strip() for line in f if line.strip()]
        if len(self.rel_paths) != self.num_samples:
            raise ValueError(
                f"Packed rel_paths count mismatch: expected {self.num_samples}, found {len(self.rel_paths)}"
            )

        self.path_to_index = {rel_path: idx for idx, rel_path in enumerate(self.rel_paths)}
        if len(self.path_to_index) != len(self.rel_paths):
            raise ValueError("Packed rel_paths contain duplicates; expected unique dataset-relative paths.")

        labels_file = self.root / manifest["labels_file"]
        self.labels = np.load(labels_file, mmap_mode="r")
        if int(self.labels.shape[0]) != self.num_samples:
            raise ValueError(
                f"Packed labels count mismatch: expected {self.num_samples}, found {self.labels.shape[0]}"
            )

        self.shards = [
            PackedShardInfo(
                path=str(shard["path"]),
                start=int(shard["start"]),
                stop=int(shard["stop"]),
            )
            for shard in manifest["shards"]
        ]
        if not self.shards:
            raise ValueError("Packed manifest contains no shards.")
        prev_stop = 0
        for shard in self.shards:
            if shard.start != prev_stop:
                raise ValueError(
                    f"Packed shard layout must be contiguous; expected start={prev_stop}, found {shard.start}"
                )
            if shard.stop <= shard.start:
                raise ValueError(
                    f"Packed shard has invalid bounds [{shard.start}, {shard.stop}) for path {shard.path}"
                )
            prev_stop = shard.stop
        if prev_stop != self.num_samples:
            raise ValueError(
                f"Packed shard coverage mismatch: last shard stops at {prev_stop}, expected {self.num_samples}"
            )

        self._shard_starts = [shard.start for shard in self.shards]
        self._shard_memmaps = [self._open_shard_memmap(shard_idx) for shard_idx in range(len(self.shards))]

    def _resolve_shard(self, global_index: int) -> Tuple[int, PackedShardInfo]:
        if global_index < 0 or global_index >= self.num_samples:
            raise IndexError(f"Packed sample index out of range: {global_index}")

        shard_idx = bisect_right(self._shard_starts, int(global_index)) - 1
        if shard_idx < 0:
            raise IndexError(f"Could not resolve shard for packed sample index {global_index}")

        shard = self.shards[shard_idx]
        if global_index >= shard.stop:
            raise IndexError(
                f"Packed sample index {global_index} falls outside shard bounds [{shard.start}, {shard.stop})"
            )
        return shard_idx, shard

    def _open_shard_memmap(self, shard_idx: int) -> np.ndarray:
        shard = self.shards[shard_idx]
        shard_path = self.root / shard.path
        mm = np.load(shard_path, mmap_mode="r")

        expected_rows = shard.stop - shard.start
        expected_shape = (expected_rows, *self.sample_shape)
        if tuple(mm.shape) != expected_shape:
            raise ValueError(
                f"Packed shard shape mismatch for {shard_path}: expected {expected_shape}, found {tuple(mm.shape)}"
            )
        if np.dtype(mm.dtype) != self.dtype:
            raise ValueError(
                f"Packed shard dtype mismatch for {shard_path}: expected {self.dtype}, found {mm.dtype}"
            )
        return mm

    def load_bt_by_index(self, global_index: int) -> np.ndarray:
        shard_idx, shard = self._resolve_shard(global_index)
        mm = self._shard_memmaps[shard_idx]
        local_index = int(global_index) - shard.start
        return np.asarray(mm[local_index], dtype=np.float32)

    def _copy_rows_into_output(
        self,
        shard_idx: int,
        items: Sequence[Tuple[int, int]],
        out: np.ndarray,
    ):
        mm = self._shard_memmaps[shard_idx]
        out_pos = np.fromiter((item[0] for item in items), dtype=np.int64, count=len(items))
        local_idx = np.fromiter((item[1] for item in items), dtype=np.int64, count=len(items))
        out[out_pos] = np.asarray(mm[local_idx], dtype=np.float32)

    def load_bt_batch_by_index(
        self,
        global_indices: Sequence[int],
        *,
        sort_within_shard: bool = False,
        max_samples_per_read: int | None = None,
    ) -> np.ndarray:
        indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("load_bt_batch_by_index requires at least one index.")
        if max_samples_per_read is not None:
            max_samples_per_read = int(max_samples_per_read)
            if max_samples_per_read <= 0:
                raise ValueError(
                    f"max_samples_per_read must be > 0 when set, got {max_samples_per_read}"
                )

        out = np.empty((indices.shape[0], *self.sample_shape), dtype=np.float32)
        by_shard: Dict[int, List[Tuple[int, int]]] = {}
        for out_pos, global_index in enumerate(indices.tolist()):
            shard_idx, shard = self._resolve_shard(int(global_index))
            local_index = int(global_index) - shard.start
            by_shard.setdefault(shard_idx, []).append((out_pos, local_index))

        for shard_idx, items in by_shard.items():
            read_items = (
                sorted(items, key=lambda item: item[1])
                if sort_within_shard
                else list(items)
            )
            if max_samples_per_read is None or len(read_items) <= max_samples_per_read:
                self._copy_rows_into_output(shard_idx, read_items, out)
                continue

            for start in range(0, len(read_items), max_samples_per_read):
                chunk = read_items[start : start + max_samples_per_read]
                self._copy_rows_into_output(shard_idx, chunk, out)

        return out

    def warmup_shards(self, rows_per_shard: int = 1):
        rows_per_shard = max(1, int(rows_per_shard))
        for mm in self._shard_memmaps:
            rows = min(rows_per_shard, int(mm.shape[0]))
            if rows <= 0:
                continue
            np.asarray(mm[:rows], dtype=np.float32)

    def load_bt(self, rel_path: str) -> np.ndarray:
        try:
            global_index = self.path_to_index[rel_path]
        except KeyError as exc:
            raise KeyError(f"Packed manifest does not contain rel_path '{rel_path}'") from exc
        return self.load_bt_by_index(global_index)

    def load_bt_batch(self, rel_paths: Sequence[str]) -> np.ndarray:
        indices = []
        for rel_path in rel_paths:
            try:
                indices.append(self.path_to_index[str(rel_path)])
            except KeyError as exc:
                raise KeyError(f"Packed manifest does not contain rel_path '{rel_path}'") from exc
        return self.load_bt_batch_by_index(indices)


def build_data_backend(data_cfg) -> BTDataBackend:
    backend_name = str(data_cfg.get("backend", "netcdf")).lower()
    if backend_name == "netcdf":
        engine = str(data_cfg.get("netcdf_engine", data_cfg.get("eval_engine", "netcdf4")))
        return NetCDFBTBackend(Path(data_cfg["data_root"]), engine=engine)

    if backend_name == "packed_memmap":
        manifest_path = data_cfg.get("packed_manifest")
        if not manifest_path:
            raise ValueError("data.packed_manifest must be set when data.backend='packed_memmap'")
        return PackedMemmapBackend(Path(manifest_path))

    raise ValueError(
        f"Unsupported data backend '{backend_name}'. Expected one of: 'netcdf', 'packed_memmap'."
    )


def resolve_eval_num_parallel_calls(data_cfg, backend: BTDataBackend):
    raw = data_cfg.get("eval_num_parallel_calls")
    if raw is None:
        return tf.data.AUTOTUNE if backend.supports_parallel_reads else 1

    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"auto", "autotune"}:
            return tf.data.AUTOTUNE

    num_parallel_calls = int(raw)
    if num_parallel_calls <= 0:
        raise ValueError(f"data.eval_num_parallel_calls must be > 0 or 'autotune', got {raw!r}")
    return num_parallel_calls


def compute_class_sampling_probs(
    class_to_files: Dict[int, List[str]],
    alpha: float,
) -> Dict[int, float]:
    """
    Compute class sampling probabilities using power-law reweighting:
        p_train(c) ∝ p_empirical(c)^alpha
    """
    counts = {c: len(v) for c, v in class_to_files.items()}
    counts = {c: n for c, n in counts.items() if n > 0}
    if not counts:
        raise RuntimeError("No non-empty classes available for sampling.")
    total = sum(counts.values())

    # empirical distribution
    p_emp = {c: counts[c] / total for c in counts}

    # reweighted distribution
    weights = {c: p_emp[c] ** alpha for c in p_emp}
    Z = sum(weights.values())

    probs = {c: weights[c] / Z for c in weights}
    return probs

def load_split_file_set(split_dir: Path, split: str) -> set[str]:
    fp = split_dir / f"files_{split}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"Split file list not found: {fp}")
    with fp.open("r") as f:
        return set(line.strip() for line in f if line.strip())


def _invert_index(class_to_files: Dict[int, List[str]]) -> Dict[str, int]:
    """Map relative file path -> class id."""
    out: Dict[str, int] = {}
    for c, paths in class_to_files.items():
        for p in paths:
            out[p] = int(c)
    return out


def _build_split_pairs(
    class_to_files: Dict[int, List[str]],
    allowed: set[str],
) -> Tuple[List[str], List[int]]:
    """Return (rel_paths, labels) for a split, in a deterministic order."""
    file_to_class = _invert_index(class_to_files)
    rel_paths = sorted([p for p in allowed if p in file_to_class])
    labels = [file_to_class[p] for p in rel_paths]
    return rel_paths, labels


def _fixed_balanced_subset(
    rel_paths: List[str],
    labels: List[int],
    per_class: int,
    seed: int,
) -> Tuple[List[str], List[int]]:
    """Pick up to `per_class` items per class, deterministically via seed."""
    rng = random.Random(int(seed))
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for p, y in zip(rel_paths, labels):
        by_class.setdefault(int(y), []).append((p, int(y)))

    picked: List[Tuple[str, int]] = []
    for c in sorted(by_class.keys()):
        items = list(by_class[c])
        rng.shuffle(items)
        picked.extend(items[: max(0, int(per_class))])

    # Deterministic final order (important for reproducible val curves)
    picked.sort(key=lambda t: t[0])
    out_paths = [p for p, _ in picked]
    out_labels = [y for _, y in picked]
    return out_paths, out_labels


def _tf_load_one_sample(
    backend: BTDataBackend,
    bt_range: Tuple[float, float],
    use_wind_speed: bool = False,
    rel_path_to_wind_kt: Dict[str, float] | None = None,
):
    """Factory returning a tf.data map fn that loads & preprocesses one sample."""
    bt_min, bt_max = float(bt_range[0]), float(bt_range[1])
    wind_lookup = dict(rel_path_to_wind_kt or {})

    def _decode_path(rel_path_obj) -> str:
        # rel_path_obj can be bytes, np.bytes_, or a 0-d numpy array
        if isinstance(rel_path_obj, (bytes, bytearray)):
            return rel_path_obj.decode("utf-8")
        if hasattr(rel_path_obj, "dtype") and rel_path_obj.dtype.kind in ("S", "O"):
            # 0-d array or object
            rel_path_obj = rel_path_obj.item()
            if isinstance(rel_path_obj, (bytes, bytearray)):
                return rel_path_obj.decode("utf-8")
            return str(rel_path_obj)
        return str(rel_path_obj)

    def _py_load(rel_path_obj, label):
        rel_path = _decode_path(rel_path_obj)
        y = int(label)
        bt = backend.load_bt(rel_path)
        bt = preprocess_bt(bt, (bt_min, bt_max))
        bt = _ensure_channel_axis(bt)
        if not use_wind_speed:
            return bt, np.int32(y)

        wind_kt = float(wind_lookup.get(rel_path, ss_class_midpoint_kt(y)))
        return bt, np.int32(y), np.float32(wind_kt)

    def _tf_map(rel_path, label):
        if not use_wind_speed:
            bt, y = tf.numpy_function(
                func=_py_load,
                inp=[rel_path, label],
                Tout=[tf.float32, tf.int32],
            )
            bt.set_shape([None, None, 1])
            y.set_shape([])
            return bt, y

        bt, y, wind_kt = tf.numpy_function(
            func=_py_load,
            inp=[rel_path, label],
            Tout=[tf.float32, tf.int32, tf.float32],
        )
        bt.set_shape([None, None, 1])
        y.set_shape([])
        wind_kt.set_shape([])
        cond = {"ss_cat": y, "wind_kt": wind_kt}
        return bt, cond

    return _tf_map


def _create_finite_eval_dataset(
    *,
    backend: BTDataBackend,
    rel_paths: List[str],
    labels: List[int],
    batch_size: int,
    seed: int,
    bt_range: Tuple[float, float],
    shuffle: bool,
    use_wind_speed: bool = False,
    rel_path_to_wind_kt: Dict[str, float] | None = None,
    num_parallel_calls=1,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(rel_paths, dtype=tf.string),
            tf.constant(labels, dtype=tf.int32),
        )
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(rel_paths),
            seed=seed,
            reshuffle_each_iteration=False,
        )

    map_fn = _tf_load_one_sample(
        backend,
        bt_range,
        use_wind_speed=use_wind_speed,
        rel_path_to_wind_kt=rel_path_to_wind_kt,
    )
    ds = ds.map(map_fn, num_parallel_calls=int(num_parallel_calls), deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE if backend.supports_parallel_reads else 1)

    opts = tf.data.Options()
    opts.deterministic = True
    if not backend.supports_parallel_reads:
        # netCDF4/HDF5 open/read paths are safer single-threaded.
        opts.threading.private_threadpool_size = 1
        opts.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(opts)

    return ds


@dataclass(frozen=True)
class SampledTrainBatchRequest:
    state_before_batch: Any
    labels: np.ndarray
    rel_paths: Tuple[str, ...] | None = None
    global_indices: np.ndarray | None = None
    wind_kt: np.ndarray | None = None


@dataclass
class PendingTrainBatch:
    request: SampledTrainBatchRequest
    payload: Tuple[np.ndarray, Any, Dict[str, float]] | Future


# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------

class BalancedTCGenerator:
    """
    Infinite generator yielding balanced TC samples.

    Yields:
        bt: (H, W, 1) float32
        cond: int32 (SS category)
    """

    def __init__(
        self,
        backend: BTDataBackend,
        class_to_files: Dict[int, List[str]],
        class_probs: Dict[int, float],
        seed: int = 42,
        bt_range: Tuple[float, float] = (117.0, 348.0),
        use_wind_speed: bool = False,
        rel_path_to_wind_kt: Dict[str, float] | None = None,
    ):
        self.backend = backend
        self.class_to_files = {int(c): list(v) for c, v in class_to_files.items()}
        self.class_probs = {int(c): float(p) for c, p in class_probs.items()}
        self.bt_range = bt_range
        self.use_wind_speed = bool(use_wind_speed)
        self.rel_path_to_wind_kt = dict(rel_path_to_wind_kt or {})
        self.alpha = None

        self.classes: List[int] = []
        self.probs = np.zeros((0,), dtype=np.float64)
        self.set_class_probs(self.class_probs, verbose=True)

        self.rng = random.Random(seed)
        self.pipeline_name = "legacy_sample_generator"
        self.profile_name = "legacy"
        self.prefetch_batches = 0

    def describe(self) -> Dict[str, Any]:
        return {
            "profile": self.profile_name,
            "loader": self.pipeline_name,
            "backend": self.backend.name,
            "prefetch_batches": int(self.prefetch_batches),
            "supports_vectorized_batch_loads": bool(
                getattr(self.backend, "supports_vectorized_batch_loads", False)
            ),
        }

    def set_class_probs(self, class_probs: Dict[int, float], verbose: bool = False):
        """Update class sampling probabilities for curriculum schedules."""
        self.class_probs = {int(c): float(p) for c, p in class_probs.items()}
        self.classes = [
            c for c in sorted(self.class_to_files.keys())
            if (len(self.class_to_files[c]) > 0) and (float(self.class_probs.get(c, 0.0)) > 0.0)
        ]
        if not self.classes:
            raise RuntimeError("No valid classes available for balanced sampling.")

        probs = np.array([float(self.class_probs[c]) for c in self.classes], dtype=np.float64)
        psum = float(probs.sum())
        if psum <= 0.0:
            raise RuntimeError("Class sampling probabilities have non-positive sum.")
        self.probs = probs / psum

        if verbose:
            print("\n[data] Class-balanced sampling probabilities:")
            for c in sorted(self.class_probs):
                print(
                    f"  class {c}: p = {self.class_probs[c]:.4f} "
                    f"(n = {len(self.class_to_files.get(c, []))})"
                )
            print()

    def set_alpha(self, alpha: float, verbose: bool = False):
        """Recompute class probabilities from alpha and apply them."""
        self.alpha = float(alpha)
        class_probs = compute_class_sampling_probs(self.class_to_files, self.alpha)
        self.set_class_probs(class_probs, verbose=verbose)

    def get_state(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "alpha": self.alpha,
            "class_probs": dict(self.class_probs),
        }

    def set_state(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            raise ValueError("BalancedTCGenerator.set_state expects a dict state payload.")

        class_probs = state.get("class_probs")
        if class_probs is not None:
            self.set_class_probs(
                {int(c): float(p) for c, p in class_probs.items()},
                verbose=False,
            )
        alpha = state.get("alpha")
        self.alpha = None if alpha is None else float(alpha)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def rewind_unconsumed(self) -> int:
        return 0

    def consume_last_batch_info(self) -> Dict[str, float] | None:
        return None

    def close(self):
        return None

    def _sample_one(self) -> Tuple[int, str]:
        cls = self.rng.choices(self.classes, weights=self.probs, k=1)[0]
        rel_path = self.rng.choice(self.class_to_files[cls])
        return int(cls), str(rel_path)

    def _build_cond_value(self, cls: int, rel_path: str) -> np.int32 | Dict[str, np.generic]:
        if not self.use_wind_speed:
            return np.int32(cls)

        wind_kt = float(self.rel_path_to_wind_kt.get(rel_path, ss_class_midpoint_kt(cls)))
        return {
            "ss_cat": np.int32(cls),
            "wind_kt": np.float32(wind_kt),
        }

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.int32] | Tuple[np.ndarray, Dict[str, np.generic]]]:
        while True:
            cls, rel_path = self._sample_one()

            # 3) load BT and normalize
            bt = self.backend.load_bt(rel_path)
            bt = preprocess_bt(bt, self.bt_range)
            bt = _ensure_channel_axis(bt)
            yield bt, self._build_cond_value(cls, rel_path)


class BalancedTCBatchGenerator(BalancedTCGenerator):
    """Infinite generator yielding already-formed balanced batches."""

    def __init__(
        self,
        backend: BTDataBackend,
        class_to_files: Dict[int, List[str]],
        class_probs: Dict[int, float],
        seed: int = 42,
        bt_range: Tuple[float, float] = (117.0, 348.0),
        use_wind_speed: bool = False,
        rel_path_to_wind_kt: Dict[str, float] | None = None,
        batch_size: int = 8,
        prefetch_batches: int = 1,
    ):
        super().__init__(
            backend=backend,
            class_to_files=class_to_files,
            class_probs=class_probs,
            seed=seed,
            bt_range=bt_range,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
        )
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if int(prefetch_batches) < 0:
            raise ValueError(f"prefetch_batches must be >= 0, got {prefetch_batches}")

        self.batch_size = int(batch_size)
        self.prefetch_batches = int(prefetch_batches)
        self.pipeline_name = "balanced_batch_generator"
        self.profile_name = "local_5090"
        self._pending: deque[PendingTrainBatch] = deque()
        self._last_batch_info: Dict[str, float] | None = None
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max(1, self.prefetch_batches),
                thread_name_prefix="tc-train-batch",
            )
            if self.prefetch_batches > 0
            else None
        )

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        if self._pending:
            state["rng_state"] = self._pending[0].request.state_before_batch
            state["pending_prefetch_batches"] = len(self._pending)
        return state

    def set_state(self, state: Dict[str, Any]):
        self.rewind_unconsumed()
        super().set_state(state)
        self._last_batch_info = None

    def consume_last_batch_info(self) -> Dict[str, float] | None:
        info = self._last_batch_info
        self._last_batch_info = None
        return info

    def _make_batch_request(self) -> SampledTrainBatchRequest:
        state_before = self.rng.getstate()
        rel_paths: List[str] = []
        labels = np.empty((self.batch_size,), dtype=np.int32)
        wind_arr = np.empty((self.batch_size,), dtype=np.float32) if self.use_wind_speed else None

        for batch_idx in range(self.batch_size):
            # Keep the legacy law exactly: sample each example independently as
            # (class -> file within class), then materialize the whole batch.
            cls, rel_path = self._sample_one()
            rel_paths.append(rel_path)
            labels[batch_idx] = np.int32(cls)
            if wind_arr is not None:
                wind_arr[batch_idx] = np.float32(
                    self.rel_path_to_wind_kt.get(rel_path, ss_class_midpoint_kt(cls))
                )

        return SampledTrainBatchRequest(
            state_before_batch=state_before,
            rel_paths=tuple(rel_paths),
            labels=labels,
            wind_kt=wind_arr,
        )

    def _materialize_batch(
        self,
        request: SampledTrainBatchRequest,
    ) -> Tuple[np.ndarray, Any, Dict[str, float]]:
        if request.rel_paths is None:
            raise ValueError("BalancedTCBatchGenerator expects rel_paths in the batch request.")
        t0 = time.perf_counter()
        bt = self.backend.load_bt_batch(request.rel_paths)
        t1 = time.perf_counter()
        bt = preprocess_bt(bt, self.bt_range)
        bt = _ensure_channel_axis(bt)
        t2 = time.perf_counter()

        if request.wind_kt is None:
            cond: Any = request.labels.copy()
        else:
            cond = {
                "ss_cat": request.labels.copy(),
                "wind_kt": request.wind_kt.copy(),
            }

        info = {
            "batch_size": float(request.labels.shape[0]),
            "load_time_sec": float(t1 - t0),
            "preprocess_time_sec": float(t2 - t1),
            "batch_prep_time_sec": float(t2 - t0),
        }
        return bt, cond, info

    def _submit_one(self):
        request = self._make_batch_request()
        if self._executor is None:
            payload = self._materialize_batch(request)
        else:
            payload = self._executor.submit(self._materialize_batch, request)
        self._pending.append(PendingTrainBatch(request=request, payload=payload))

    def _fill_prefetch_queue(self, target_size: int):
        while len(self._pending) < int(target_size):
            self._submit_one()

    def _resolve_pending(
        self,
        pending: PendingTrainBatch,
    ) -> Tuple[np.ndarray, Any, Dict[str, float]]:
        payload = pending.payload
        if isinstance(payload, Future):
            return payload.result()
        return payload

    def _next_batch(self) -> Tuple[np.ndarray, Any]:
        self._fill_prefetch_queue(self.prefetch_batches + 1)
        pending = self._pending.popleft()
        bt, cond, info = self._resolve_pending(pending)
        self._last_batch_info = info
        self._fill_prefetch_queue(self.prefetch_batches)
        return bt, cond

    def rewind_unconsumed(self) -> int:
        if not self._pending:
            self._last_batch_info = None
            return 0

        # Prefetch intentionally samples future batches ahead of consumption.
        # Before an epoch boundary or checkpoint save, rewind to the first
        # not-yet-consumed batch so resumable training tracks consumed data.
        rewind_state = self._pending[0].request.state_before_batch
        pending_count = len(self._pending)
        while self._pending:
            pending = self._pending.popleft()
            payload = pending.payload
            if isinstance(payload, Future):
                payload.cancel()
                try:
                    payload.result()
                except CancelledError:
                    pass
        self.rng.setstate(rewind_state)
        self._last_batch_info = None
        return pending_count

    def close(self):
        self.rewind_unconsumed()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=True)
            self._executor = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Any]]:
        while True:
            yield self._next_batch()


class CassandraHPCBatchGenerator(BalancedTCBatchGenerator):
    """Cassandra/GPFS-optimized packed-memmap batch generator."""

    def __init__(
        self,
        backend: PackedMemmapBackend,
        class_to_files: Dict[int, List[str]],
        class_probs: Dict[int, float],
        seed: int = 42,
        bt_range: Tuple[float, float] = (117.0, 348.0),
        use_wind_speed: bool = False,
        rel_path_to_wind_kt: Dict[str, float] | None = None,
        batch_size: int = 8,
        prefetch_batches: int = 2,
        sort_within_shard: bool = True,
        max_samples_per_read: int | None = None,
        warmup_shards: bool = False,
    ):
        if not isinstance(backend, PackedMemmapBackend):
            raise TypeError(
                "CassandraHPCBatchGenerator requires PackedMemmapBackend to preserve "
                "the packed-memmap transport path."
            )
        super().__init__(
            backend=backend,
            class_to_files=class_to_files,
            class_probs=class_probs,
            seed=seed,
            bt_range=bt_range,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
        )
        self.pipeline_name = "cassandra_hpc_batch_generator"
        self.profile_name = "cassandra_hpc"
        self.sort_within_shard = bool(sort_within_shard)
        self.max_samples_per_read = (
            None if max_samples_per_read is None else int(max_samples_per_read)
        )
        if self.max_samples_per_read is not None and self.max_samples_per_read <= 0:
            raise ValueError(
                f"max_samples_per_read must be > 0 when set, got {self.max_samples_per_read}"
            )
        self.warmup_shards = bool(warmup_shards)
        self.class_to_global_indices: Dict[int, Tuple[int, ...]] = {}
        self.global_index_to_wind_kt: Dict[int, float] = {}

        for cls, rel_paths in self.class_to_files.items():
            global_indices: List[int] = []
            for rel_path in rel_paths:
                try:
                    global_index = int(self.backend.path_to_index[rel_path])
                except KeyError as exc:
                    raise KeyError(
                        f"Packed manifest does not contain training rel_path '{rel_path}'"
                    ) from exc
                global_indices.append(global_index)
                if self.use_wind_speed:
                    self.global_index_to_wind_kt[global_index] = float(
                        self.rel_path_to_wind_kt.get(rel_path, ss_class_midpoint_kt(cls))
                    )
            self.class_to_global_indices[int(cls)] = tuple(global_indices)

        if self.warmup_shards:
            self.backend.warmup_shards(rows_per_shard=1)

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "sort_within_shard": self.sort_within_shard,
                "max_samples_per_read": self.max_samples_per_read,
                "warmup_shards": self.warmup_shards,
            }
        )
        return info

    def _make_batch_request(self) -> SampledTrainBatchRequest:
        state_before = self.rng.getstate()
        global_indices = np.empty((self.batch_size,), dtype=np.int64)
        labels = np.empty((self.batch_size,), dtype=np.int32)
        wind_arr = np.empty((self.batch_size,), dtype=np.float32) if self.use_wind_speed else None

        for batch_idx in range(self.batch_size):
            cls = self.rng.choices(self.classes, weights=self.probs, k=1)[0]
            global_index = int(self.rng.choice(self.class_to_global_indices[int(cls)]))
            global_indices[batch_idx] = np.int64(global_index)
            labels[batch_idx] = np.int32(cls)
            if wind_arr is not None:
                wind_arr[batch_idx] = np.float32(
                    self.global_index_to_wind_kt[global_index]
                )

        return SampledTrainBatchRequest(
            state_before_batch=state_before,
            labels=labels,
            global_indices=global_indices,
            wind_kt=wind_arr,
        )

    def _materialize_batch(
        self,
        request: SampledTrainBatchRequest,
    ) -> Tuple[np.ndarray, Any, Dict[str, float]]:
        if request.global_indices is None:
            raise ValueError("CassandraHPCBatchGenerator expects global_indices in the batch request.")
        t0 = time.perf_counter()
        bt = self.backend.load_bt_batch_by_index(
            request.global_indices,
            sort_within_shard=self.sort_within_shard,
            max_samples_per_read=self.max_samples_per_read,
        )
        t1 = time.perf_counter()
        bt = preprocess_bt(bt, self.bt_range)
        bt = _ensure_channel_axis(bt)
        t2 = time.perf_counter()

        if request.wind_kt is None:
            cond: Any = request.labels.copy()
        else:
            cond = {
                "ss_cat": request.labels.copy(),
                "wind_kt": request.wind_kt.copy(),
            }

        info = {
            "batch_size": float(request.labels.shape[0]),
            "load_time_sec": float(t1 - t0),
            "preprocess_time_sec": float(t2 - t1),
            "batch_prep_time_sec": float(t2 - t0),
        }
        return bt, cond, info


def _resolve_profile_alias(raw: str) -> str:
    aliases = {
        "legacy": "legacy",
        "legacy_sample_generator": "legacy",
        "legacy_generator": "legacy",
        "sample_generator": "legacy",
        "local": "local_5090",
        "local_5090": "local_5090",
        "local_batch": "local_5090",
        "balanced_batch": "local_5090",
        "balanced_batch_generator": "local_5090",
        "batch": "local_5090",
        "batch_generator": "local_5090",
        "cassandra": "cassandra_hpc",
        "cassandra_hpc": "cassandra_hpc",
        "cassandra_batch": "cassandra_hpc",
        "cassandra_hpc_batch_generator": "cassandra_hpc",
        "hpc": "cassandra_hpc",
        "hpc_batch": "cassandra_hpc",
    }
    key = str(raw).strip().lower()
    if key not in aliases:
        raise ValueError(
            "Unsupported training pipeline selector "
            f"{raw!r}. Expected one of: 'legacy', 'local_5090', 'cassandra_hpc'."
        )
    return aliases[key]


def resolve_train_pipeline_profile(data_cfg) -> str:
    # Backward-compatible rule: if the deprecated train_loader key is set, let
    # it override the higher-level profile so existing CLI overrides still work.
    raw_loader = data_cfg.get("train_loader")
    if raw_loader is not None:
        return _resolve_profile_alias(str(raw_loader))

    raw_profile = data_cfg.get("train_pipeline_profile", "local_5090")
    return _resolve_profile_alias(str(raw_profile))


def resolve_hpc_loader_cfg(data_cfg) -> Dict[str, Any]:
    cfg = data_cfg.get("hpc_loader", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("data.hpc_loader must be a mapping when set.")

    max_samples_per_read = cfg.get("max_samples_per_read")
    if max_samples_per_read is not None:
        max_samples_per_read = int(max_samples_per_read)
        if max_samples_per_read <= 0:
            raise ValueError(
                f"data.hpc_loader.max_samples_per_read must be > 0 when set, got {max_samples_per_read}"
            )

    return {
        "prefetch_batches": int(cfg.get("prefetch_batches", 2)),
        "sort_within_shard": bool(cfg.get("sort_within_shard", True)),
        "max_samples_per_read": max_samples_per_read,
        "warmup_shards": bool(cfg.get("warmup_shards", False)),
    }


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def create_dataset(
    cfg,
    split,
    return_train_generator: bool = False,
) -> tf.data.Dataset | Tuple[Any, BalancedTCGenerator]:
    """
    Create a dataset-like iterable.

    - For split=='train': infinite class-balanced sampling from either the
      legacy per-sample generator, the local 5090 batch generator, or the
      Cassandra HPC batch generator, selected by data.train_pipeline_profile
      (or the deprecated data.train_loader alias).
    - For split in {'val','test'}: by default, a finite, deterministic dataset that
      makes exactly one pass over the split files (robust validation).

    Evaluation modes (val/test only):
      - eval_mode='full' (default): iterate every file in the split once.
      - eval_mode='balanced_fixed': deterministic fixed-size stratified subset.
    """

    data_cfg = cfg["data"]
    cond_cfg = cfg.get("conditioning", {})
    use_wind_speed = bool(cond_cfg.get("use_wind_speed", False))
    index_path = Path(data_cfg["dataset_index"])
    backend = build_data_backend(data_cfg)

    batch_size = int(data_cfg["batch_size"])
    train_seed_cfg = cfg.get("training", {}).get("seed", 42)
    seed = 42 if train_seed_cfg is None else int(train_seed_cfg)

    # ---- load index ----
    if use_wind_speed:
        class_to_files, sample_meta = load_dataset_index(index_path, return_sample_meta=True)
    else:
        class_to_files = load_dataset_index(index_path)
        sample_meta = {}
    # apply split
    split_dir = Path(data_cfg.get("split_dir", "data/splits"))
    allowed = load_split_file_set(split_dir, split)

    # Use alpha from config only for training; use empirical (alpha=1.0) for val/test.
    # If curriculum is enabled, initialize at start_alpha (train_loop will update every epoch).
    alpha = float(data_cfg.get("class_balance_alpha", 1.0))
    curr_cfg = data_cfg.get("class_balance_curriculum", {})
    if split == "train" and isinstance(curr_cfg, dict) and bool(curr_cfg.get("enabled", False)):
        alpha = float(curr_cfg.get("start_alpha", alpha))

    # Filter class->files to this split
    class_to_files = {
        c: [p for p in paths if p in allowed]
        for c, paths in class_to_files.items()
    }

    # Sanity: ensure non-empty
    total = sum(len(v) for v in class_to_files.values())
    if total == 0:
        raise RuntimeError(f"No files left after applying split='{split}'. Check split manifests and dataset_index paths.")

    # --------------------------------------------------------
    # Finite evaluation dataset (recommended)
    # --------------------------------------------------------

    eval_mode = str(data_cfg.get("eval_mode", "full"))  # 'full' | 'balanced_fixed'
    eval_seed = seed

    bt_range = (float(data_cfg["bt_min_k"]), float(data_cfg["bt_max_k"]))

    if split in ("val", "test"):
        # Build deterministic (path,label) pairs from the split manifest.
        rel_paths, labels = _build_split_pairs(class_to_files, allowed)
        if len(rel_paths) == 0:
            raise RuntimeError(
                f"No files matched for split='{split}'. Check that split manifests contain paths present in dataset_index."
            )

        if eval_mode == "balanced_fixed":
            per_class = int(data_cfg.get("val_balanced_per_class", 5))
            rel_paths, labels = _fixed_balanced_subset(rel_paths, labels, per_class=per_class, seed=eval_seed)
            if len(rel_paths) == 0:
                raise RuntimeError(
                    f"balanced_fixed produced an empty subset (per_class={per_class})."
                )

        rel_path_to_wind_kt = None
        if use_wind_speed:
            rel_path_to_wind_kt, n_fallback = build_relpath_to_wind_kt(rel_paths, labels, sample_meta)
            if n_fallback == len(rel_paths):
                raise RuntimeError(
                    "conditioning.use_wind_speed=true, but dataset_index has no usable per-sample wind metadata "
                    "for this split. Rebuild data/dataset_index.json with scripts/create_dataset_index.py."
                )
            if n_fallback > 0:
                print(
                    f"[data] Warning: missing wind metadata for {n_fallback}/{len(rel_paths)} "
                    "validation/test samples; using class-midpoint fallback."
                )

        num_parallel_calls = resolve_eval_num_parallel_calls(data_cfg, backend)
        # NOTE: shuffle=False ensures a stable, full-pass metric per epoch.
        return _create_finite_eval_dataset(
            backend=backend,
            rel_paths=rel_paths,
            labels=labels,
            batch_size=batch_size,
            seed=eval_seed,
            bt_range=bt_range,
            shuffle=False,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
            num_parallel_calls=num_parallel_calls,
        )

    train_rel_paths = []
    train_labels = []
    for c, paths in class_to_files.items():
        for rel_path in paths:
            train_rel_paths.append(rel_path)
            train_labels.append(int(c))

    rel_path_to_wind_kt = None
    if use_wind_speed:
        rel_path_to_wind_kt, n_fallback = build_relpath_to_wind_kt(
            train_rel_paths,
            train_labels,
            sample_meta,
        )
        if n_fallback == len(train_rel_paths):
            raise RuntimeError(
                "conditioning.use_wind_speed=true, but dataset_index has no usable per-sample wind metadata "
                "for training. Rebuild data/dataset_index.json with scripts/create_dataset_index.py."
            )
        if n_fallback > 0:
            print(
                f"[data] Warning: missing wind metadata for {n_fallback}/{len(train_rel_paths)} "
                "training samples; using class-midpoint fallback."
            )

    # ---- compute sampling probabilities ----
    class_probs = compute_class_sampling_probs(class_to_files, alpha)

    train_pipeline_profile = resolve_train_pipeline_profile(data_cfg)

    if train_pipeline_profile == "cassandra_hpc":
        if not isinstance(backend, PackedMemmapBackend):
            raise ValueError(
                "data.train_pipeline_profile='cassandra_hpc' requires "
                "data.backend='packed_memmap'."
            )
        hpc_cfg = resolve_hpc_loader_cfg(data_cfg)
        gen = CassandraHPCBatchGenerator(
            backend=backend,
            class_to_files=class_to_files,
            class_probs=class_probs,
            seed=seed,
            bt_range=bt_range,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
            batch_size=batch_size,
            prefetch_batches=hpc_cfg["prefetch_batches"],
            sort_within_shard=hpc_cfg["sort_within_shard"],
            max_samples_per_read=hpc_cfg["max_samples_per_read"],
            warmup_shards=hpc_cfg["warmup_shards"],
        )
        ds = gen
    elif train_pipeline_profile == "local_5090":
        prefetch_batches = int(data_cfg.get("train_prefetch_batches", 1))
        gen = BalancedTCBatchGenerator(
            backend=backend,
            class_to_files=class_to_files,
            class_probs=class_probs,
            seed=seed,
            bt_range=bt_range,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
        )
        ds = gen
    else:
        gen = BalancedTCGenerator(
            backend=backend,
            class_to_files=class_to_files,
            class_probs=class_probs,
            seed=seed,
            bt_range=bt_range,
            use_wind_speed=use_wind_speed,
            rel_path_to_wind_kt=rel_path_to_wind_kt,
        )

        if use_wind_speed:
            output_signature = (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                {
                    "ss_cat": tf.TensorSpec(shape=(), dtype=tf.int32),
                    "wind_kt": tf.TensorSpec(shape=(), dtype=tf.float32),
                },
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )

        ds = tf.data.Dataset.from_generator(
            lambda: iter(gen),
            output_signature=output_signature,
        )

        ds = ds.batch(batch_size, drop_remainder=True)
        opts = tf.data.Options()
        opts.deterministic = True
        ds = ds.with_options(opts)

    if return_train_generator:
        return ds, gen
    return ds
