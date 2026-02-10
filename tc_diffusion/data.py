import json
import random
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np
import tensorflow as tf
import xarray as xr

def _build_aug_policy(num_classes: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return classwise augmentation parameters sized to num_classes."""
    if int(num_classes) <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    # Anchor policy for the canonical 6 SS classes; resampled for other counts.
    p_anchor = np.array([0.15, 0.20, 0.35, 0.60, 0.75, 0.85], dtype=np.float32)
    shift_anchor = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
    bars_anchor = np.array([1, 1, 1, 2, 2, 3], dtype=np.int32)

    if int(num_classes) == int(p_anchor.shape[0]):
        p_aug = p_anchor
        max_shift = shift_anchor
        max_bars = bars_anchor
    else:
        src = np.linspace(0.0, 1.0, num=p_anchor.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=int(num_classes), dtype=np.float32)
        p_aug = np.interp(dst, src, p_anchor).astype(np.float32)
        max_shift = np.rint(
            np.interp(dst, src, shift_anchor.astype(np.float32))
        ).astype(np.int32)
        max_bars = np.rint(
            np.interp(dst, src, bars_anchor.astype(np.float32))
        ).astype(np.int32)

    max_shift = np.maximum(max_shift, 0)
    max_bars = np.maximum(max_bars, 0)
    return (
        tf.constant(p_aug, dtype=tf.float32),
        tf.constant(max_shift, dtype=tf.int32),
        tf.constant(max_bars, dtype=tf.int32),
    )

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def rot90_only(x):
    # x: [H, W, C] or [H, W]
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, k)

def random_shift_reflect(x, max_shift):
    # max_shift: int scalar tensor
    if max_shift <= 0:
        return x

    h = tf.shape(x)[0]
    w = tf.shape(x)[1]

    dy = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    dx = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)

    # reflect pad enough to allow cropping after shift
    pad_y = max_shift
    pad_x = max_shift
    x_pad = tf.pad(
        x,
        paddings=[[pad_y, pad_y], [pad_x, pad_x], [0, 0]] if x.shape.rank == 3 else [[pad_y, pad_y], [pad_x, pad_x]],
        mode="REFLECT",
    )

    # crop with offset
    y0 = pad_y + dy
    x0 = pad_x + dx
    if x.shape.rank == 3:
        return tf.image.crop_to_bounding_box(x_pad, y0, x0, h, w)
    else:
        # expand to use crop_to_bounding_box then squeeze
        x_pad3 = x_pad[..., None]
        out = tf.image.crop_to_bounding_box(x_pad3, y0, x0, h, w)
        return tf.squeeze(out, axis=-1)

def random_bar_erasing(x, max_bars, min_w=1, max_w=5, fill_mode="local"):
    """
    Erase bars spanning full height or width.
    TF-graph-safe + shape-invariant in while_loop.
    """

    # Ensure [H, W, C]
    if x.shape.rank == 2:
        x3 = x[..., None]
        squeeze_back = True
    else:
        x3 = x
        squeeze_back = False

    # Dynamic spatial dims
    H = tf.shape(x3)[0]
    W = tf.shape(x3)[1]

    # IMPORTANT: keep a *static* channel dim if available (e.g., 1)
    C_static = x3.shape[-1]  # python int or None
    if C_static is None:
        C = tf.shape(x3)[2]   # fallback dynamic
    else:
        C = C_static          # use static for shapes below

    def no_erase():
        return x3

    def do_erase():
        nbars = tf.random.uniform([], 0, max_bars + 1, dtype=tf.int32)

        def body(i, img):
            horiz = tf.random.uniform([], 0, 2, dtype=tf.int32)  # 0=vertical, 1=horizontal
            bw = tf.random.uniform([], min_w, max_w + 1, dtype=tf.int32)

            if fill_mode == "local":
                mu = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
                sigma = tf.math.reduce_std(img, axis=[0, 1], keepdims=True)
                fill = mu + tf.random.normal([H, W, C], dtype=img.dtype) * (0.1 * (sigma + 1e-6))
            else:
                fill = tf.random.normal([H, W, C], dtype=img.dtype)

            def vertical():
                x0 = tf.random.uniform([], 0, W - bw + 1, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([H, x0, C], dtype=img.dtype),
                    tf.zeros([H, bw, C], dtype=img.dtype),
                    tf.ones([H, W - x0 - bw, C], dtype=img.dtype),
                ], axis=1)
                return img * mask + fill * (1.0 - mask)

            def horizontal():
                y0 = tf.random.uniform([], 0, H - bw + 1, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([y0, W, C], dtype=img.dtype),
                    tf.zeros([bw, W, C], dtype=img.dtype),
                    tf.ones([H - y0 - bw, W, C], dtype=img.dtype),
                ], axis=0)
                return img * mask + fill * (1.0 - mask)

            img2 = tf.cond(horiz == 0, vertical, horizontal)

            # Force static shape preservation (critical for while_loop)
            img2.set_shape(x3.shape)
            return i + 1, img2

        # Shape invariants: allow H,W to be unknown, but keep channel rank consistent
        img_inv = tf.TensorShape([None, None, C_static if C_static is not None else None])

        _, out = tf.while_loop(
            lambda i, _: i < nbars,
            body,
            loop_vars=[tf.constant(0, tf.int32), x3],
            shape_invariants=[tf.TensorShape([]), img_inv],
        )

        out.set_shape(x3.shape)
        return out

    out = tf.cond(max_bars > 0, do_erase, no_erase)
    out.set_shape(x3.shape)

    if squeeze_back:
        return tf.squeeze(out, axis=-1)
    return out

def augment_x_given_y(
    x,
    y,
    p_aug: tf.Tensor,
    max_shift_per_class: tf.Tensor,
    max_bars_per_class: tf.Tensor,
):
    """
    x: [H,W,1] (or [H,W]) normalized already (e.g., to [-1,1])
    y: int label
    """
    y = tf.cast(y, tf.int32)

    p = tf.gather(p_aug, y)
    max_shift = tf.gather(max_shift_per_class, y)
    max_bars = tf.gather(max_bars_per_class, y)

    def do_aug():
        x1 = rot90_only(x)
        x2 = random_shift_reflect(x1, max_shift=max_shift)
        x3 = random_bar_erasing(x2, max_bars=max_bars, min_w=1, max_w=5, fill_mode="local")
        return x3

    return tf.cond(tf.random.uniform([]) < p, do_aug, lambda: x)

def preprocess_bt(bt: np.ndarray, bt_range: Tuple[float, float]) -> np.ndarray:
    bt = bt.astype(np.float32)
    bt = np.nan_to_num(bt, nan=0.0, posinf=0.0, neginf=0.0)

    bt_min, bt_max = bt_range
    bt = np.clip(bt, bt_min, bt_max)

    bt = (bt - bt_min) / (bt_max - bt_min)  # [0,1]
    bt = bt * 2.0 - 1.0                     # [-1,1]
    return bt

def load_dataset_index(index_path: Path) -> Dict[str, List[str]]:
    """
    Load dataset_index.json and return class -> list of relative file paths.
    """
    with open(index_path, "r") as f:
        index = json.load(f)

    class_to_files = {
        int(k): v for k, v in index["classes"].items()
    }

    return class_to_files


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


def _tf_load_one_netcdf(
    data_root: Path,
    bt_range: Tuple[float, float],
    engine: str = "netcdf4",
):
    """Factory returning a tf.data mapping fn that loads & preprocesses NetCDF.

    IMPORTANT: netCDF4/HDF5 is not reliably thread-safe under concurrent opens.
    Use num_parallel_calls=1 in tf.data when using this.
    """
    bt_min, bt_max = float(bt_range[0]), float(bt_range[1])

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
        nc_path = data_root / rel_path

        with xr.open_dataset(nc_path, engine=engine) as ds:
            bt = ds["bt"].values.astype(np.float32)

        bt = preprocess_bt(bt, (bt_min, bt_max))
        if bt.ndim == 2:
            bt = bt[..., None]
        return bt, np.int32(label)

    def _tf_map(rel_path, label):
        bt, y = tf.numpy_function(
            func=_py_load,
            inp=[rel_path, label],
            Tout=[tf.float32, tf.int32],
        )
        bt.set_shape([None, None, 1])
        y.set_shape([])
        return bt, y

    return _tf_map


def _create_finite_eval_dataset(
    *,
    data_root: Path,
    rel_paths: List[str],
    labels: List[int],
    batch_size: int,
    seed: int,
    bt_range: Tuple[float, float],
    shuffle: bool,
    num_parallel_calls: int = 1,
    engine: str = "netcdf4",
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

    # Thread-safety: keep netCDF opens single-threaded.
    map_fn = _tf_load_one_netcdf(data_root, bt_range, engine=engine)
    ds = ds.map(map_fn, num_parallel_calls=int(num_parallel_calls), deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)

    # Stronger guarantee: single-threaded tf.data pipeline for eval
    opts = tf.data.Options()
    opts.deterministic = True
    opts.threading.private_threadpool_size = 1
    opts.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(opts)

    return ds
    
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
        data_root: Path,
        class_to_files: Dict[int, List[str]],
        class_probs: Dict[int, float],
        seed: int = 42,
        bt_range: Tuple[float, float] = (117.0, 348.0),
    ):
        self.data_root = data_root
        self.class_to_files = class_to_files
        self.class_probs = class_probs
        self.bt_range = bt_range
        self.classes = sorted(class_to_files.keys())
        self.probs = np.array([class_probs[c] for c in self.classes])

        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.int32]]:
        while True:
            # 1) sample class
            cls = self.rng.choices(self.classes, weights=self.probs, k=1)[0]

            # 2) sample file within class
            rel_path = self.rng.choice(self.class_to_files[cls])
            nc_path = self.data_root / rel_path

            # 3) load NetCDF (BT only) and normalize
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                bt = ds["bt"].values.astype(np.float32)
            bt = preprocess_bt(bt, self.bt_range)
            
            # ensure shape (H, W, 1)
            if bt.ndim == 2:
                bt = bt[..., None]

            yield bt, np.int32(cls)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def create_dataset(cfg, split) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset.

    - For split=='train': infinite class-balanced sampling (BalancedTCGenerator).
    - For split in {'val','test'}: by default, a finite, deterministic dataset that
      makes exactly one pass over the split files (robust validation).

    Evaluation modes (val/test only):
      - eval_mode='full' (default): iterate every file in the split once.
      - eval_mode='balanced_fixed': deterministic fixed-size stratified subset.
    """

    data_cfg = cfg["data"]
    data_root = Path(data_cfg["data_root"])
    index_path = Path(data_cfg["dataset_index"])
    alpha = float(data_cfg.get("class_balance_alpha", 1.0))

    batch_size = int(data_cfg["batch_size"])
    seed = int(cfg.get("seed", 42))

    # ---- load index ----
    class_to_files = load_dataset_index(index_path)
    # apply split
    split_dir = Path(data_cfg.get("split_dir", "data/splits"))
    allowed = load_split_file_set(split_dir, split)

    # Use alpha from config only for training; use empirical (alpha=1.0) for val/test
    alpha = float(data_cfg.get("class_balance_alpha", 1.0))

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
    eval_seed = int(data_cfg.get("eval_seed", seed))

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

        num_parallel_calls = int(data_cfg.get("eval_num_parallel_calls", 1))
        engine = str(data_cfg.get("eval_engine", "netcdf4"))
        # NOTE: shuffle=False ensures a stable, full-pass metric per epoch.
        return _create_finite_eval_dataset(
            data_root=data_root,
            rel_paths=rel_paths,
            labels=labels,
            batch_size=batch_size,
            seed=eval_seed,
            bt_range=bt_range,
            shuffle=False,
            num_parallel_calls=num_parallel_calls,
            engine=engine,
        )

    # ---- compute sampling probabilities ----
    class_probs = compute_class_sampling_probs(class_to_files, alpha)

    # print once for sanity
    print("\n[data] Class-balanced sampling probabilities:")
    for c in sorted(class_probs):
        print(f"  class {c}: p = {class_probs[c]:.4f} "
              f"(n = {len(class_to_files[c])})")
    print()

    # ---- generator ----
    gen = BalancedTCGenerator(
        data_root=data_root,
        class_to_files=class_to_files,
        class_probs=class_probs,
        seed=seed,
        bt_range=bt_range,
    )

    # ---- tf.data.Dataset ----
    ds = tf.data.Dataset.from_generator(
        lambda: iter(gen),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # shuffle only within small buffer (structure comes from sampler)
    ds = ds.shuffle(buffer_size=4 * batch_size, seed=seed)
    p_aug, max_shift, max_bars = _build_aug_policy(int(cfg["conditioning"]["num_ss_classes"]))
    ds = ds.map(
        lambda x, y: (augment_x_given_y(x, y, p_aug, max_shift, max_bars), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
