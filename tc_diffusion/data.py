import json
import random
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np
import tensorflow as tf
import xarray as xr


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

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
    total = sum(counts.values())

    # empirical distribution
    p_emp = {c: counts[c] / total for c in counts}

    # reweighted distribution
    weights = {c: p_emp[c] ** alpha for c in p_emp}
    Z = sum(weights.values())

    probs = {c: weights[c] / Z for c in weights}
    return probs


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
    ):
        self.data_root = data_root
        self.class_to_files = class_to_files
        self.class_probs = class_probs

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

            # 3) load NetCDF (BT only)
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                bt = ds["bt"].values.astype(np.float32)

            # ensure shape (H, W, 1)
            if bt.ndim == 2:
                bt = bt[..., None]

            yield bt, np.int32(cls)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def create_dataset(cfg) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset with class-balanced sampling.
    """

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    data_root = Path(data_cfg["data_root"])
    index_path = Path(data_cfg["dataset_index"])
    alpha = float(data_cfg.get("class_balance_alpha", 1.0))

    batch_size = int(data_cfg["batch_size"])
    seed = int(cfg.get("seed", 42))

    # ---- load index ----
    class_to_files = load_dataset_index(index_path)

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

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
