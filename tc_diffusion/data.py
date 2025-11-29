# tc_diffusion/data.py
import os
from glob import glob

import numpy as np
import xarray as xr
import tensorflow as tf


def _list_nc_files(gridsat_dir):
    paths = sorted(glob(os.path.join(gridsat_dir, "*.nc")))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {gridsat_dir}")
    return paths


def _load_example_np(path, bt_min, bt_max, image_size):
    """
    Pure NumPy/xarray loader for a single .nc file.

    Returns:
        img: np.float32 array (H, W, 1) scaled to [-1, 1]
        cond: np.float32 scalar (placeholder conditioning)
    """
    ds = xr.open_dataset(path)

    # Assuming variable name is "bt"
    bt = ds["bt"].values.astype(np.float32)  # (H, W)
    ds.close()

    # Ensure correct size
    if bt.shape[0] != image_size or bt.shape[1] != image_size:
        raise ValueError(
            f"Unexpected bt shape {bt.shape}, expected ({image_size}, {image_size})"
        )

    # Normalize BT from [bt_min, bt_max] -> [-1, 1]
    bt_clipped = np.clip(bt, bt_min, bt_max)
    bt_norm01 = (bt_clipped - bt_min) / (bt_max - bt_min)
    bt_norm11 = bt_norm01 * 2.0 - 1.0
    img = bt_norm11[..., None]  # (H, W, 1)

    # TODO: replace with real conditioning (wind speed, basin, etc.)
    cond = np.array(0.0, dtype=np.float32)

    return img.astype(np.float32), cond.astype(np.float32)


def create_dataset(cfg, split="train"):
    """
    Return a tf.data.Dataset yielding (image, cond_scalar) pairs.

    For now:
      - cond is a scalar float (placeholder).
      - no train/val split differentiation yet.
      - I/O is strictly sequential to avoid HDF5 thread issues.
    """
    gridsat_dir = cfg["data"]["gridsat_dir"]
    bt_min = float(cfg["data"]["bt_min_k"])
    bt_max = float(cfg["data"]["bt_max_k"])
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"]["batch_size"])

    files = _list_nc_files(gridsat_dir)

    def generator():
        # This generator will be (re)called by tf.data when the dataset is iterated
        for path in files:
            img, cond = _load_example_np(path, bt_min, bt_max, image_size)
            yield img, cond

    output_signature = (
        tf.TensorSpec(
            shape=(image_size, image_size, 1), dtype=tf.float32
        ),  # image
        tf.TensorSpec(shape=(), dtype=tf.float32),  # cond scalar
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Basic shuffling/batching/prefetching.
    # These operate on already-loaded tensors, so HDF5 I/O stays single-threaded.
    ds = ds.shuffle(buffer_size=min(len(files), 10000))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds