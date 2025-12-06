# tc_diffusion/data.py
import os
from glob import glob
import random

import numpy as np
import xarray as xr
import tensorflow as tf


def _list_nc_files(gridsat_dir):
    paths = sorted(glob(os.path.join(gridsat_dir, "*.nc")))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {gridsat_dir}")
    return paths


def _wind_to_ss_category(w_knots: float) -> int:
    """
    Map WMO_WIND (knots) to Saffir–Simpson category index:

      0: Tropical Storm (35–63 kt)
      1: Category 1 (64–82 kt)
      2: Category 2 (83–95 kt)
      3: Category 3 (96–112 kt)
      4: Category 4 (113–136 kt)
      5: Category 5 (>136 kt)

    If w < 35, we still map to 0 (TS / sub-TS bundled together).
    """
    w = float(w_knots)
    if w <= 63.0:
        return 0
    elif w <= 82.0:
        return 1
    elif w <= 95.0:
        return 2
    elif w <= 112.0:
        return 3
    elif w <= 136.0:
        return 4
    else:
        return 5
    

def _load_example_np(path, bt_min, bt_max, image_size):
    """
    Pure NumPy/xarray loader for a single .nc file.

    Returns:
        img:  np.float32 (H, W, 1) scaled to [-1, 1]
        cond: np.int32 scalar = SS category index (0..5)
    """
    ds = xr.open_dataset(path)

    # Brightness temperature
    bt = ds["bt"].values.astype(np.float32)
    if bt.shape[0] != image_size or bt.shape[1] != image_size:
        raise ValueError(
            f"Unexpected bt shape {bt.shape}, expected ({image_size}, {image_size})"
        )

    # Wind → SS category
    w_knots = ds.wmo_wind
    ss_cat = _wind_to_ss_category(w_knots)

    ds.close()

    # Normalize BT from [bt_min, bt_max] -> [-1, 1]
    bt_clipped = np.clip(bt, bt_min, bt_max)
    bt_norm01 = (bt_clipped - bt_min) / (bt_max - bt_min)
    bt_norm11 = bt_norm01 * 2.0 - 1.0
    img = bt_norm11[..., None]  # (H, W, 1)

    cond = np.array(ss_cat, dtype=np.int32)  # SS category index

    return img.astype(np.float32), cond


def create_dataset(cfg, split="train"):
    """
    Return a tf.data.Dataset yielding (image, ss_category) pairs.

    ss_category is an int32 scalar in [0, 5].
    """
    gridsat_dir = cfg["data"]["gridsat_dir"]
    bt_min = float(cfg["data"]["bt_min_k"])
    bt_max = float(cfg["data"]["bt_max_k"])
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"]["batch_size"])

    files = _list_nc_files(gridsat_dir)
    random.shuffle(files)                          # limit for testing, REMOVE LATER
    files = files[: int(cfg["data"]["max_files"])] # limit for testing, REMOVE LATER

    def generator():
        for path in files:
            img, cond = _load_example_np(path, bt_min, bt_max, image_size)
            yield img, cond

    output_signature = (
        tf.TensorSpec(
            shape=(image_size, image_size, 1),
            dtype=tf.float32,
        ),  # image
        tf.TensorSpec(shape=(), dtype=tf.int32),  # SS category
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=min(len(files), 5000))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
