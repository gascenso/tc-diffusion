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


def _load_example_py(path_tensor, bt_min, bt_max, image_size):
    """
    Python-side loader for a single .nc file.

    Args:
        path_tensor: tf.EagerTensor with dtype string (path to file)
        bt_min, bt_max: floats in Kelvin used for normalization
        image_size: int target size (we assert it's already that)

    Returns:
        img: np.float32 array (H, W, 1) scaled to [-1, 1]
        cond: np.float32 scalar (placeholder conditioning)
    """
    # Convert Tensor -> Python string
    # path_tensor is a scalar tf.Tensor of type tf.string
    if isinstance(path_tensor, bytes):  # very defensive, usually it's a Tensor
        path = path_tensor.decode("utf-8")
    else:
        path = path_tensor.numpy().decode("utf-8")

    ds = xr.open_dataset(path)

    # Assuming variable name is "bt"
    bt = ds["bt"].values.astype(np.float32)  # (H, W)
    ds.close()

    # Optionally resize / crop to image_size if needed
    assert bt.shape[0] == image_size and bt.shape[1] == image_size, (
        f"Unexpected bt shape {bt.shape}, expected ({image_size}, {image_size})"
    )

    # Normalize BT from [bt_min, bt_max] -> [-1, 1]
    bt_clipped = np.clip(bt, bt_min, bt_max)
    bt_norm01 = (bt_clipped - bt_min) / (bt_max - bt_min)
    bt_norm11 = bt_norm01 * 2.0 - 1.0
    img = bt_norm11[..., None]  # (H, W, 1)

    # TODO: real conditioning here (e.g. wind speed, basin)
    cond = np.array(0.0, dtype=np.float32)

    return img.astype(np.float32), cond


def create_dataset(cfg, split="train"):
    """
    Return a tf.data.Dataset yielding (image, cond_scalar) pairs.
    For now:
      - cond is a scalar float (placeholder).
      - no train/val split differentiation (you can add later).
    """
    gridsat_dir = cfg["data"]["gridsat_dir"]
    bt_min = float(cfg["data"]["bt_min_k"])
    bt_max = float(cfg["data"]["bt_max_k"])
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"]["batch_size"])

    files = _list_nc_files(gridsat_dir)
    ds = tf.data.Dataset.from_tensor_slices(files)

    def _map_fn(path):
        # path is a tf.Tensor of dtype string
        img, cond = tf.py_function(
            func=lambda p: _load_example_py(p, bt_min, bt_max, image_size),
            inp=[path],
            Tout=(tf.float32, tf.float32),
        )
        # Set static shapes (important for Keras)
        img.set_shape((image_size, image_size, 1))
        cond.set_shape(())
        return img, cond

    ds = ds.shuffle(buffer_size=min(len(files), 10000))
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
