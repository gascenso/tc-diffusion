# scripts/sample.py
import argparse
from pathlib import Path
import glob
import shutil

import tensorflow as tf

from tc_diffusion.config import load_config
from tc_diffusion.model_unet import build_unet
from tc_diffusion.diffusion import Diffusion
from tc_diffusion.plotting import save_image_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights .weights.h5 file. "
             "If not provided, will try to find the latest in experiment.output_dir.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", type=str, default="samples/sample_grid.png")
    p.add_argument(
        "--windows_out",
        type=str,
        default=None,
        help="Optional path on Windows (e.g. /mnt/c/Users/guido/Desktop/sample.png) "
             "to also copy the PNG to.",
    )
    return p.parse_args()


def find_latest_weights(output_dir: str):
    pattern = str(Path(output_dir) / "weights_*.weights.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weight files found in {output_dir}")
    return files[-1]


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=[])

    # GPU memory growth (optional, same as train)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    image_size = int(cfg["data"]["image_size"])

    model = build_unet(cfg)
    diffusion = Diffusion(cfg)

    if args.weights is None:
        out_dir = cfg["experiment"]["output_dir"]
        weights_path = find_latest_weights(out_dir)
    else:
        weights_path = args.weights

    print(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)

    print("Sampling...")
    x_samples = diffusion.sample(
        model=model,
        batch_size=args.batch_size,
        image_size=image_size,
        cond_scalar=0.0,  # unconditional for now
    )

    out_path = Path(args.out)
    save_image_grid(
        x_samples,
        path=str(out_path),
        bt_min_k=float(cfg["data"]["bt_min_k"]),
        bt_max_k=float(cfg["data"]["bt_max_k"]),
    )
    print(f"Saved sample grid to {out_path}")

    if args.windows_out is not None:
        win_path = Path(args.windows_out)
        win_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, win_path)
        print(f"Copied sample grid to Windows path {win_path}")