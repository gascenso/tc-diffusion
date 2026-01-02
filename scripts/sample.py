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
    p.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    p.add_argument(
        "--out",
        type=str,
        default="samples/sample_grid.png"
    )
    p.add_argument(
        "--windows_out",
        type=str,
        default="/mnt/c/Users/guido/Desktop/sample.png",
        help="Optional path on Windows (e.g. /mnt/c/Users/guido/Desktop/sample.png) "
             "to also copy the PNG to.",
    )
    p.add_argument(
        "--ss_cat",
        type=int,
        default=5,
        help=(
            "Saffir–Simpson category index to condition on:\n"
            "0=TS (35–63 kt), 1=Cat1, 2=Cat2, 3=Cat3, 4=Cat4, 5=Cat5."
        ),
    )
    p.add_argument(
        "--name",
        type=str,
        default="baseline_ddpm_tc",
        help="Run name under runs/ to load weights from.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. 0 disables CFG. Typical: 1-5."
    )
    return p.parse_args()


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

    weights_path = f"runs/{args.name}/weights_best.weights.h5"

    print(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)

    print("Sampling...")
    x_samples = diffusion.sample(
        model=model,
        batch_size=args.batch_size,
        image_size=image_size,
        cond_value=args.ss_cat,  # unconditional for now
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