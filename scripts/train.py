# scripts/train.py

# usage (from inside tc-diffusion): python -m scripts.train --config configs/base.yaml
import argparse
from pathlib import Path
import shutil

from tc_diffusion.config import load_config
from tc_diffusion.train_loop import train
from tc_diffusion.plotting import save_loss_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries, e.g. training.lr=5e-4 data.batch_size=8",
    )
    p.add_argument(
        "--windows_loss_out",
        type=str,
        default=None,
        help=(
            "Optional path on Windows (e.g. /mnt/c/Users/guido/Desktop/loss_curve.png) "
            "to also copy the training loss plot."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    history = train(cfg)

    # Save loss curve to Linux path (under experiment output_dir)
    out_dir = Path(cfg["experiment"]["output_dir"])
    loss_png = out_dir / "loss_curve.png"
    save_loss_curve(history["steps"], history["loss"], str(loss_png))
    print(f"Saved loss curve to {loss_png}")

    # Optionally copy to Windows path
    if args.windows_loss_out is not None:
        win_path = Path(args.windows_loss_out)
        win_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loss_png, win_path)
        print(f"Copied loss curve to Windows path {win_path}")
