# scripts/train.py
import argparse
from tc_diffusion.config import load_config
from tc_diffusion.train_loop import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries, e.g. training.lr=5e-4 data.batch_size=8",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    train(cfg)
