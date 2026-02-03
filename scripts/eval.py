# scripts/eval.py
import argparse
from pathlib import Path
import yaml

import tensorflow as tf

from tc_diffusion.config import load_config
from tc_diffusion.model_unet import build_unet
from tc_diffusion.diffusion import Diffusion
from tc_diffusion.evaluation.evaluator import TCEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--name", type=str, required=True, help="Name of run under runs/ to load weights from")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (defaults to run dir inferred from weights path)")
    p.add_argument("--tag", type=str, default="manual_eval")
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--progress", action="store_true", help="Show progress bars during evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    weights_path = Path("runs") / args.name / "weights_ema_best_val.weights.h5"
    if args.out_dir is None:
        # if weights are in runs/<name>/..., use that as out_dir
        out_dir = weights_path.parent
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build + load
    model = build_unet(cfg)
    model.load_weights(str(weights_path))
    diffusion = Diffusion(cfg)

    evaluator = TCEvaluator(cfg)
    rep = evaluator.run(
        model=model,
        diffusion=diffusion,
        out_dir=out_dir,
        tag=args.tag,
        heavy=bool(args.heavy),
        show_progress=bool(args.progress),
    )

    print("Wrote evaluation report to:", out_dir / "eval" / args.tag / "metrics.json")
