# scripts/eval.py

# Usage:
# (evaluate finished run)     python -m scripts.eval --name <RUN_NAME>
# (override config)           python -m scripts.eval --name <RUN_NAME> --override evaluation.n_per_class_heavy=100
# (show progress bar)         python -m scripts.eval --name <RUN_NAME> --show_progress

import argparse
from pathlib import Path

from tc_diffusion.config import load_config
from tc_diffusion.model_unet import build_unet
from tc_diffusion.diffusion import Diffusion
from tc_diffusion.evaluation.evaluator import TCEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to runs/<name>/config.yaml saved at training time.",
    )
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--name", type=str, required=True, help="Name of run under runs/ to load weights from")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (defaults to run dir inferred from weights path)")
    p.add_argument("--tag", type=str, default="manual_eval")
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--show_progress", action="store_true", help="Show progress bars during evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dir = Path("runs") / args.name
    config_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Pass --config explicitly or ensure the run has a saved config.yaml."
        )
    cfg = load_config(str(config_path), overrides=args.override)

    if args.out_dir is None:
        out_dir = run_dir
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_weights_path() -> Path:
        ema_candidates = [
            run_dir / "weights_ema_best_val.weights.h5",
            run_dir / "weights_ema_best.weights.h5",
        ]
        for p in ema_candidates:
            if p.exists():
                return p

        non_ema_candidates = [
            run_dir / "weights_best_val.weights.h5",
            run_dir / "weights_best.weights.h5",
        ]
        for p in non_ema_candidates:
            if p.exists():
                print(f"[warn] EMA checkpoint not found; using non-EMA weights: {p.name}")
                return p

        tried = [str(p) for p in (ema_candidates + non_ema_candidates)]
        raise FileNotFoundError(f"Could not find evaluation weights. Tried: {tried}")

    weights_path = resolve_weights_path()

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
        show_progress=bool(args.show_progress),
    )

    print("Wrote evaluation report to:", out_dir / "eval" / args.tag / "metrics.json")
