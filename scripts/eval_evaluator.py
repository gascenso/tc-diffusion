#!/usr/bin/env python3

import argparse
from pathlib import Path

from tc_diffusion.config import load_config
from tc_diffusion.evaluator.train_loop import evaluate_saved_evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained supervised TC evaluator.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to runs/<name>/config.yaml.",
    )
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--name", type=str, required=True, help="Run folder name under runs/.")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights file. Defaults to best evaluator weights in runs/<name>/.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--tag", type=str, default="manual_eval")
    p.add_argument("--show_progress", action="store_true")
    return p.parse_args()


def resolve_weights(run_dir: Path, explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Evaluator weights not found: {path}")
        return path

    candidates = [
        run_dir / "weights_best_tail.weights.h5",
        run_dir / "weights_best.weights.h5",
        run_dir / "weights_last.weights.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find evaluator weights. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


if __name__ == "__main__":
    args = parse_args()
    run_dir = Path("runs") / args.name
    config_path = Path(args.config) if args.config else run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass --config or train the evaluator first."
        )

    cfg = load_config(str(config_path), overrides=args.override)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["output_dir"] = str(run_dir)

    weights = resolve_weights(run_dir, args.weights)
    report = evaluate_saved_evaluator(
        cfg,
        weights_path=weights,
        split_name=args.split,
        out_dir=run_dir,
        tag=args.tag,
        show_progress=bool(args.show_progress),
    )
    print(
        f"Wrote evaluator {args.split} metrics to "
        f"{run_dir / 'eval_evaluator' / args.tag / (args.split + '_metrics.json')}"
    )
    print(
        "Summary: "
        f"balanced_accuracy={report['classification']['balanced_accuracy']}, "
        f"macro_f1={report['classification']['macro_f1']}, "
        f"mae_kt={report['regression']['mae_kt']}, "
        f"tail_score={report['selection']['tail_score']}"
    )
