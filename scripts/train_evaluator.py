#!/usr/bin/env python3

import argparse
import datetime
import json
import shutil
import time
from pathlib import Path

import yaml

from tc_diffusion.config import load_config
from tc_diffusion.evaluator.train_loop import train_evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Train the supervised TC evaluator on real data.")
    p.add_argument("--config", type=str, default="configs/evaluator_local.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--name", type=str, required=True, help="Run folder name under runs/.")
    p.add_argument("--resume", action="store_true", help="Resume from runs/<name>/train_checkpoints/.")
    return p.parse_args()


def write_run_config(run_dir: Path, cfg: dict):
    path = run_dir / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def write_run_meta(run_dir: Path, meta: dict):
    path = run_dir / "meta.json"
    with path.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    t0 = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    run_dir = Path("runs") / args.name
    if run_dir.exists() and not args.resume:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["output_dir"] = str(run_dir)

    write_run_config(run_dir, cfg)
    write_run_meta(
        run_dir,
        {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "config_path": args.config,
            "overrides": args.override,
            "name": args.name,
            "resume": bool(args.resume),
            "entrypoint": "scripts.train_evaluator",
        },
    )

    train_evaluator(cfg, resume=bool(args.resume))

    elapsed = time.perf_counter() - t0
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    print(f"Total evaluator wall time: {hrs:02d}:{mins:02d}:{secs:02d} (hh:mm:ss)")
