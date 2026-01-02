# scripts/train.py

# Usage:
# (start new run)     python -m scripts.train --name <RUN_NAME>
# (resume run)        python -m scripts.train --name <RUN_NAME> --resume

import argparse
from pathlib import Path
import shutil
import yaml
import json
import datetime

from tc_diffusion.config import load_config
from tc_diffusion.train_loop import train
from tc_diffusion.plotting import save_loss_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml"
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries, e.g. training.lr=5e-4 data.batch_size=8",
    )
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name (folder under runs/). Example: --name improved_embeddings",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest weights_last.* in runs/<name>/ if available.",
    )

    p.add_argument(
        "--windows_loss_out",
        type=str,
        default="/mnt/c/Users/guido/Desktop/loss_curve.png",
        help=(
            "Optional path on Windows (e.g. /mnt/c/Users/guido/Desktop/loss_curve.png) "
            "to also copy the training loss plot."
        ),
    )
    return p.parse_args()

def write_run_config(run_dir: Path, cfg: dict):
    # Save resolved config as YAML (machine-readable manifest)
    path = run_dir / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def write_run_meta(run_dir: Path, meta: dict):
    # Optional extra metadata (timestamps, CLI, etc.)
    path = run_dir / "meta.json"
    with path.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    # Resolve run directory (under repo/runs/)
    runs_root = Path("runs")
    run_dir = runs_root / args.name

    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        # Starting fresh in an existing directory is allowed.
        # Clean up files that would cause accidental resume or confusion.
        for p in run_dir.glob("weights_last.epoch_*.weights.h5"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in [
            run_dir / "weights_best.weights.h5",
            run_dir / "run_state.json",
        ]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Inject output_dir into config so everything uses this run folder
    cfg.setdefault("experiment", {})
    cfg["experiment"]["output_dir"] = str(run_dir)

    # Save config + metadata at launch time
    write_run_config(run_dir, cfg)
    write_run_meta(
        run_dir,
        {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "config_path": args.config,
            "overrides": args.override,
            "name": args.name,
            "resume": bool(args.resume),
        },
    )

    history = train(cfg, resume=args.resume)

    loss_png = run_dir / "loss_curve.png"
    save_loss_curve(history["epoch"], history["epoch_loss"], str(loss_png))
    print(f"Saved loss curve to {loss_png}")

    if args.windows_loss_out is not None:
        win_path = Path(args.windows_loss_out)
        win_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loss_png, win_path)
        print(f"Copied loss curve to Windows path {win_path}")
