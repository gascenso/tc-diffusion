# scripts/eval.py

# Usage:
# (evaluate finished run)     python -m scripts.eval --name <RUN_NAME>
# (light evaluation)          python -m scripts.eval --name <RUN_NAME> --light
# (override config)           python -m scripts.eval --name <RUN_NAME> --override evaluation.n_per_class_heavy=100
# (show progress bar)         python -m scripts.eval --name <RUN_NAME> --show_progress

import argparse
from copy import deepcopy
from pathlib import Path

import tensorflow as tf

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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--heavy", dest="heavy", action="store_true", help="Run heavy evaluation (default).")
    mode.add_argument("--light", dest="heavy", action="store_false", help="Run light evaluation.")
    p.set_defaults(heavy=True)
    p.add_argument("--show_progress", action="store_true", help="Show progress bars during evaluation")
    return p.parse_args()


def configure_runtime(cfg):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    use_mixed_precision = bool(cfg.get("training", {}).get("mixed_precision", False))
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[eval] Mixed precision enabled (mixed_float16)")


def _legacy_model_cfg(cfg):
    legacy_cfg = deepcopy(cfg)
    legacy_cfg.setdefault("model", {})
    legacy_cfg["model"]["decoder_skip_mode"] = "per_level"
    legacy_cfg["model"]["output_head_pre_norm"] = False
    return legacy_cfg


def _load_model_with_arch_fallback(cfg, weights_path: Path):
    model_cfg = cfg.get("model", {})
    has_explicit_arch_keys = (
        "decoder_skip_mode" in model_cfg or "output_head_pre_norm" in model_cfg
    )

    attempts = [("saved config", cfg)]
    if not has_explicit_arch_keys:
        attempts.append(("legacy checkpoint compatibility mode", _legacy_model_cfg(cfg)))

    errors = []
    last_exc = None
    for label, cfg_variant in attempts:
        tf.keras.backend.clear_session()
        model = build_unet(cfg_variant)
        try:
            model.load_weights(str(weights_path))
            if label != "saved config":
                print(f"[eval] Loaded weights using {label}.")
            return model
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            last_exc = exc

    msg = "Could not load checkpoint with any known architecture variant:\n"
    msg += "\n".join(f"  - {err}" for err in errors)
    raise ValueError(msg) from last_exc


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
    configure_runtime(cfg)

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
    model = _load_model_with_arch_fallback(cfg, weights_path)
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
