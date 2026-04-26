# scripts/eval.py

# Usage:
# (evaluate finished run)     python -m scripts.eval --name <RUN_NAME>
# (evaluate on test split)    python -m scripts.eval --name <RUN_NAME> --split test
# (full test-set reals)       python -m scripts.eval --name <RUN_NAME> --split test --full_test
# (light evaluation)          python -m scripts.eval --name <RUN_NAME> --light
# (override config)           python -m scripts.eval --name <RUN_NAME> --override evaluation.n_per_class_heavy=100
# (disable progress bars)     python -m scripts.eval --name <RUN_NAME> --verbose 0

import argparse
from copy import deepcopy
from pathlib import Path

import tensorflow as tf

from tc_diffusion.config import load_config
from tc_diffusion.model_unet import build_unet
from tc_diffusion.diffusion import Diffusion
from tc_diffusion.evaluation.evaluator import TCEvaluator, resolve_eval_root
from tc_diffusion.sample_bank import SampleBank, model_outputs_root, repo_relative_or_abs, repo_root, resolve_sample_bank_dir


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
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output dir. Defaults to runs/<name> for online evals and outputs/<name> when --sample_bank is used."
        ),
    )
    p.add_argument("--tag", type=str, default="manual_eval")
    p.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Dataset split to evaluate on.")
    p.add_argument(
        "--sample_bank",
        type=str,
        default=None,
        help=(
            "Optional cached sample-bank reference. Resolved under outputs/<name>/sample_banks/<split>/<sample_bank> "
            "unless an absolute path is given."
        ),
    )
    p.add_argument(
        "--generated_limit",
        type=int,
        default=None,
        help="Optional per-class cap when loading from --sample_bank. Defaults to all cached samples in the bank.",
    )
    p.add_argument(
        "--full_test",
        action="store_true",
        help=(
            "Test-only mode: use all available real test samples per class as the reference distribution, "
            "while keeping generated sample counts controlled by evaluation.n_per_class_*."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--heavy", dest="heavy", action="store_true", help="Run heavy evaluation (default).")
    mode.add_argument("--light", dest="heavy", action="store_false", help="Run light evaluation.")
    p.set_defaults(heavy=True)
    p.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1],
        default=1,
        help="Verbosity level. Use 1 to show progress bars (default) or 0 for quiet mode.",
    )
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
    if args.full_test and args.split != "test":
        raise ValueError("--full_test is only supported together with --split test.")
    if args.generated_limit is not None and args.generated_limit <= 0:
        raise ValueError("--generated_limit must be > 0 when provided.")

    repo = repo_root()
    run_dir = repo / "runs" / args.name
    config_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Pass --config explicitly or ensure the run has a saved config.yaml."
        )
    cfg = load_config(str(config_path), overrides=args.override)
    configure_runtime(cfg)

    if args.out_dir is None:
        out_dir = model_outputs_root(repo, args.name) if args.sample_bank else run_dir
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    diffusion = None
    generated_bank = None
    generated_limit = None
    generated_source = None
    if args.sample_bank:
        bank_dir = resolve_sample_bank_dir(repo, args.name, args.split, args.sample_bank)
        sample_bank = SampleBank.from_dir(bank_dir)
        if sample_bank.run_name != args.name:
            raise ValueError(
                f"Sample bank {bank_dir} was created for run {sample_bank.run_name!r}, not {args.name!r}."
            )
        if sample_bank.split != args.split:
            raise ValueError(
                f"Sample bank {bank_dir} targets split {sample_bank.split!r}, "
                f"but eval was requested on split {args.split!r}."
            )

        limit = args.generated_limit if args.generated_limit is not None else sample_bank.available_n_per_class()
        if limit <= 0:
            raise ValueError(f"Sample bank {bank_dir} has no usable generated samples.")
        for class_id, available in sorted(sample_bank.generated_counts_by_class.items()):
            if int(available) < int(limit):
                raise ValueError(
                    f"Sample bank {bank_dir} only has {available} samples for class {class_id}, "
                    f"cannot satisfy generated_limit={limit}."
                )
        generated_bank = sample_bank
        generated_limit = int(limit)
        generated_source = {
            "sample_bank": {
                "schema": sample_bank.manifest.get("schema"),
                "schema_version": sample_bank.manifest.get("schema_version"),
                "bank_name": sample_bank.name,
                "split": sample_bank.split,
                "path": repo_relative_or_abs(sample_bank.root, repo),
                "created_at_utc": sample_bank.manifest.get("created_at_utc"),
                "updated_at_utc": sample_bank.manifest.get("updated_at_utc"),
                "total_n_per_class_available": int(sample_bank.available_n_per_class()),
                "generated_limit": int(limit),
                "generation": dict(sample_bank.generation),
                "conditioning_targets": dict(sample_bank.conditioning_targets),
            }
        }
    else:
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
        model = _load_model_with_arch_fallback(cfg, weights_path)
        diffusion = Diffusion(cfg)

    evaluator = TCEvaluator(cfg)
    rep = evaluator.run(
        model=model,
        diffusion=diffusion,
        out_dir=out_dir,
        tag=args.tag,
        split=args.split,
        full_test=bool(args.full_test),
        heavy=bool(args.heavy),
        show_progress=bool(args.verbose),
        generated_bank=generated_bank,
        generated_limit=generated_limit,
        generated_source=generated_source,
    )

    print("Wrote evaluation report to:", resolve_eval_root(out_dir, args.tag, args.split) / "metrics.json")
