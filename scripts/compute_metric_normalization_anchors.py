from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tc_diffusion.config import load_config
from tc_diffusion.data import build_data_backend
from tc_diffusion.evaluation.evaluator import (
    METRIC_NORMALIZATION_CONTROL_NAMES,
    _add_distributional_values_to_aggregate,
    _build_class_metric_cache,
    _build_class_metric_cache_from_real_cache,
    _build_metric_normalization_control_bt_k,
    _compute_control_aggregate_primary_raw,
    _compute_embedding_distributional_metrics,
    _compute_evaluator_fd,
    _compute_evaluator_fd_real_vs_real_reference,
    _compute_real_to_train_memorization_reference,
    _compute_real_vs_real_primary_reference,
    _default_eval_cfg,
    _encode_real_paths_by_class,
    _load_real_group_from_paths,
    _parse_metric_normalization_control_names,
    _sample_real_images_for_negative_control,
    _select_real_by_class,
)
from tc_diffusion.evaluation.metrics import DAVComputer, PolarBinner
from tc_diffusion.evaluator.features import EvaluatorEmbeddingExtractor
from tc_diffusion.sample_bank import (
    SampleBank,
    repo_relative_or_abs,
    repo_root,
    resolve_sample_bank_dir,
)


ANCHOR_SCHEMA = "tc_diffusion.metric_normalization_ranges.v1"
RANGE_DIRECTIONS = {
    "pixel_hist_w1": "lower",
    "dav_abs_gap_deg2": "lower",
    "radial_profile_mae_k": "lower",
    "cold_cloud_fraction_200K_abs_gap": "lower",
    "psd_l2": "lower",
    "eye_contrast_proxy_abs_gap": "lower",
    "evaluator_fd": "lower",
    "evaluator_embedding_diversity_closeness": "higher",
    "evaluator_embedding_coverage": "higher",
    "nearest_train_q01_distance": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute metric-normalization anchors for the model metric spider plot. "
            "The output JSON stores real-vs-real good references and mean corrupted-real "
            "bad anchors on the same metric keys used by eval.py."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to runs/<name>/config.yaml when --name is set, else configs/common.yaml.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional model run name, used to resolve a default config and sample bank.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument(
        "--real-reference-mode",
        choices=["all", "sampled"],
        default="all",
        help="Use all real samples in the split, or sample up to --real-limit-per-class.",
    )
    parser.add_argument(
        "--real-limit-per-class",
        type=int,
        default=None,
        help="Per-class real reference cap when --real-reference-mode=sampled.",
    )
    parser.add_argument(
        "--generated-n-per-class",
        type=int,
        default=None,
        help=(
            "Pseudo-generated/corrupted sample count per class. Defaults to --generated-limit, "
            "the sample-bank count, or evaluation.n_per_class_heavy."
        ),
    )
    parser.add_argument(
        "--sample-bank",
        type=str,
        default=None,
        help=(
            "Optional sample-bank reference used only to infer generated counts. "
            "Resolved like scripts/eval.py and requires --name."
        ),
    )
    parser.add_argument(
        "--generated-limit",
        type=int,
        default=None,
        help="Optional count cap when inferring generated counts from --sample-bank.",
    )
    parser.add_argument(
        "--controls",
        nargs="*",
        default=None,
        help="Negative controls. Defaults to evaluation.metric_normalization.controls.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--real-reference-reps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. Defaults to outputs/metric_normalization_anchors/"
            "<split>_<real-mode>_n<count>_seed<seed>/metric_normalization.json."
        ),
    )
    parser.add_argument(
        "--include-evaluator-features",
        dest="include_evaluator_features",
        action="store_true",
        default=True,
        help="Include evaluator-FD, diversity, coverage, and memorization anchors.",
    )
    parser.add_argument(
        "--skip-evaluator-features",
        dest="include_evaluator_features",
        action="store_false",
        help="Only compute physical/radiometric anchors.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def _configure_runtime(cfg: Dict[str, Any]) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if bool(cfg.get("training", {}).get("mixed_precision", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def _resolve_config_path(args: argparse.Namespace, repo: Path) -> Path:
    if args.config:
        path = Path(args.config)
    elif args.name:
        path = repo / "runs" / str(args.name) / "config.yaml"
    else:
        path = repo / "configs" / "common.yaml"
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _resolve_generated_counts(
    *,
    args: argparse.Namespace,
    repo: Path,
    ev: Dict[str, Any],
    num_classes: int,
) -> tuple[Dict[int, int], Dict[str, Any]]:
    if args.generated_limit is not None and int(args.generated_limit) <= 0:
        raise ValueError("--generated-limit must be > 0 when provided.")
    if args.generated_n_per_class is not None and int(args.generated_n_per_class) <= 0:
        raise ValueError("--generated-n-per-class must be > 0 when provided.")

    source: Dict[str, Any] = {"mode": "explicit_or_configured_count"}
    if args.sample_bank:
        if not args.name:
            raise ValueError("--sample-bank requires --name so the bank can be resolved.")
        bank_dir = resolve_sample_bank_dir(repo, str(args.name), str(args.split), str(args.sample_bank))
        bank = SampleBank.from_dir(bank_dir)
        if bank.run_name != str(args.name):
            raise ValueError(
                f"Sample bank {bank_dir} was created for run {bank.run_name!r}, not {args.name!r}."
            )
        if bank.split != str(args.split):
            raise ValueError(
                f"Sample bank {bank_dir} targets split {bank.split!r}, not requested split {args.split!r}."
            )
        if args.generated_n_per_class is not None:
            n_per = int(args.generated_n_per_class)
        elif args.generated_limit is not None:
            n_per = int(args.generated_limit)
        else:
            n_per = int(bank.available_n_per_class())
        insufficient = {
            int(c): int(n)
            for c, n in bank.generated_counts_by_class.items()
            if int(c) < num_classes and int(n) < n_per
        }
        if insufficient:
            raise ValueError(
                f"Sample bank {bank_dir} cannot satisfy n_per_class={n_per}; insufficient counts: {insufficient}"
            )
        source = {
            "mode": "sample_bank_count",
            "sample_bank": {
                "schema": bank.manifest.get("schema"),
                "schema_version": bank.manifest.get("schema_version"),
                "bank_name": bank.name,
                "split": bank.split,
                "path": repo_relative_or_abs(bank.root, repo),
                "created_at_utc": bank.manifest.get("created_at_utc"),
                "updated_at_utc": bank.manifest.get("updated_at_utc"),
                "total_n_per_class_available": int(bank.available_n_per_class()),
                "generated_limit": int(n_per),
            },
        }
    elif args.generated_n_per_class is not None:
        n_per = int(args.generated_n_per_class)
    elif args.generated_limit is not None:
        n_per = int(args.generated_limit)
    else:
        n_per = int(ev.get("n_per_class_heavy", 100))

    return {int(c): int(n_per) for c in range(num_classes)}, source


def _resolve_output_path(
    *,
    args: argparse.Namespace,
    repo: Path,
    split: str,
    real_mode: str,
    n_per: int,
    seed: int,
) -> Path:
    if args.output:
        out = Path(args.output)
        return out if out.is_absolute() else repo / out
    anchor_id = f"{split}_{real_mode}_n{int(n_per)}_seed{int(seed)}"
    return repo / "outputs" / "metric_normalization_anchors" / anchor_id / "metric_normalization.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _metric_ranges(good_values: Dict[str, float], bad_values: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(good_values) & set(bad_values)):
        good = float(good_values[key])
        bad = float(bad_values[key])
        direction = RANGE_DIRECTIONS.get(key, "lower")
        usable = (
            bool(np.isfinite(good) and np.isfinite(bad))
            and abs(float(good) - float(bad)) > 1.0e-12
            and ((direction == "lower" and bad > good) or (direction == "higher" and good > bad))
        )
        out[key] = {
            "direction": direction,
            "best": good,
            "worst": bad,
            "usable_for_0_1_score": usable,
        }
    return out


def main() -> None:
    args = parse_args()
    repo = repo_root()
    config_path = _resolve_config_path(args, repo)
    cfg = load_config(str(config_path), overrides=args.override)
    ev = _default_eval_cfg(cfg)
    _configure_runtime(cfg)

    split = str(args.split).strip().lower()
    data_cfg = cfg["data"]
    bt_min_k = float(data_cfg["bt_min_k"])
    bt_max_k = float(data_cfg["bt_max_k"])
    image_size = int(data_cfg.get("image_size", 256))
    num_classes = int(cfg["conditioning"]["num_ss_classes"])

    norm_cfg = dict(ev.get("metric_normalization", {}))
    seed = int(args.seed if args.seed is not None else norm_cfg.get("seed", ev.get("seed", 123)))
    real_reference_reps = int(
        args.real_reference_reps
        if args.real_reference_reps is not None
        else norm_cfg.get("real_reference_reps", 200)
    )
    if real_reference_reps < 0:
        raise ValueError("--real-reference-reps must be >= 0.")

    controls = _parse_metric_normalization_control_names(
        args.controls if args.controls is not None else norm_cfg.get("controls", list(METRIC_NORMALIZATION_CONTROL_NAMES))
    )
    if not controls:
        raise ValueError("At least one negative control is required.")

    gaussian_sigma_px = float(norm_cfg.get("gaussian_blur_sigma_px", 6.0))
    gaussian_kernel_size = int(norm_cfg.get("gaussian_blur_kernel_size", 25))
    bt_warm_shift_k = float(norm_cfg.get("bt_warm_shift_k", 15.0))
    center_jitter_max_px = int(norm_cfg.get("center_jitter_max_px", 24))

    generated_counts_by_class, generated_source = _resolve_generated_counts(
        args=args,
        repo=repo,
        ev=ev,
        num_classes=num_classes,
    )
    n_per_values = sorted(set(generated_counts_by_class.values()))
    n_per_for_name = int(n_per_values[0]) if len(n_per_values) == 1 else int(max(n_per_values))

    if args.real_reference_mode == "sampled":
        real_limit = int(
            args.real_limit_per_class
            if args.real_limit_per_class is not None
            else ev.get("n_per_class_heavy", n_per_for_name)
        )
        if real_limit <= 0:
            raise ValueError("--real-limit-per-class must be > 0 for sampled real references.")
        use_all_real = False
    else:
        real_limit = 1
        use_all_real = True

    output_path = _resolve_output_path(
        args=args,
        repo=repo,
        split=split,
        real_mode=str(args.real_reference_mode),
        n_per=n_per_for_name,
        seed=seed,
    )

    real_backend = build_data_backend(data_cfg)
    real_paths_by_class, _ = _select_real_by_class(
        cfg,
        n_per_class=real_limit,
        seed=int(ev.get("real_seed", seed)),
        split=split,
        use_all=use_all_real,
    )
    valid_class_ids = [
        int(c)
        for c in range(num_classes)
        if c in real_paths_by_class and int(generated_counts_by_class.get(c, 0)) > 0
    ]
    if not valid_class_ids:
        raise ValueError("No overlapping real-reference classes and generated counts were available.")

    evaluator_fd_cfg = dict(ev.get("evaluator_feature_metric", {}))
    distributional_cfg = dict(evaluator_fd_cfg.get("distributional", {}))
    covariance_eps = float(evaluator_fd_cfg.get("covariance_eps", 1.0e-6))
    coverage_k = int(distributional_cfg.get("coverage_k", 3))
    distance_block_size = int(distributional_cfg.get("distance_block_size", 256))
    memorization_train_split = str(distributional_cfg.get("memorization_train_split", "train")).strip().lower()
    raw_train_limit = distributional_cfg.get("memorization_train_limit_per_class", None)
    memorization_train_limit = None if raw_train_limit is None else int(raw_train_limit)
    memorization_train_seed = int(distributional_cfg.get("memorization_train_seed", ev.get("real_seed", seed)))

    feature_extractor = None
    if bool(args.include_evaluator_features):
        feature_extractor = EvaluatorEmbeddingExtractor.from_run(
            run_name=str(evaluator_fd_cfg.get("run_name", "evaluator_cat4plus_mild_rebalanced_s035")),
            config_path=evaluator_fd_cfg.get("config_path"),
            weights_path=evaluator_fd_cfg.get("weights_path"),
            weights_name=str(evaluator_fd_cfg.get("weights_name", "best_tail")),
            embedding_layer=str(evaluator_fd_cfg.get("embedding_layer", "embedding_silu")),
            batch_size=int(evaluator_fd_cfg.get("batch_size", 32)),
        )

    binner = PolarBinner(image_size, image_size, int(ev["profile_bins"]), 360)
    dav_computer = DAVComputer(
        image_size,
        image_size,
        pixel_size_km=float(ev.get("dav_pixel_size_km", 8.0)),
        radius_km=float(ev.get("dav_radius_km", 300.0)),
        center_region_size=int(ev.get("dav_center_region_size", 3)),
    )

    class_caches = {}
    metric_control_caches: Dict[str, Dict[int, Any]] = {name: {} for name in controls}
    metric_control_embeddings: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in controls}
    real_embeddings_by_class: Dict[int, np.ndarray] = {}
    source_sampling_by_class: Dict[int, Dict[str, Any]] = {}
    transformations: Dict[str, Dict[str, Any]] = {}
    real_counts_by_class: Dict[int, int] = {}

    iterator = valid_class_ids
    if args.verbose:
        iterator = tqdm(valid_class_ids, desc="Building anchor class summaries", unit="class")
    for class_id in iterator:
        real_k = _load_real_group_from_paths(
            real_backend,
            real_paths_by_class[class_id],
            bt_min_k=bt_min_k,
            bt_max_k=bt_max_k,
        )
        real_counts_by_class[class_id] = int(real_k.shape[0])
        draw_n = int(generated_counts_by_class[class_id])
        class_rng = np.random.default_rng(int(seed + 104729 * (class_id + 1)))
        pseudo_gen_k, source_meta = _sample_real_images_for_negative_control(
            real_k,
            draw_n=draw_n,
            rng=class_rng,
        )
        source_sampling_by_class[class_id] = source_meta
        cache = _build_class_metric_cache(
            real_k=real_k,
            gen_k_raw=pseudo_gen_k,
            binner=binner,
            dav_computer=dav_computer,
            psd_bins=int(ev["psd_bins"]),
            bt_min_k=bt_min_k,
            bt_max_k=bt_max_k,
        )
        class_caches[class_id] = cache

        if feature_extractor is not None:
            real_embeddings_by_class[class_id] = feature_extractor.encode_bt_k(real_k)

        for control_idx, control_name in enumerate(controls):
            control_seed = int(seed + 1009 * (control_idx + 1) + 104729 * (class_id + 1))
            control_k, transformation = _build_metric_normalization_control_bt_k(
                control_name,
                pseudo_gen_k,
                seed=control_seed,
                gaussian_sigma_px=gaussian_sigma_px,
                gaussian_kernel_size=gaussian_kernel_size,
                bt_warm_shift_k=bt_warm_shift_k,
                center_jitter_max_px=center_jitter_max_px,
            )
            transformations.setdefault(control_name, transformation)
            metric_control_caches[control_name][class_id] = _build_class_metric_cache_from_real_cache(
                real_cache=cache,
                gen_k_raw=control_k,
                binner=binner,
                dav_computer=dav_computer,
                psd_bins=int(ev["psd_bins"]),
                bt_min_k=bt_min_k,
                bt_max_k=bt_max_k,
            )
            if feature_extractor is not None:
                metric_control_embeddings[control_name][class_id] = feature_extractor.encode_bt_k(control_k)

    good_reference = _compute_real_vs_real_primary_reference(
        class_caches=class_caches,
        reps=real_reference_reps,
        seed=seed,
    )
    good_values = dict(good_reference.get("values", {})) if isinstance(good_reference, dict) else {}

    evaluator_fd_real_vs_real = None
    train_embeddings_by_class = None
    real_to_train_reference = None
    if feature_extractor is not None and real_embeddings_by_class:
        if real_reference_reps > 0:
            evaluator_fd_real_vs_real = _compute_evaluator_fd_real_vs_real_reference(
                real_embeddings_by_class=real_embeddings_by_class,
                generated_counts_by_class=generated_counts_by_class,
                covariance_eps=covariance_eps,
                reps=real_reference_reps,
                seed=seed,
            )
            fd_summary = evaluator_fd_real_vs_real.get("summary", {})
            if isinstance(fd_summary, dict) and fd_summary.get("q50") is not None:
                good_values["evaluator_fd"] = float(fd_summary["q50"])
        good_values.setdefault("evaluator_fd", 0.0)
        good_values.setdefault("evaluator_embedding_diversity_ratio", 1.0)
        good_values.setdefault("evaluator_embedding_diversity_closeness", 1.0)
        good_values.setdefault("evaluator_embedding_coverage", 1.0)

        if memorization_train_split == "train":
            train_paths_by_class, _ = _select_real_by_class(
                cfg,
                n_per_class=(
                    int(memorization_train_limit)
                    if memorization_train_limit is not None
                    else 1
                ),
                seed=memorization_train_seed,
                split=memorization_train_split,
                use_all=memorization_train_limit is None,
            )
            train_paths_by_class = {
                int(c): list(paths)
                for c, paths in train_paths_by_class.items()
                if int(c) in real_embeddings_by_class and paths
            }
            if train_paths_by_class:
                train_embeddings_by_class = _encode_real_paths_by_class(
                    backend=real_backend,
                    paths_by_class=train_paths_by_class,
                    feature_extractor=feature_extractor,
                    bt_min_k=bt_min_k,
                    bt_max_k=bt_max_k,
                    load_batch_size=int(evaluator_fd_cfg.get("batch_size", 32)),
                    show_progress=bool(args.verbose),
                    progress_desc="Encoding train references",
                )
                real_to_train_reference = _compute_real_to_train_memorization_reference(
                    real_embeddings_by_class=real_embeddings_by_class,
                    train_embeddings_by_class=train_embeddings_by_class,
                    distance_block_size=distance_block_size,
                )
                if isinstance(real_to_train_reference, dict):
                    macro = real_to_train_reference.get("macro", {})
                    if isinstance(macro, dict) and macro.get("q01_real_to_train_nn") is not None:
                        good_values["nearest_train_q01_distance"] = float(macro["q01_real_to_train_nn"])

    control_descriptions = {
        "gaussian_blur": "Real reference images degraded by Gaussian blur.",
        "pixel_shuffle": "Real reference images with per-image BT histograms preserved but spatial layout shuffled.",
        "bt_warm_shift": "Real reference images shifted warmer by a fixed Kelvin offset.",
        "center_jitter": "Real reference images translated away from the assumed storm center.",
        "mode_collapse": "A single same-class real reference image repeated to the generated count.",
        "train_copy": "Exact same-class train embeddings used as generated samples for memorization-risk anchoring.",
    }
    controls_out: Dict[str, Any] = {}
    for control_name in controls:
        control_caches = metric_control_caches.get(control_name, {})
        if not control_caches:
            continue
        aggregate = _compute_control_aggregate_primary_raw(control_caches)
        node: Dict[str, Any] = {
            "description": control_descriptions.get(control_name, ""),
            "source": "corrupted_real_reference_images",
            "transformation": transformations.get(control_name, {"name": control_name}),
            "aggregate_primary_raw": aggregate,
            "valid_class_ids": [int(c) for c in sorted(control_caches)],
        }
        control_embeddings = metric_control_embeddings.get(control_name, {})
        if feature_extractor is not None and real_embeddings_by_class and control_embeddings:
            fd = _compute_evaluator_fd(
                real_embeddings_by_class=real_embeddings_by_class,
                gen_embeddings_by_class=control_embeddings,
                covariance_eps=covariance_eps,
            )
            node["evaluator_fd"] = fd
            aggregate["evaluator_fd"] = float(fd["value"])

            distributional = _compute_embedding_distributional_metrics(
                real_embeddings_by_class=real_embeddings_by_class,
                gen_embeddings_by_class=control_embeddings,
                train_embeddings_by_class=None,
                coverage_k=coverage_k,
                distance_block_size=distance_block_size,
            )
            node["distributional_feature_metric"] = distributional
            _add_distributional_values_to_aggregate(aggregate, distributional)
        controls_out[control_name] = node

    if train_embeddings_by_class is not None:
        controls_out["train_copy"] = {
            "description": control_descriptions["train_copy"],
            "source": "same_class_train_embeddings",
            "applies_to": ["nearest_train_q01_distance"],
            "aggregate_primary_raw": {"nearest_train_q01_distance": 0.0},
        }

    control_aggregates = {
        name: node.get("aggregate_primary_raw", {})
        for name, node in controls_out.items()
        if isinstance(node, dict)
    }
    bad_values: Dict[str, float] = {}
    for key in sorted({
        key
        for aggregate in control_aggregates.values()
        if isinstance(aggregate, dict)
        for key in aggregate.keys()
    }):
        if key == "nearest_train_q01_distance" and "train_copy" in control_aggregates:
            bad_values[key] = 0.0
            continue
        values = [
            float(aggregate[key])
            for name, aggregate in control_aggregates.items()
            if name != "train_copy"
            and isinstance(aggregate, dict)
            and aggregate.get(key) is not None
            and np.isfinite(float(aggregate[key]))
        ]
        if values:
            bad_values[key] = float(np.mean(values))

    metric_normalization = {
        "enabled": True,
        "schema": "tc_diffusion.metric_normalization.v1",
        "source": "precomputed_corrupted_real_reference_images",
        "controls_requested": list(controls),
        "match_generated_counts": True,
        "seed": int(seed),
        "real_reference_reps": int(real_reference_reps),
        "bad_anchor_aggregation": "mean",
        "good_reference": {
            "description": (
                "Finite-sample real-vs-real reference for unbounded physical distances; "
                "ideal bounded references for diversity/coverage when needed."
            ),
            "values": good_values,
            "physical_real_vs_real": good_reference,
            "evaluator_fd_real_vs_real": evaluator_fd_real_vs_real,
            "real_to_train_memorization": real_to_train_reference,
        },
        "negative_controls": {
            "controls": controls_out,
            "source_sampling_by_class": {
                str(c): dict(meta) for c, meta in sorted(source_sampling_by_class.items())
            },
        },
        "bad_anchor": {
            "aggregation": "mean",
            "values": bad_values,
        },
    }

    output = {
        "schema": ANCHOR_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": repo_relative_or_abs(config_path, repo),
        "split": split,
        "real_reference_mode": str(args.real_reference_mode),
        "real_counts_by_class": {str(c): int(real_counts_by_class[c]) for c in sorted(real_counts_by_class)},
        "generated_counts_by_class": {
            str(c): int(generated_counts_by_class[c]) for c in sorted(generated_counts_by_class)
        },
        "generated_source": generated_source,
        "evaluator_feature_metric": {
            "included": feature_extractor is not None,
            "run_name": str(evaluator_fd_cfg.get("run_name", "evaluator_cat4plus_mild_rebalanced_s035")),
            "weights_name": str(evaluator_fd_cfg.get("weights_name", "best_tail")),
            "embedding_layer": str(evaluator_fd_cfg.get("embedding_layer", "embedding_silu")),
            "coverage_k": int(coverage_k),
            "distance_block_size": int(distance_block_size),
            "covariance_eps": float(covariance_eps),
        },
        "metric_ranges": _metric_ranges(good_values, bad_values),
        "metric_normalization": metric_normalization,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"Wrote metric normalization anchors to: {output_path}")


if __name__ == "__main__":
    main()
