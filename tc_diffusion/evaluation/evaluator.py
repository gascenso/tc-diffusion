# tc_diffusion/evaluation/evaluator.py
from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from .metrics import (
    DAVComputer,
    PolarBinner,
    cold_cloud_fraction,
    denorm_bt,
    eye_contrast_proxy,
    flatten_features_for_diversity,
    frechet_distance_from_stats,
    js_divergence,
    psd_radial_batch,
    radial_profile_batch,
    rbf_mmd2,
    summary_stats,
    weighted_mean_and_cov,
    wasserstein1_from_hist,
)
from .plots import plot_hist_overlay, plot_psd, plot_radial_profiles
from ..data import build_data_backend, load_dataset_index, load_split_file_set
from ..diffusion import resolve_sampling_timestep_schedule
from ..evaluator.features import EvaluatorEmbeddingExtractor
from ..plotting import save_real_generated_comparison_grid
from ..sample_bank import SampleBank
from ..sampling_guidance import sampling_guidance_summary


PRIMARY_RAW_AGG_KEYS = (
    "pixel_hist_js",
    "pixel_hist_w1",
    "cold_cloud_fraction_200K_abs_gap",
    "dav_abs_gap_deg2",
    "eye_contrast_proxy_abs_gap",
    "psd_l2",
    "feature_mmd2",
    "diversity_feature_space_abs_gap",
    "gen_exceedance_rate_total",
)

REPORT_SCHEMA_VERSION = 9
PAPER_READY_PIXEL_ARTIFACT_SCHEMA = "tc_diffusion.paper_ready.pixel_plausibility.v2"
PAPER_READY_RADIAL_BT_ARTIFACT_SCHEMA = "tc_diffusion.paper_ready.radial_bt_profile.v1"
PAPER_READY_DAV_ARTIFACT_SCHEMA = "tc_diffusion.paper_ready.dav.v1"
PAPER_READY_COLD_CLOUD_ARTIFACT_SCHEMA = "tc_diffusion.paper_ready.cold_cloud_fraction.v1"
PER_CLASS_METRICS_CSV_SCHEMA = "tc_diffusion.per_class_metrics.v1"
EVALUATOR_FD_NEGATIVE_CONTROL_NAMES = ("gaussian_blur", "pixel_shuffle")


@dataclass
class ClassMetricCache:
    bins: np.ndarray
    real_hist_counts: np.ndarray
    gen_raw_hist_counts: np.ndarray
    gen_post_hist_counts: np.ndarray
    real_mean_profiles: np.ndarray
    real_azstd_profiles: np.ndarray
    gen_raw_mean_profiles: np.ndarray
    gen_raw_azstd_profiles: np.ndarray
    gen_post_mean_profiles: np.ndarray
    gen_post_azstd_profiles: np.ndarray
    real_psd_profiles: np.ndarray
    gen_raw_psd_profiles: np.ndarray
    gen_post_psd_profiles: np.ndarray
    real_cold: np.ndarray
    gen_raw_cold: np.ndarray
    gen_post_cold: np.ndarray
    real_dav: np.ndarray
    gen_raw_dav: np.ndarray
    gen_post_dav: np.ndarray
    real_eye: np.ndarray
    gen_raw_eye: np.ndarray
    gen_post_eye: np.ndarray
    real_features: np.ndarray
    gen_raw_features: np.ndarray
    gen_post_features: np.ndarray
    feature_gamma_raw: float
    feature_gamma_post: float
    real_pixel_mean: np.ndarray
    gen_raw_pixel_mean: np.ndarray
    gen_post_pixel_mean: np.ndarray
    gen_raw_exceed_below: np.ndarray
    gen_raw_exceed_above: np.ndarray
    gen_raw_exceed_total: np.ndarray
    pixel_stats_real: Dict[str, float]
    pixel_stats_gen_raw: Dict[str, float]
    pixel_stats_gen_post: Dict[str, float]
    gen_exceedance_rate: Dict[str, float]


# -----------------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------------


def _default_eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = dict(cfg.get("evaluation", {}))
    ev.setdefault("enabled", True)
    ev.setdefault("every_epochs", 5)
    ev.setdefault("heavy_every_epochs", 25)
    ev.setdefault("n_plot_per_group", 25)
    ev.setdefault("n_per_class_light", ev["n_plot_per_group"])
    ev.setdefault("n_per_class_heavy", 100)
    ev.setdefault("gen_batch_size", None)
    ev.setdefault("guidance_scale", 0.0)
    ev.setdefault("sampler", "dpmpp_2m")
    ev.setdefault("sampling_steps", 25)
    ev.setdefault("timestep_schedule", resolve_sampling_timestep_schedule(cfg))
    ev.setdefault("ddim_steps", None)
    ev.setdefault("ddim_eta", 0.0)
    ev.setdefault("seed", 123)
    ev.setdefault("real_seed", 123)
    ev.setdefault("profile_bins", 96)
    ev.setdefault("psd_bins", 96)
    ev.setdefault("bootstrap_reps", 500)
    ev.setdefault("bootstrap_reps_full_test", 0)
    ev.setdefault("bootstrap_ci_level", 0.95)
    phys_cfg = cfg.get("physics_loss", {})
    ev.setdefault("dav_radius_km", float(phys_cfg.get("dav_radius_km", 300.0)))
    ev.setdefault("dav_pixel_size_km", float(phys_cfg.get("pixel_size_km", 8.0)))
    ev.setdefault("dav_center_region_size", 3)
    evaluator_fd_cfg = dict(ev.get("evaluator_feature_metric", {}))
    evaluator_fd_cfg.setdefault("enabled", "auto")
    evaluator_fd_cfg.setdefault("run_name", "evaluator_cat4plus_mild_rebalanced_s035")
    evaluator_fd_cfg.setdefault("config_path", None)
    evaluator_fd_cfg.setdefault("weights_path", None)
    evaluator_fd_cfg.setdefault("weights_name", "best_tail")
    evaluator_fd_cfg.setdefault("embedding_layer", "embedding_silu")
    evaluator_fd_cfg.setdefault("batch_size", max(8, int(cfg.get("data", {}).get("batch_size", 8))))
    evaluator_fd_cfg.setdefault("covariance_eps", 1.0e-6)
    evaluator_fd_cfg.setdefault("null_reference_reps", 0)
    evaluator_fd_cfg.setdefault("null_reference_seed", int(ev.get("seed", 123)))
    negative_controls_cfg = dict(evaluator_fd_cfg.get("negative_controls", {}))
    negative_controls_cfg.setdefault("enabled", "auto")
    negative_controls_cfg.setdefault("controls", list(EVALUATOR_FD_NEGATIVE_CONTROL_NAMES))
    negative_controls_cfg.setdefault("seed", int(ev.get("seed", 123)))
    negative_controls_cfg.setdefault("match_generated_counts", True)
    negative_controls_cfg.setdefault("gaussian_blur_sigma_px", 6.0)
    negative_controls_cfg.setdefault("gaussian_blur_kernel_size", 25)
    evaluator_fd_cfg["negative_controls"] = negative_controls_cfg
    ev["evaluator_feature_metric"] = evaluator_fd_cfg
    return ev


def _write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def resolve_eval_root(out_dir: Path, tag: str, split: str) -> Path:
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Evaluation split must be 'val' or 'test', got {split!r}")
    if split == "val":
        return out_dir / "eval" / tag
    return out_dir / "eval" / split / tag


def _ss_class_midpoint_kt(ss_cat: int) -> float:
    cls = int(ss_cat)
    if cls == 0:
        return 49.0
    if cls == 1:
        return 73.0
    if cls == 2:
        return 89.0
    if cls == 3:
        return 104.0
    if cls == 4:
        return 124.5
    return 145.0


def _resolve_eval_wind_target_kt(cfg: Dict[str, Any], ss_cat: int) -> float:
    cond_cfg = cfg.get("conditioning", {})
    targets = cond_cfg.get("eval_wind_targets_kt")
    if isinstance(targets, dict):
        key = str(int(ss_cat))
        if key in targets:
            try:
                return float(targets[key])
            except Exception:
                pass
    return _ss_class_midpoint_kt(int(ss_cat))


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if den <= 0.0:
        return None
    return float(np.sum(x * y) / den)


def _flatten_metric_tree(tree: Any, prefix: str = "", out: Dict[str, float] | None = None) -> Dict[str, float]:
    if out is None:
        out = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metric_tree(value, prefix=child_prefix, out=out)
        return out

    out[prefix] = np.nan if tree is None else float(tree)
    return out


def _append_bootstrap_sample(samples_by_path: Dict[str, list[float]], tree: Dict[str, Any]):
    flat = _flatten_metric_tree(tree)
    for path, value in flat.items():
        samples_by_path.setdefault(path, []).append(value)


def _ci_bounds(values: list[float], ci_level: float) -> Dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"low": None, "high": None}

    alpha = 0.5 * (1.0 - float(ci_level))
    return {
        "low": float(np.quantile(arr, alpha)),
        "high": float(np.quantile(arr, 1.0 - alpha)),
    }


def _build_ci_tree(point_tree: Any, samples_by_path: Dict[str, list[float]], ci_level: float, prefix: str = "") -> Any:
    if isinstance(point_tree, dict):
        out = {}
        for key, value in point_tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out[key] = _build_ci_tree(value, samples_by_path, ci_level, prefix=child_prefix)
        return out

    return _ci_bounds(samples_by_path.get(prefix, []), ci_level)


# -----------------------------------------------------------------------------
# real data loading
# -----------------------------------------------------------------------------


def _load_one_bt_k(backend, rel_path: str, bt_min_k: float, bt_max_k: float) -> np.ndarray:
    bt = backend.load_bt(rel_path)
    bt = np.nan_to_num(bt, nan=bt_min_k)
    bt = np.clip(bt, bt_min_k, bt_max_k)
    return bt


def _select_real_by_class(
    cfg: Dict[str, Any],
    n_per_class: int,
    seed: int,
    split: str,
    use_all: bool = False,
) -> Tuple[Dict[int, list[str]], Dict[int, np.ndarray]]:
    """Select real BT file paths and wind speeds, grouped by SS class."""
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Real-data evaluation split must be 'val' or 'test', got {split!r}")

    data_cfg = cfg["data"]
    index_path = Path(data_cfg["dataset_index"])
    split_dir = Path(data_cfg["split_dir"])

    class_to_files, sample_meta = load_dataset_index(index_path, return_sample_meta=True)
    allowed = load_split_file_set(split_dir, split=split)

    rng = np.random.default_rng(seed)
    out_paths: Dict[int, list[str]] = {}
    out_winds: Dict[int, np.ndarray] = {}

    for c, rels in sorted(class_to_files.items()):
        rels = [r for r in rels if r in allowed]
        if not rels:
            continue

        if use_all:
            pick = rng.permutation(len(rels))
        else:
            k = min(n_per_class, len(rels))
            pick = rng.choice(len(rels), size=k, replace=False)

        selected_rels = []
        winds = []
        for idx in pick:
            rel = rels[int(idx)]
            selected_rels.append(rel)
            meta = sample_meta.get(rel, {})
            w = meta.get("wmo_wind_kt")
            if w is not None and np.isfinite(float(w)):
                winds.append(float(w))
            else:
                winds.append(_ss_class_midpoint_kt(c))

        out_paths[int(c)] = selected_rels
        out_winds[int(c)] = np.array(winds, dtype=np.float32)

    return out_paths, out_winds


def _load_real_group_from_paths(
    backend,
    rel_paths: list[str],
    *,
    bt_min_k: float,
    bt_max_k: float,
) -> np.ndarray:
    if not rel_paths:
        raise ValueError("Expected at least one real-data path when loading a class group.")
    imgs = [_load_one_bt_k(backend, rel, bt_min_k, bt_max_k) for rel in rel_paths]
    return np.stack(imgs, axis=0)


# -----------------------------------------------------------------------------
# cached metric computation
# -----------------------------------------------------------------------------


def _hist_counts_batch(imgs: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros((imgs.shape[0], bins.shape[0] - 1), dtype=np.float64)
    for i in range(imgs.shape[0]):
        out[i], _ = np.histogram(imgs[i].reshape(-1), bins=bins)
    return out


def _hist_density_from_counts(counts: np.ndarray) -> np.ndarray:
    h = np.asarray(counts, dtype=np.float64)
    if h.ndim == 2:
        h = h.sum(axis=0)
    return h / (h.sum() + 1e-12)


def _mean_pair_dist(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return 0.0

    block_size = 128
    total = 0.0
    norms = np.sum(z * z, axis=1)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        zi = z[i0:i1]
        zi_norm = norms[i0:i1][:, None]
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            zj = z[j0:j1]
            zj_norm = norms[j0:j1][None, :]
            d2 = zi_norm + zj_norm - 2.0 * (zi @ zj.T)
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            if i0 == j0:
                total += float(d2.sum() - np.trace(d2))
            else:
                total += 2.0 * float(d2.sum())

    return float(total / (n * (n - 1)))


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y]).astype(np.float64)
    if z.shape[0] < 2:
        return 1.0

    rng = np.random.default_rng(0)
    idx = rng.choice(z.shape[0], size=min(200, z.shape[0]), replace=False)
    zs = z[idx]
    d2 = np.sum((zs[:, None, :] - zs[None, :, :]) ** 2, axis=-1)
    positive = d2[d2 > 0]
    if positive.size == 0:
        return 1.0
    med = float(np.median(positive))
    return 1.0 / (2.0 * (med + 1e-12))


def _compute_feature_metrics(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    gamma: float,
) -> Tuple[float, float, float]:
    mmd2 = rbf_mmd2(real_features, gen_features, gamma=gamma)
    div_real = _mean_pair_dist(real_features)
    div_gen = _mean_pair_dist(gen_features)
    return float(mmd2), float(div_real), float(div_gen)


def _compute_structural_suite(
    *,
    bins: np.ndarray,
    real_hist_counts: np.ndarray,
    gen_hist_counts: np.ndarray,
    real_mean_profiles: np.ndarray,
    gen_mean_profiles: np.ndarray,
    real_azstd_profiles: np.ndarray,
    gen_azstd_profiles: np.ndarray,
    real_psd_profiles: np.ndarray,
    gen_psd_profiles: np.ndarray,
    real_cold: np.ndarray,
    gen_cold: np.ndarray,
    real_dav: np.ndarray,
    gen_dav: np.ndarray,
    real_eye: np.ndarray,
    gen_eye: np.ndarray,
    real_features: np.ndarray,
    gen_features: np.ndarray,
    feature_gamma: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rh = _hist_density_from_counts(real_hist_counts)
    gh = _hist_density_from_counts(gen_hist_counts)
    js = js_divergence(rh, gh)
    w1 = wasserstein1_from_hist(rh, gh, bin_edges=bins)

    r_mean_mu = real_mean_profiles.mean(axis=0)
    r_mean_sd = real_mean_profiles.std(axis=0)
    g_mean_mu = gen_mean_profiles.mean(axis=0)
    g_mean_sd = gen_mean_profiles.std(axis=0)

    r_az_mu = real_azstd_profiles.mean(axis=0)
    r_az_sd = real_azstd_profiles.std(axis=0)
    g_az_mu = gen_azstd_profiles.mean(axis=0)
    g_az_sd = gen_azstd_profiles.std(axis=0)

    r_psd_mu = real_psd_profiles.mean(axis=0)
    r_psd_sd = real_psd_profiles.std(axis=0)
    g_psd_mu = gen_psd_profiles.mean(axis=0)
    g_psd_sd = gen_psd_profiles.std(axis=0)

    mmd2, div_real, div_gen = _compute_feature_metrics(real_features, gen_features, feature_gamma)

    metrics = {
        "pixel_hist_js": float(js),
        "pixel_hist_w1": float(w1),
        "cold_cloud_fraction_200K": {
            "real_mean": float(real_cold.mean()),
            "real_std": float(real_cold.std()),
            "gen_mean": float(gen_cold.mean()),
            "gen_std": float(gen_cold.std()),
        },
        "dav_deg2": {
            "real_mean": float(real_dav.mean()),
            "real_std": float(real_dav.std()),
            "gen_mean": float(gen_dav.mean()),
            "gen_std": float(gen_dav.std()),
        },
        "eye_contrast_proxy": {
            "real_mean": float(real_eye.mean()),
            "real_std": float(real_eye.std()),
            "gen_mean": float(gen_eye.mean()),
            "gen_std": float(gen_eye.std()),
        },
        "psd_l2": float(((r_psd_mu - g_psd_mu) ** 2).mean()),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space": {
            "real": float(div_real),
            "gen": float(div_gen),
        },
    }

    curves = {
        "bt_bins": bins.tolist(),
        "hist_real": rh.tolist(),
        "hist_gen": gh.tolist(),
        "radial_mean_real_mu": r_mean_mu.tolist(),
        "radial_mean_real_sd": r_mean_sd.tolist(),
        "radial_mean_gen_mu": g_mean_mu.tolist(),
        "radial_mean_gen_sd": g_mean_sd.tolist(),
        "radial_azstd_real_mu": r_az_mu.tolist(),
        "radial_azstd_real_sd": r_az_sd.tolist(),
        "radial_azstd_gen_mu": g_az_mu.tolist(),
        "radial_azstd_gen_sd": g_az_sd.tolist(),
        "psd_real_mu": r_psd_mu.tolist(),
        "psd_real_sd": r_psd_sd.tolist(),
        "psd_gen_mu": g_psd_mu.tolist(),
        "psd_gen_sd": g_psd_sd.tolist(),
    }
    return metrics, curves


def _build_class_metric_cache(
    *,
    real_k: np.ndarray,
    gen_k_raw: np.ndarray,
    binner: PolarBinner,
    dav_computer: DAVComputer,
    psd_bins: int,
    bt_min_k: float,
    bt_max_k: float,
) -> ClassMetricCache:
    bins = np.linspace(bt_min_k, bt_max_k, 129)
    gen_k_post = np.clip(gen_k_raw, bt_min_k, bt_max_k)

    real_hist_counts = _hist_counts_batch(real_k, bins)
    gen_raw_hist_counts = _hist_counts_batch(gen_k_raw, bins)
    gen_post_hist_counts = _hist_counts_batch(gen_k_post, bins)

    real_mean_profiles, real_azstd_profiles = radial_profile_batch(real_k, binner)
    gen_raw_mean_profiles, gen_raw_azstd_profiles = radial_profile_batch(gen_k_raw, binner)
    gen_post_mean_profiles, gen_post_azstd_profiles = radial_profile_batch(gen_k_post, binner)

    real_psd_profiles = psd_radial_batch(real_k, psd_bins)
    gen_raw_psd_profiles = psd_radial_batch(gen_k_raw, psd_bins)
    gen_post_psd_profiles = psd_radial_batch(gen_k_post, psd_bins)

    real_cold = cold_cloud_fraction(real_k, threshold_k=200.0)
    gen_raw_cold = cold_cloud_fraction(gen_k_raw, threshold_k=200.0)
    gen_post_cold = cold_cloud_fraction(gen_k_post, threshold_k=200.0)

    real_dav = dav_computer.batch(real_k)
    gen_raw_dav = dav_computer.batch(gen_k_raw)
    gen_post_dav = dav_computer.batch(gen_k_post)

    real_eye = eye_contrast_proxy(real_mean_profiles)
    gen_raw_eye = eye_contrast_proxy(gen_raw_mean_profiles)
    gen_post_eye = eye_contrast_proxy(gen_post_mean_profiles)

    real_features, feat_scaler = flatten_features_for_diversity(real_mean_profiles, real_psd_profiles)
    gen_raw_features, _ = flatten_features_for_diversity(
        gen_raw_mean_profiles,
        gen_raw_psd_profiles,
        scaler=feat_scaler,
    )
    gen_post_features, _ = flatten_features_for_diversity(
        gen_post_mean_profiles,
        gen_post_psd_profiles,
        scaler=feat_scaler,
    )

    feature_gamma_raw = _median_heuristic_gamma(real_features, gen_raw_features)
    feature_gamma_post = _median_heuristic_gamma(real_features, gen_post_features)

    gen_raw_exceed_below = np.mean(gen_k_raw < bt_min_k, axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_above = np.mean(gen_k_raw > bt_max_k, axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_total = gen_raw_exceed_below + gen_raw_exceed_above

    return ClassMetricCache(
        bins=bins,
        real_hist_counts=real_hist_counts,
        gen_raw_hist_counts=gen_raw_hist_counts,
        gen_post_hist_counts=gen_post_hist_counts,
        real_mean_profiles=real_mean_profiles,
        real_azstd_profiles=real_azstd_profiles,
        gen_raw_mean_profiles=gen_raw_mean_profiles,
        gen_raw_azstd_profiles=gen_raw_azstd_profiles,
        gen_post_mean_profiles=gen_post_mean_profiles,
        gen_post_azstd_profiles=gen_post_azstd_profiles,
        real_psd_profiles=real_psd_profiles,
        gen_raw_psd_profiles=gen_raw_psd_profiles,
        gen_post_psd_profiles=gen_post_psd_profiles,
        real_cold=real_cold,
        gen_raw_cold=gen_raw_cold,
        gen_post_cold=gen_post_cold,
        real_dav=real_dav,
        gen_raw_dav=gen_raw_dav,
        gen_post_dav=gen_post_dav,
        real_eye=real_eye,
        gen_raw_eye=gen_raw_eye,
        gen_post_eye=gen_post_eye,
        real_features=real_features,
        gen_raw_features=gen_raw_features,
        gen_post_features=gen_post_features,
        feature_gamma_raw=feature_gamma_raw,
        feature_gamma_post=feature_gamma_post,
        real_pixel_mean=real_k.mean(axis=(1, 2)).astype(np.float32),
        gen_raw_pixel_mean=gen_k_raw.mean(axis=(1, 2)).astype(np.float32),
        gen_post_pixel_mean=gen_k_post.mean(axis=(1, 2)).astype(np.float32),
        gen_raw_exceed_below=gen_raw_exceed_below,
        gen_raw_exceed_above=gen_raw_exceed_above,
        gen_raw_exceed_total=gen_raw_exceed_total,
        pixel_stats_real=summary_stats(real_k.reshape(-1)),
        pixel_stats_gen_raw=summary_stats(gen_k_raw.reshape(-1)),
        pixel_stats_gen_post=summary_stats(gen_k_post.reshape(-1)),
        gen_exceedance_rate={
            "below_bt_min": float(gen_raw_exceed_below.mean()),
            "above_bt_max": float(gen_raw_exceed_above.mean()),
            "total": float(gen_raw_exceed_total.mean()),
            "raw_min": float(np.min(gen_k_raw)),
            "raw_max": float(np.max(gen_k_raw)),
        },
    )


def _compute_per_class_report(cache: ClassMetricCache) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_metrics, raw_curves = _compute_structural_suite(
        bins=cache.bins,
        real_hist_counts=cache.real_hist_counts,
        gen_hist_counts=cache.gen_raw_hist_counts,
        real_mean_profiles=cache.real_mean_profiles,
        gen_mean_profiles=cache.gen_raw_mean_profiles,
        real_azstd_profiles=cache.real_azstd_profiles,
        gen_azstd_profiles=cache.gen_raw_azstd_profiles,
        real_psd_profiles=cache.real_psd_profiles,
        gen_psd_profiles=cache.gen_raw_psd_profiles,
        real_cold=cache.real_cold,
        gen_cold=cache.gen_raw_cold,
        real_dav=cache.real_dav,
        gen_dav=cache.gen_raw_dav,
        real_eye=cache.real_eye,
        gen_eye=cache.gen_raw_eye,
        real_features=cache.real_features,
        gen_features=cache.gen_raw_features,
        feature_gamma=cache.feature_gamma_raw,
    )
    post_metrics, _ = _compute_structural_suite(
        bins=cache.bins,
        real_hist_counts=cache.real_hist_counts,
        gen_hist_counts=cache.gen_post_hist_counts,
        real_mean_profiles=cache.real_mean_profiles,
        gen_mean_profiles=cache.gen_post_mean_profiles,
        real_azstd_profiles=cache.real_azstd_profiles,
        gen_azstd_profiles=cache.gen_post_azstd_profiles,
        real_psd_profiles=cache.real_psd_profiles,
        gen_psd_profiles=cache.gen_post_psd_profiles,
        real_cold=cache.real_cold,
        gen_cold=cache.gen_post_cold,
        real_dav=cache.real_dav,
        gen_dav=cache.gen_post_dav,
        real_eye=cache.real_eye,
        gen_eye=cache.gen_post_eye,
        real_features=cache.real_features,
        gen_features=cache.gen_post_features,
        feature_gamma=cache.feature_gamma_post,
    )

    metrics = {
        "gen_exceedance_rate": cache.gen_exceedance_rate,
        "pixel_stats": {
            "real": cache.pixel_stats_real,
            "gen_raw": cache.pixel_stats_gen_raw,
        },
        **raw_metrics,
        "postprocessed_metrics": {
            "pixel_stats": {
                "real": cache.pixel_stats_real,
                "gen_postprocessed": cache.pixel_stats_gen_post,
            },
            **post_metrics,
        },
    }
    return metrics, raw_curves


def _select_rows(arr: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    if idx is None:
        return arr
    return arr[idx]


def _compute_class_scalar_summary(
    cache: ClassMetricCache,
    *,
    real_idx: np.ndarray | None = None,
    gen_idx: np.ndarray | None = None,
) -> Dict[str, float]:
    real_hist_counts = _select_rows(cache.real_hist_counts, real_idx)
    gen_hist_counts = _select_rows(cache.gen_raw_hist_counts, gen_idx)
    rh = _hist_density_from_counts(real_hist_counts)
    gh = _hist_density_from_counts(gen_hist_counts)

    real_psd = _select_rows(cache.real_psd_profiles, real_idx)
    gen_psd = _select_rows(cache.gen_raw_psd_profiles, gen_idx)
    mmd2, div_real, div_gen = _compute_feature_metrics(
        _select_rows(cache.real_features, real_idx),
        _select_rows(cache.gen_raw_features, gen_idx),
        cache.feature_gamma_raw,
    )

    real_cold = _select_rows(cache.real_cold, real_idx)
    gen_cold = _select_rows(cache.gen_raw_cold, gen_idx)
    real_dav = _select_rows(cache.real_dav, real_idx)
    gen_dav = _select_rows(cache.gen_raw_dav, gen_idx)
    real_eye = _select_rows(cache.real_eye, real_idx)
    gen_eye = _select_rows(cache.gen_raw_eye, gen_idx)

    return {
        "pixel_hist_js": float(js_divergence(rh, gh)),
        "pixel_hist_w1": float(wasserstein1_from_hist(rh, gh, bin_edges=cache.bins)),
        "cold_cloud_fraction_200K_abs_gap": float(abs(real_cold.mean() - gen_cold.mean())),
        "dav_abs_gap_deg2": float(abs(real_dav.mean() - gen_dav.mean())),
        "eye_contrast_proxy_abs_gap": float(abs(real_eye.mean() - gen_eye.mean())),
        "psd_l2": float(((real_psd.mean(axis=0) - gen_psd.mean(axis=0)) ** 2).mean()),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space_abs_gap": float(abs(div_real - div_gen)),
        "gen_exceedance_rate_total": float(_select_rows(cache.gen_raw_exceed_total, gen_idx).mean()),
        "real_pixel_mean": float(_select_rows(cache.real_pixel_mean, real_idx).mean()),
        "gen_pixel_mean": float(_select_rows(cache.gen_raw_pixel_mean, gen_idx).mean()),
        "real_dav_mean_deg2": float(real_dav.mean()),
        "gen_dav_mean_deg2": float(gen_dav.mean()),
        "real_eye_mean": float(real_eye.mean()),
        "gen_eye_mean": float(gen_eye.mean()),
    }


def _default_paper_ready_class_labels(class_ids: list[int]) -> Dict[int, str]:
    class_ids = sorted(int(c) for c in class_ids)
    if class_ids == list(range(6)):
        return {
            0: "< Cat 1",
            1: "Cat 1",
            2: "Cat 2",
            3: "Cat 3",
            4: "Cat 4",
            5: "Cat 5",
        }
    return {c: f"Class {c}" for c in class_ids}


def _flatten_mapping_for_csv(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping_for_csv(value, prefix=child_prefix))
        elif isinstance(value, (list, tuple)):
            flat[child_prefix] = json.dumps(value)
        elif isinstance(value, np.generic):
            flat[child_prefix] = value.item()
        else:
            flat[child_prefix] = value
    return flat


def _write_per_class_metrics_csv(
    path: Path,
    *,
    per_class_metrics: Dict[int, Dict[str, Any]],
    class_labels: Dict[int, str],
    split: str,
    tag: str,
    heavy: bool,
    n_per_class: int,
) -> None:
    rows = []
    for class_id in sorted(int(c) for c in per_class_metrics.keys()):
        metrics = per_class_metrics[class_id]
        row = {
            "class_id": int(class_id),
            "class_label": str(class_labels.get(class_id, f"Class {class_id}")),
            "split": str(split),
            "tag": str(tag),
            "heavy": int(bool(heavy)),
            "n_per_class": int(n_per_class),
        }
        row.update(_flatten_mapping_for_csv(metrics))
        rows.append(row)

    fieldnames = ["class_id", "class_label", "split", "tag", "heavy", "n_per_class"]
    extra_fields = sorted({key for row in rows for key in row.keys() if key not in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_paper_ready_pixel_plausibility_npz(
    path: Path,
    *,
    class_caches: Dict[int, ClassMetricCache],
) -> Dict[str, Any]:
    if not class_caches:
        return {}

    class_ids = sorted(int(c) for c in class_caches.keys())
    class_labels = _default_paper_ready_class_labels(class_ids)

    bins_ref: np.ndarray | None = None
    real_hist_blocks = []
    gen_hist_blocks = []
    real_mass_rows = []
    gen_mass_rows = []
    js_rows = []
    w1_rows = []
    real_hist_rows = []
    gen_hist_rows = []
    real_class_offsets = [0]
    gen_class_offsets = [0]

    for class_id in class_ids:
        cache = class_caches[class_id]
        bins = np.asarray(cache.bins, dtype=np.float64)
        if bins_ref is None:
            bins_ref = bins
        elif bins.shape != bins_ref.shape or not np.allclose(bins, bins_ref):
            raise ValueError("Paper-ready pixel histograms require identical BT bins across classes.")

        real_hist_counts = np.asarray(cache.real_hist_counts, dtype=np.float64)
        gen_hist_counts = np.asarray(cache.gen_raw_hist_counts, dtype=np.float64)
        real_mass = _hist_density_from_counts(real_hist_counts)
        gen_mass = _hist_density_from_counts(gen_hist_counts)

        real_hist_blocks.append(real_hist_counts)
        gen_hist_blocks.append(gen_hist_counts)
        real_mass_rows.append(real_mass)
        gen_mass_rows.append(gen_mass)
        js_rows.append(float(js_divergence(real_mass, gen_mass)))
        w1_rows.append(float(wasserstein1_from_hist(real_mass, gen_mass, bin_edges=bins)))
        real_hist_rows.append(real_hist_counts.astype(np.uint32, copy=False))
        gen_hist_rows.append(gen_hist_counts.astype(np.uint32, copy=False))
        real_class_offsets.append(real_class_offsets[-1] + int(real_hist_counts.shape[0]))
        gen_class_offsets.append(gen_class_offsets[-1] + int(gen_hist_counts.shape[0]))

    assert bins_ref is not None

    overall_real_mass = _hist_density_from_counts(np.concatenate(real_hist_blocks, axis=0))
    overall_gen_mass = _hist_density_from_counts(np.concatenate(gen_hist_blocks, axis=0))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PAPER_READY_PIXEL_ARTIFACT_SCHEMA),
        class_ids=np.asarray(class_ids, dtype=np.int32),
        class_labels=np.asarray([class_labels[c] for c in class_ids]),
        bt_bins=np.asarray(bins_ref, dtype=np.float64),
        real_probability_mass=np.asarray(real_mass_rows, dtype=np.float64),
        gen_probability_mass=np.asarray(gen_mass_rows, dtype=np.float64),
        pixel_hist_js=np.asarray(js_rows, dtype=np.float64),
        pixel_hist_w1_k=np.asarray(w1_rows, dtype=np.float64),
        overall_real_probability_mass=np.asarray(overall_real_mass, dtype=np.float64),
        overall_gen_probability_mass=np.asarray(overall_gen_mass, dtype=np.float64),
        overall_pixel_hist_js=np.asarray([js_divergence(overall_real_mass, overall_gen_mass)], dtype=np.float64),
        overall_pixel_hist_w1_k=np.asarray(
            [wasserstein1_from_hist(overall_real_mass, overall_gen_mass, bin_edges=bins_ref)],
            dtype=np.float64,
        ),
        real_hist_counts_flat=np.concatenate(real_hist_rows, axis=0),
        gen_hist_counts_flat=np.concatenate(gen_hist_rows, axis=0),
        real_class_offsets=np.asarray(real_class_offsets, dtype=np.int32),
        gen_class_offsets=np.asarray(gen_class_offsets, dtype=np.int32),
    )

    return {
        "schema": PAPER_READY_PIXEL_ARTIFACT_SCHEMA,
        "hist_normalization": "probability_mass",
        "metric_variant": "raw",
        "bin_units": "K",
        "metric_units": {"pixel_hist_w1": "K"},
        "class_ids": [int(c) for c in class_ids],
        "class_labels": {str(c): class_labels[c] for c in class_ids},
        "per_image_histograms": {
            "available": True,
            "encoding": "flat_rows_with_class_offsets",
            "fields": {
                "real": {"counts": "real_hist_counts_flat", "offsets": "real_class_offsets"},
                "generated": {"counts": "gen_hist_counts_flat", "offsets": "gen_class_offsets"},
            },
        },
    }


def _write_paper_ready_radial_bt_profile_npz(
    path: Path,
    *,
    class_caches: Dict[int, ClassMetricCache],
) -> Dict[str, Any]:
    if not class_caches:
        return {}

    class_ids = sorted(int(c) for c in class_caches.keys())
    class_labels = _default_paper_ready_class_labels(class_ids)

    radius_ref: np.ndarray | None = None
    real_mean_rows = []
    gen_mean_rows = []
    profile_mae_rows = []
    real_profile_rows = []
    gen_profile_rows = []
    real_class_offsets = [0]
    gen_class_offsets = [0]

    for class_id in class_ids:
        cache = class_caches[class_id]
        real_profiles = np.asarray(cache.real_mean_profiles, dtype=np.float32)
        gen_profiles = np.asarray(cache.gen_raw_mean_profiles, dtype=np.float32)
        if real_profiles.ndim != 2 or gen_profiles.ndim != 2:
            raise ValueError("Paper-ready radial BT artifacts require 2D per-image profile blocks.")
        if real_profiles.shape[1] != gen_profiles.shape[1]:
            raise ValueError("Real and generated radial BT profile widths must match.")

        radius = np.linspace(0.0, 1.0, real_profiles.shape[1], dtype=np.float64)
        if radius_ref is None:
            radius_ref = radius
        elif radius.shape != radius_ref.shape or not np.allclose(radius, radius_ref):
            raise ValueError("Paper-ready radial BT profiles require identical normalized-radius bins across classes.")

        real_mean = np.asarray(real_profiles.mean(axis=0), dtype=np.float64)
        gen_mean = np.asarray(gen_profiles.mean(axis=0), dtype=np.float64)

        real_mean_rows.append(real_mean)
        gen_mean_rows.append(gen_mean)
        profile_mae_rows.append(float(np.mean(np.abs(real_mean - gen_mean))))
        real_profile_rows.append(real_profiles.astype(np.float32, copy=False))
        gen_profile_rows.append(gen_profiles.astype(np.float32, copy=False))
        real_class_offsets.append(real_class_offsets[-1] + int(real_profiles.shape[0]))
        gen_class_offsets.append(gen_class_offsets[-1] + int(gen_profiles.shape[0]))

    assert radius_ref is not None

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PAPER_READY_RADIAL_BT_ARTIFACT_SCHEMA),
        class_ids=np.asarray(class_ids, dtype=np.int32),
        class_labels=np.asarray([class_labels[c] for c in class_ids]),
        radius_normalized=np.asarray(radius_ref, dtype=np.float64),
        real_mean_profile_k=np.asarray(real_mean_rows, dtype=np.float64),
        gen_mean_profile_k=np.asarray(gen_mean_rows, dtype=np.float64),
        radial_profile_mae_k=np.asarray(profile_mae_rows, dtype=np.float64),
        real_mean_profiles_flat=np.concatenate(real_profile_rows, axis=0),
        gen_mean_profiles_flat=np.concatenate(gen_profile_rows, axis=0),
        real_class_offsets=np.asarray(real_class_offsets, dtype=np.int32),
        gen_class_offsets=np.asarray(gen_class_offsets, dtype=np.int32),
    )

    return {
        "schema": PAPER_READY_RADIAL_BT_ARTIFACT_SCHEMA,
        "metric_variant": "raw",
        "x_units": "normalized_radius",
        "y_units": "K",
        "metric_name": "radial_profile_mae_k",
        "metric_units": {"radial_profile_mae_k": "K"},
        "class_ids": [int(c) for c in class_ids],
        "class_labels": {str(c): class_labels[c] for c in class_ids},
        "per_image_profiles": {
            "available": True,
            "encoding": "flat_rows_with_class_offsets",
            "fields": {
                "real": {"profiles": "real_mean_profiles_flat", "offsets": "real_class_offsets"},
                "generated": {"profiles": "gen_mean_profiles_flat", "offsets": "gen_class_offsets"},
            },
        },
    }


def _write_paper_ready_dav_npz(
    path: Path,
    *,
    class_caches: Dict[int, ClassMetricCache],
    radius_km: float,
    pixel_size_km: float,
    center_region_size: int,
) -> Dict[str, Any]:
    if not class_caches:
        return {}

    class_ids = sorted(int(c) for c in class_caches.keys())
    class_labels = _default_paper_ready_class_labels(class_ids)

    real_dav_rows = []
    gen_dav_rows = []
    real_mean_rows = []
    gen_mean_rows = []
    dav_gap_rows = []
    real_class_offsets = [0]
    gen_class_offsets = [0]

    for class_id in class_ids:
        cache = class_caches[class_id]
        real_dav = np.asarray(cache.real_dav, dtype=np.float32).reshape(-1)
        gen_dav = np.asarray(cache.gen_raw_dav, dtype=np.float32).reshape(-1)
        real_dav_rows.append(real_dav)
        gen_dav_rows.append(gen_dav)
        real_mean_rows.append(float(real_dav.mean()))
        gen_mean_rows.append(float(gen_dav.mean()))
        dav_gap_rows.append(float(abs(real_dav.mean() - gen_dav.mean())))
        real_class_offsets.append(real_class_offsets[-1] + int(real_dav.shape[0]))
        gen_class_offsets.append(gen_class_offsets[-1] + int(gen_dav.shape[0]))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PAPER_READY_DAV_ARTIFACT_SCHEMA),
        class_ids=np.asarray(class_ids, dtype=np.int32),
        class_labels=np.asarray([class_labels[c] for c in class_ids]),
        dav_radius_km=np.asarray([float(radius_km)], dtype=np.float64),
        dav_pixel_size_km=np.asarray([float(pixel_size_km)], dtype=np.float64),
        dav_center_region_size=np.asarray([int(center_region_size)], dtype=np.int32),
        real_mean_dav_deg2=np.asarray(real_mean_rows, dtype=np.float64),
        gen_mean_dav_deg2=np.asarray(gen_mean_rows, dtype=np.float64),
        dav_abs_gap_deg2=np.asarray(dav_gap_rows, dtype=np.float64),
        real_dav_flat=np.concatenate(real_dav_rows, axis=0),
        gen_dav_flat=np.concatenate(gen_dav_rows, axis=0),
        real_class_offsets=np.asarray(real_class_offsets, dtype=np.int32),
        gen_class_offsets=np.asarray(gen_class_offsets, dtype=np.int32),
    )

    return {
        "schema": PAPER_READY_DAV_ARTIFACT_SCHEMA,
        "metric_variant": "raw",
        "metric_name": "dav_abs_gap_deg2",
        "metric_units": {"dav_abs_gap_deg2": "deg^2", "dav_mean": "deg^2"},
        "radius_km": float(radius_km),
        "pixel_size_km": float(pixel_size_km),
        "center_region_size": int(center_region_size),
        "class_ids": [int(c) for c in class_ids],
        "class_labels": {str(c): class_labels[c] for c in class_ids},
        "per_image_dav": {
            "available": True,
            "encoding": "flat_rows_with_class_offsets",
            "fields": {
                "real": {"values": "real_dav_flat", "offsets": "real_class_offsets"},
                "generated": {"values": "gen_dav_flat", "offsets": "gen_class_offsets"},
            },
        },
    }


def _write_paper_ready_cold_cloud_fraction_npz(
    path: Path,
    *,
    class_caches: Dict[int, ClassMetricCache],
    threshold_k: float,
) -> Dict[str, Any]:
    if not class_caches:
        return {}

    class_ids = sorted(int(c) for c in class_caches.keys())
    class_labels = _default_paper_ready_class_labels(class_ids)

    real_rows = []
    gen_rows = []
    real_mean_rows = []
    gen_mean_rows = []
    gap_rows = []
    real_class_offsets = [0]
    gen_class_offsets = [0]

    for class_id in class_ids:
        cache = class_caches[class_id]
        real_cold = np.asarray(cache.real_cold, dtype=np.float32).reshape(-1)
        gen_cold = np.asarray(cache.gen_raw_cold, dtype=np.float32).reshape(-1)
        real_rows.append(real_cold)
        gen_rows.append(gen_cold)
        real_mean_rows.append(float(real_cold.mean()))
        gen_mean_rows.append(float(gen_cold.mean()))
        gap_rows.append(float(abs(real_cold.mean() - gen_cold.mean())))
        real_class_offsets.append(real_class_offsets[-1] + int(real_cold.shape[0]))
        gen_class_offsets.append(gen_class_offsets[-1] + int(gen_cold.shape[0]))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PAPER_READY_COLD_CLOUD_ARTIFACT_SCHEMA),
        class_ids=np.asarray(class_ids, dtype=np.int32),
        class_labels=np.asarray([class_labels[c] for c in class_ids]),
        cold_threshold_k=np.asarray([float(threshold_k)], dtype=np.float64),
        real_mean_fraction=np.asarray(real_mean_rows, dtype=np.float64),
        gen_mean_fraction=np.asarray(gen_mean_rows, dtype=np.float64),
        cold_abs_gap_fraction=np.asarray(gap_rows, dtype=np.float64),
        real_fraction_flat=np.concatenate(real_rows, axis=0),
        gen_fraction_flat=np.concatenate(gen_rows, axis=0),
        real_class_offsets=np.asarray(real_class_offsets, dtype=np.int32),
        gen_class_offsets=np.asarray(gen_class_offsets, dtype=np.int32),
    )

    return {
        "schema": PAPER_READY_COLD_CLOUD_ARTIFACT_SCHEMA,
        "metric_variant": "raw",
        "metric_name": "cold_abs_gap_fraction",
        "metric_units": {"cold_abs_gap_fraction": "fraction", "cold_mean": "fraction"},
        "threshold_k": float(threshold_k),
        "class_ids": [int(c) for c in class_ids],
        "class_labels": {str(c): class_labels[c] for c in class_ids},
        "per_image_fraction": {
            "available": True,
            "encoding": "flat_rows_with_class_offsets",
            "fields": {
                "real": {"values": "real_fraction_flat", "offsets": "real_class_offsets"},
                "generated": {"values": "gen_fraction_flat", "offsets": "gen_class_offsets"},
            },
        },
    }


def _compute_aggregate_primary_raw(class_scalars: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    if not class_scalars:
        return {}

    class_ids = sorted(int(c) for c in class_scalars.keys())
    return {
        key: float(np.mean([class_scalars[c][key] for c in class_ids]))
        for key in PRIMARY_RAW_AGG_KEYS
    }


def _compute_macro_metrics(class_scalars: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    if not class_scalars:
        return {}

    class_ids = sorted(int(c) for c in class_scalars.keys())
    classes = np.asarray(class_ids, dtype=np.float64)
    real_mean = np.asarray([class_scalars[c]["real_pixel_mean"] for c in class_ids], dtype=np.float64)
    gen_mean = np.asarray([class_scalars[c]["gen_pixel_mean"] for c in class_ids], dtype=np.float64)
    real_dav = np.asarray([class_scalars[c]["real_dav_mean_deg2"] for c in class_ids], dtype=np.float64)
    gen_dav = np.asarray([class_scalars[c]["gen_dav_mean_deg2"] for c in class_ids], dtype=np.float64)
    real_eye = np.asarray([class_scalars[c]["real_eye_mean"] for c in class_ids], dtype=np.float64)
    gen_eye = np.asarray([class_scalars[c]["gen_eye_mean"] for c in class_ids], dtype=np.float64)

    return {
        "pixel_mean_bt": {
            "corr_class_real": _pearson(classes, real_mean),
            "corr_class_gen": _pearson(classes, gen_mean),
            "corr_real_gen": _pearson(real_mean, gen_mean),
            "mae_gen_vs_real": float(np.mean(np.abs(gen_mean - real_mean))),
        },
        "dav_deg2": {
            "corr_class_real": _pearson(classes, real_dav),
            "corr_class_gen": _pearson(classes, gen_dav),
            "corr_real_gen": _pearson(real_dav, gen_dav),
            "mae_gen_vs_real": float(np.mean(np.abs(gen_dav - real_dav))),
        },
        "eye_contrast_proxy": {
            "corr_class_real": _pearson(classes, real_eye),
            "corr_class_gen": _pearson(classes, gen_eye),
            "corr_real_gen": _pearson(real_eye, gen_eye),
            "mae_gen_vs_real": float(np.mean(np.abs(gen_eye - real_eye))),
        },
    }


def _resolve_auto_bool(
    raw: Any,
    *,
    auto_value: bool,
    field_name: str,
) -> tuple[bool, str]:
    if isinstance(raw, bool):
        return bool(raw), "bool"
    text = str(raw).strip().lower()
    if text == "auto":
        return bool(auto_value), "auto"
    if text in {"1", "true", "yes", "on"}:
        return True, "boolish"
    if text in {"0", "false", "no", "off"}:
        return False, "boolish"
    raise ValueError(
        f"{field_name} must be a boolean or 'auto', got {raw!r}"
    )


def _effective_sample_size_from_weights(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return 0.0
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def _parse_evaluator_fd_negative_control_names(raw: Any) -> list[str]:
    aliases = {
        "gaussian_blur": "gaussian_blur",
        "blur": "gaussian_blur",
        "gaussian": "gaussian_blur",
        "pixel_shuffle": "pixel_shuffle",
        "shuffle": "pixel_shuffle",
        "pixel_shuffled_real": "pixel_shuffle",
    }
    if raw is None:
        items: list[str] = []
    elif isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw if str(part).strip()]
    else:
        raise ValueError(
            "evaluation.evaluator_feature_metric.negative_controls.controls must be "
            f"a string or list of strings, got {type(raw).__name__}."
        )

    out: list[str] = []
    seen = set()
    for item in items:
        key = str(item).strip().lower()
        if key not in aliases:
            raise ValueError(
                "Unsupported evaluator-FD negative control "
                f"{item!r}; expected one of {sorted(EVALUATOR_FD_NEGATIVE_CONTROL_NAMES)}."
            )
        name = aliases[key]
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _sample_real_images_for_negative_control(
    real_k: np.ndarray,
    *,
    draw_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(real_k, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected real_k shaped [N,H,W] for evaluator-FD negative controls, got {arr.shape}."
        )

    draw_n = int(draw_n)
    if draw_n <= 0:
        raise ValueError(f"Negative-control draw_n must be > 0, got {draw_n}.")
    source_n = int(arr.shape[0])
    if source_n <= 0:
        raise ValueError("Cannot build evaluator-FD negative controls from an empty real set.")

    replace = bool(draw_n > source_n)
    if replace:
        idx = rng.integers(source_n, size=draw_n)
        draw_mode = "with_replacement"
    elif draw_n == source_n:
        idx = np.arange(source_n, dtype=np.int64)
        draw_mode = "full_without_replacement"
    else:
        idx = rng.choice(source_n, size=draw_n, replace=False)
        draw_mode = "without_replacement"

    sampled = np.asarray(arr[idx], dtype=np.float32)
    return sampled, {
        "draw_n": int(draw_n),
        "source_n": int(source_n),
        "draw_mode": str(draw_mode),
    }


def _gaussian_kernel_2d(*, sigma_px: float, kernel_size: int) -> np.ndarray:
    sigma_px = float(sigma_px)
    kernel_size = int(kernel_size)
    if sigma_px <= 0.0:
        raise ValueError(f"gaussian_blur_sigma_px must be > 0, got {sigma_px}")
    if kernel_size <= 0 or (kernel_size % 2) != 1:
        raise ValueError(
            "gaussian_blur_kernel_size must be a positive odd integer, "
            f"got {kernel_size}"
        )

    radius = kernel_size // 2
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (grid / sigma_px) ** 2)
    kernel_1d /= np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d


def _apply_gaussian_blur_bt_k(
    bt_k: np.ndarray,
    *,
    sigma_px: float,
    kernel_size: int,
) -> np.ndarray:
    arr = np.asarray(bt_k, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            "Expected BT array shaped [N,H,W] or [H,W] for gaussian blur control, "
            f"got {arr.shape}."
        )

    kernel = _gaussian_kernel_2d(sigma_px=sigma_px, kernel_size=kernel_size)
    kernel_tf = tf.convert_to_tensor(kernel[:, :, None, None], dtype=tf.float32)
    field = tf.convert_to_tensor(arr[..., None], dtype=tf.float32)
    pad = kernel.shape[0] // 2
    field = tf.pad(field, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    blurred = tf.nn.depthwise_conv2d(
        field,
        filter=kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    return np.asarray(blurred.numpy()[..., 0], dtype=np.float32)


def _build_shared_pixel_shuffle_permutation(
    *,
    image_shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape for pixel-shuffle control: {(h, w)}")
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.permutation(h * w), dtype=np.int64)


def _apply_shared_pixel_shuffle_bt_k(
    bt_k: np.ndarray,
    *,
    permutation: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(bt_k, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            "Expected BT array shaped [N,H,W] or [H,W] for pixel-shuffle control, "
            f"got {arr.shape}."
        )

    flat = arr.reshape(arr.shape[0], -1)
    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if flat.shape[1] != perm.shape[0]:
        raise ValueError(
            "Pixel-shuffle permutation length must match flattened image size, "
            f"got {perm.shape[0]} vs {flat.shape[1]}."
        )
    shuffled = flat[:, perm]
    return np.asarray(shuffled.reshape(arr.shape), dtype=np.float32)


def _compute_evaluator_fd(
    *,
    real_embeddings_by_class: Dict[int, np.ndarray],
    gen_embeddings_by_class: Dict[int, np.ndarray],
    covariance_eps: float,
) -> Dict[str, Any]:
    valid_class_ids = sorted(
        int(c) for c in real_embeddings_by_class.keys()
        if c in gen_embeddings_by_class
        and int(real_embeddings_by_class[c].shape[0]) > 0
        and int(gen_embeddings_by_class[c].shape[0]) > 0
    )
    if not valid_class_ids:
        raise ValueError("Cannot compute evaluator FD without overlapping real/generated classes.")

    real_counts = {c: int(real_embeddings_by_class[c].shape[0]) for c in valid_class_ids}
    gen_counts = {c: int(gen_embeddings_by_class[c].shape[0]) for c in valid_class_ids}
    total_real = int(sum(real_counts.values()))
    if total_real <= 0:
        raise ValueError("Cannot compute evaluator FD from an empty real reference set.")

    real_blocks = []
    gen_blocks = []
    gen_weight_blocks = []
    class_priors = {}
    feature_dim: int | None = None

    for c in valid_class_ids:
        real_block = np.asarray(real_embeddings_by_class[c], dtype=np.float64)
        gen_block = np.asarray(gen_embeddings_by_class[c], dtype=np.float64)
        if real_block.ndim != 2 or gen_block.ndim != 2:
            raise ValueError("Evaluator embeddings must be 2D [N,D] blocks.")
        if real_block.shape[1] != gen_block.shape[1]:
            raise ValueError(
                f"Evaluator embedding width mismatch for class {c}: "
                f"{real_block.shape[1]} vs {gen_block.shape[1]}"
            )
        if feature_dim is None:
            feature_dim = int(real_block.shape[1])
        elif int(real_block.shape[1]) != feature_dim:
            raise ValueError("Evaluator embedding width must match across classes.")

        prior = float(real_counts[c]) / float(total_real)
        class_priors[c] = prior
        real_blocks.append(real_block)
        gen_blocks.append(gen_block)
        gen_weight_blocks.append(
            np.full((gen_block.shape[0],), prior / float(gen_block.shape[0]), dtype=np.float64)
        )

    assert feature_dim is not None
    real_all = np.concatenate(real_blocks, axis=0)
    gen_all = np.concatenate(gen_blocks, axis=0)
    real_weights = np.full((real_all.shape[0],), 1.0 / float(real_all.shape[0]), dtype=np.float64)
    gen_weights = np.concatenate(gen_weight_blocks, axis=0)

    mu_real, cov_real = weighted_mean_and_cov(
        real_all,
        real_weights,
        covariance_eps=covariance_eps,
    )
    mu_gen, cov_gen = weighted_mean_and_cov(
        gen_all,
        gen_weights,
        covariance_eps=covariance_eps,
    )
    fd = frechet_distance_from_stats(mu_real, cov_real, mu_gen, cov_gen)

    return {
        "value": float(fd),
        "feature_dim": int(feature_dim),
        "reference_prior": "empirical_real_reference",
        "valid_class_ids": [int(c) for c in valid_class_ids],
        "real_counts_by_class": {str(c): int(real_counts[c]) for c in valid_class_ids},
        "generated_counts_by_class": {str(c): int(gen_counts[c]) for c in valid_class_ids},
        "class_priors": {str(c): float(class_priors[c]) for c in valid_class_ids},
        "num_real_images": int(real_all.shape[0]),
        "num_generated_images": int(gen_all.shape[0]),
        "effective_num_real_images": _effective_sample_size_from_weights(real_weights),
        "effective_num_generated_images": _effective_sample_size_from_weights(gen_weights),
    }


def _summarize_numeric_distribution(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty numeric distribution.")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _compute_evaluator_fd_real_vs_real_reference(
    *,
    real_embeddings_by_class: Dict[int, np.ndarray],
    generated_counts_by_class: Dict[int, int],
    covariance_eps: float,
    reps: int,
    seed: int,
    observed_value: float | None = None,
) -> Dict[str, Any]:
    reps = int(reps)
    if reps <= 0:
        raise ValueError(f"real-vs-real reference reps must be > 0, got {reps}")

    valid_class_ids = sorted(
        int(c) for c in real_embeddings_by_class.keys()
        if int(generated_counts_by_class.get(int(c), 0)) > 0
        and int(real_embeddings_by_class[c].shape[0]) > 0
    )
    if not valid_class_ids:
        raise ValueError("Cannot compute evaluator FD real-vs-real reference without valid classes.")

    real_counts = {c: int(real_embeddings_by_class[c].shape[0]) for c in valid_class_ids}
    total_real = int(sum(real_counts.values()))
    if total_real <= 0:
        raise ValueError("Cannot compute evaluator FD real-vs-real reference from an empty real set.")

    class_priors = {c: float(real_counts[c]) / float(total_real) for c in valid_class_ids}
    real_blocks = [np.asarray(real_embeddings_by_class[c], dtype=np.float64) for c in valid_class_ids]
    real_all = np.concatenate(real_blocks, axis=0)
    real_weights = np.full((real_all.shape[0],), 1.0 / float(real_all.shape[0]), dtype=np.float64)
    mu_real, cov_real = weighted_mean_and_cov(
        real_all,
        real_weights,
        covariance_eps=covariance_eps,
    )

    rng = np.random.default_rng(int(seed))
    null_values = np.zeros((reps,), dtype=np.float64)
    for rep_idx in range(reps):
        pseudo_blocks = []
        pseudo_weight_blocks = []
        for c in valid_class_ids:
            emb = np.asarray(real_embeddings_by_class[c], dtype=np.float64)
            draw_n = int(generated_counts_by_class[c])
            idx = rng.integers(emb.shape[0], size=draw_n)
            pseudo_blocks.append(emb[idx])
            pseudo_weight_blocks.append(
                np.full((draw_n,), class_priors[c] / float(draw_n), dtype=np.float64)
            )

        pseudo_all = np.concatenate(pseudo_blocks, axis=0)
        pseudo_weights = np.concatenate(pseudo_weight_blocks, axis=0)
        mu_pseudo, cov_pseudo = weighted_mean_and_cov(
            pseudo_all,
            pseudo_weights,
            covariance_eps=covariance_eps,
        )
        null_values[rep_idx] = frechet_distance_from_stats(mu_real, cov_real, mu_pseudo, cov_pseudo)

    out: Dict[str, Any] = {
        "enabled": True,
        "variant": "fixed_reference_pseudo_generated_from_real",
        "draw_mode": "with_replacement",
        "reps": int(reps),
        "seed": int(seed),
        "pseudo_generated_counts_by_class": {
            str(c): int(generated_counts_by_class[c]) for c in valid_class_ids
        },
        "summary": _summarize_numeric_distribution(null_values),
        "valid_class_ids": [int(c) for c in valid_class_ids],
    }
    if observed_value is not None:
        observed = float(observed_value)
        ge_count = int(np.sum(null_values >= observed))
        le_count = int(np.sum(null_values <= observed))
        q50 = out["summary"]["q50"]
        q95 = out["summary"]["q95"]
        out.update(
            {
                "observed_value": observed,
                "observed_minus_q50": float(observed - q50),
                "observed_minus_q95": float(observed - q95),
                "observed_percentile_within_null": float(le_count / float(reps)),
                "upper_tail_pvalue": float((ge_count + 1.0) / (reps + 1.0)),
            }
        )
    return out


def _compute_evaluator_fd_negative_controls(
    *,
    real_embeddings_by_class: Dict[int, np.ndarray],
    control_embeddings_by_name: Dict[str, Dict[int, np.ndarray]],
    source_sampling_by_class: Dict[int, Dict[str, Any]],
    covariance_eps: float,
    observed_value: float | None = None,
    negative_controls_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    controls_out: Dict[str, Any] = {}
    gaussian_sigma_px = float(negative_controls_cfg.get("gaussian_blur_sigma_px", 6.0))
    gaussian_kernel_size = int(negative_controls_cfg.get("gaussian_blur_kernel_size", 25))
    pixel_shuffle_seed = int(negative_controls_cfg.get("seed", 123))

    descriptions = {
        "gaussian_blur": (
            "Real test images degraded by a fixed Gaussian blur before evaluator-feature encoding."
        ),
        "pixel_shuffle": (
            "Real test images with a shared random pixel permutation applied per image, "
            "preserving per-image BT histograms while destroying spatial structure."
        ),
    }
    severities = {
        "gaussian_blur": "moderate",
        "pixel_shuffle": "strong",
    }

    for control_name, control_embeddings_by_class in control_embeddings_by_name.items():
        summary = _compute_evaluator_fd(
            real_embeddings_by_class=real_embeddings_by_class,
            gen_embeddings_by_class=control_embeddings_by_class,
            covariance_eps=covariance_eps,
        )
        control_out: Dict[str, Any] = {
            **summary,
            "description": descriptions.get(control_name, ""),
            "severity": severities.get(control_name, "custom"),
            "source": "corrupted_real_test_images",
            "source_sampling_by_class": {
                str(c): dict(meta) for c, meta in sorted(source_sampling_by_class.items())
            },
        }
        if control_name == "gaussian_blur":
            control_out["transformation"] = {
                "name": "gaussian_blur",
                "sigma_px": float(gaussian_sigma_px),
                "kernel_size": int(gaussian_kernel_size),
            }
        elif control_name == "pixel_shuffle":
            control_out["transformation"] = {
                "name": "pixel_shuffle",
                "permutation": "shared_flattened_pixel_permutation",
                "seed": int(pixel_shuffle_seed),
            }
        if observed_value is not None:
            control_value = float(control_out["value"])
            observed = float(observed_value)
            control_out.update(
                {
                    "observed_model_value": observed,
                    "observed_minus_control": float(observed - control_value),
                    "observed_over_control": (
                        float(observed / control_value) if control_value > 0.0 else None
                    ),
                }
            )
        controls_out[control_name] = control_out
    return controls_out


def _bootstrap_ci_blocks(
    *,
    class_caches: Dict[int, ClassMetricCache],
    aggregate_primary_raw: Dict[str, Any],
    macro: Dict[str, Any],
    bootstrap_reps: int,
    bootstrap_ci_level: float,
    seed: int,
    show_progress: bool,
    progress_desc: str = "Eval 3/3: bootstrap CIs",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if bootstrap_reps <= 0 or not class_caches:
        return {}, {}

    rng = np.random.default_rng(seed)
    aggregate_samples: Dict[str, list[float]] = {}
    macro_samples: Dict[str, list[float]] = {}

    rep_iter = range(bootstrap_reps)
    if show_progress:
        rep_iter = tqdm(
            rep_iter,
            total=bootstrap_reps,
            desc=str(progress_desc),
            unit="rep",
            leave=True,
        )

    for _ in rep_iter:
        class_scalars = {}
        for c in sorted(class_caches.keys()):
            cache = class_caches[c]
            real_n = cache.real_pixel_mean.shape[0]
            gen_n = cache.gen_raw_pixel_mean.shape[0]
            real_idx = rng.integers(real_n, size=real_n)
            gen_idx = rng.integers(gen_n, size=gen_n)
            class_scalars[c] = _compute_class_scalar_summary(cache, real_idx=real_idx, gen_idx=gen_idx)

        aggregate_rep = _compute_aggregate_primary_raw(class_scalars)
        macro_rep = _compute_macro_metrics(class_scalars)
        _append_bootstrap_sample(aggregate_samples, aggregate_rep)
        _append_bootstrap_sample(macro_samples, macro_rep)

    aggregate_ci = _build_ci_tree(aggregate_primary_raw, aggregate_samples, bootstrap_ci_level)
    macro_ci = _build_ci_tree(macro, macro_samples, bootstrap_ci_level)
    return aggregate_ci, macro_ci


# -----------------------------------------------------------------------------
# main evaluator
# -----------------------------------------------------------------------------


@dataclass
class TCEvaluator:
    cfg: Dict[str, Any]

    def run(
        self,
        *,
        model: tf.keras.Model | None = None,
        diffusion=None,
        out_dir: Path,
        tag: str,
        split: str = "val",
        full_test: bool = False,
        heavy: bool,
        show_progress: bool = False,
        generated_bank: SampleBank | None = None,
        generated_limit: int | None = None,
        generated_source: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        ev = _default_eval_cfg(cfg)
        sampling_guidance_cfg = sampling_guidance_summary(cfg)
        split = str(split).strip().lower()
        if split not in {"val", "test"}:
            raise ValueError(f"Evaluation split must be 'val' or 'test', got {split!r}")
        full_test = bool(full_test)
        if full_test and split != "test":
            raise ValueError("full_test=True is only supported when split='test'.")
        using_cached_generations = generated_bank is not None

        data_cfg = cfg["data"]
        bt_min_k = float(data_cfg["bt_min_k"])
        bt_max_k = float(data_cfg["bt_max_k"])
        image_size = int(data_cfg.get("image_size", 256))
        real_backend = build_data_backend(data_cfg)

        num_classes = int(cfg["conditioning"]["num_ss_classes"])
        use_wind_speed = bool(cfg.get("conditioning", {}).get("use_wind_speed", False))

        generated_source = generated_source if isinstance(generated_source, dict) else {}
        generated_source_info = dict(generated_source)
        sample_bank_info = generated_source_info.get("sample_bank")
        if using_cached_generations:
            assert generated_bank is not None
            missing_classes = [c for c in range(num_classes) if c not in set(generated_bank.class_ids)]
            if missing_classes:
                raise ValueError(
                    "Cached generated samples are missing required classes: "
                    f"{missing_classes}. Expected classes 0..{num_classes - 1}."
                )
            cached_counts = {
                c: int(generated_bank.generated_counts_by_class[c])
                for c in range(num_classes)
            }
            if generated_limit is None:
                n_per = int(min(cached_counts.values()))
            else:
                n_per = int(generated_limit)
                insufficient = {c: available for c, available in cached_counts.items() if available < n_per}
                if insufficient:
                    raise ValueError(
                        "Cached generated bank cannot satisfy the requested generated_limit. "
                        f"Available counts: {insufficient}"
                    )
        else:
            if model is None or diffusion is None:
                raise ValueError("Online evaluation requires both model and diffusion instances.")
            n_per = int(ev["n_per_class_heavy"] if heavy else ev["n_per_class_light"])
            if heavy and tag == "post_training" and n_per < 100:
                print(f"[eval] post_training heavy eval requested {n_per} samples/class; raising to 100.")
                n_per = 100
        if n_per <= 0:
            raise ValueError(f"evaluation n_per_class must be > 0, got {n_per}")

        n_plot = int(ev["n_plot_per_group"])
        if n_plot <= 0:
            raise ValueError(f"evaluation n_plot_per_group must be > 0, got {n_plot}")
        if n_per < n_plot:
            mode_name = "heavy" if heavy else "light"
            print(
                f"[eval] Requested n_plot_per_group={n_plot}, but {mode_name} evaluation only uses "
                f"n_per_class={n_per}; plotting {n_per} per group instead."
            )

        if using_cached_generations:
            generation_meta = sample_bank_info.get("generation", {}) if isinstance(sample_bank_info, dict) else {}
            gen_batch_size_cfg = generation_meta.get("gen_batch_size")
            if gen_batch_size_cfg is None:
                gen_batch_size_cfg = ev.get("gen_batch_size")
        else:
            gen_batch_size_cfg = ev.get("gen_batch_size")
        if gen_batch_size_cfg is None:
            gen_batch_size = int(data_cfg.get("batch_size", n_per))
        else:
            gen_batch_size = int(gen_batch_size_cfg)
        if gen_batch_size <= 0:
            raise ValueError(f"evaluation.gen_batch_size must be > 0, got {gen_batch_size}")
        gen_batch_size = min(gen_batch_size, n_per)
        batches_per_class = (n_per + gen_batch_size - 1) // gen_batch_size

        bootstrap_reps_key = "bootstrap_reps_full_test" if full_test else "bootstrap_reps"
        bootstrap_reps = int(ev.get(bootstrap_reps_key, 0 if full_test else 500))
        if bootstrap_reps < 0:
            raise ValueError(f"evaluation.{bootstrap_reps_key} must be >= 0, got {bootstrap_reps}")
        bootstrap_ci_level = float(ev.get("bootstrap_ci_level", 0.95))
        if not (0.0 < bootstrap_ci_level < 1.0):
            raise ValueError(
                f"evaluation.bootstrap_ci_level must lie strictly between 0 and 1, got {bootstrap_ci_level}"
            )
        dav_radius_km = float(ev.get("dav_radius_km", 300.0))
        dav_pixel_size_km = float(ev.get("dav_pixel_size_km", 8.0))
        dav_center_region_size = int(ev.get("dav_center_region_size", 3))
        if dav_radius_km <= 0.0:
            raise ValueError(f"evaluation.dav_radius_km must be > 0, got {dav_radius_km}")
        if dav_pixel_size_km <= 0.0:
            raise ValueError(f"evaluation.dav_pixel_size_km must be > 0, got {dav_pixel_size_km}")
        if dav_center_region_size <= 0 or (dav_center_region_size % 2) != 1:
            raise ValueError(
                "evaluation.dav_center_region_size must be a positive odd integer, "
                f"got {dav_center_region_size}"
            )

        evaluator_fd_cfg = dict(ev.get("evaluator_feature_metric", {}))
        evaluator_fd_enabled, evaluator_fd_enable_mode = _resolve_auto_bool(
            evaluator_fd_cfg.get("enabled", "auto"),
            auto_value=bool(split == "test" and full_test and using_cached_generations),
            field_name="evaluation.evaluator_feature_metric.enabled",
        )
        evaluator_fd_covariance_eps = float(evaluator_fd_cfg.get("covariance_eps", 1.0e-6))
        if evaluator_fd_covariance_eps < 0.0:
            raise ValueError(
                "evaluation.evaluator_feature_metric.covariance_eps must be >= 0, "
                f"got {evaluator_fd_covariance_eps}"
            )
        evaluator_fd_null_reps = int(evaluator_fd_cfg.get("null_reference_reps", 0))
        if evaluator_fd_null_reps < 0:
            raise ValueError(
                "evaluation.evaluator_feature_metric.null_reference_reps must be >= 0, "
                f"got {evaluator_fd_null_reps}"
            )
        evaluator_fd_null_seed = int(evaluator_fd_cfg.get("null_reference_seed", ev["seed"]))
        negative_controls_cfg = dict(evaluator_fd_cfg.get("negative_controls", {}))
        negative_controls_enabled, negative_controls_enable_mode = _resolve_auto_bool(
            negative_controls_cfg.get("enabled", "auto"),
            auto_value=bool(evaluator_fd_enabled),
            field_name="evaluation.evaluator_feature_metric.negative_controls.enabled",
        )
        negative_control_names = _parse_evaluator_fd_negative_control_names(
            negative_controls_cfg.get("controls", list(EVALUATOR_FD_NEGATIVE_CONTROL_NAMES))
        )
        negative_controls_seed = int(negative_controls_cfg.get("seed", ev["seed"]))
        negative_controls_match_generated_counts = bool(
            negative_controls_cfg.get("match_generated_counts", True)
        )
        negative_controls_blur_sigma = float(
            negative_controls_cfg.get("gaussian_blur_sigma_px", 6.0)
        )
        negative_controls_blur_kernel_size = int(
            negative_controls_cfg.get("gaussian_blur_kernel_size", 25)
        )
        if negative_controls_blur_sigma <= 0.0:
            raise ValueError(
                "evaluation.evaluator_feature_metric.negative_controls.gaussian_blur_sigma_px "
                f"must be > 0, got {negative_controls_blur_sigma}"
            )
        if negative_controls_blur_kernel_size <= 0 or (negative_controls_blur_kernel_size % 2) != 1:
            raise ValueError(
                "evaluation.evaluator_feature_metric.negative_controls.gaussian_blur_kernel_size "
                f"must be a positive odd integer, got {negative_controls_blur_kernel_size}"
            )

        tf.random.set_seed(ev["seed"])
        np.random.seed(ev["seed"])

        eval_root = resolve_eval_root(out_dir, tag, split)
        eval_root.mkdir(parents=True, exist_ok=True)

        evaluator_feature_metric_report: Dict[str, Any] = {
            "enabled": False,
            "enable_mode": str(evaluator_fd_enable_mode),
            "configured_enabled": evaluator_fd_cfg.get("enabled", "auto"),
            "metric_name": "evaluator_fd",
            "reference_prior": "empirical_real_reference",
            "covariance_eps": float(evaluator_fd_covariance_eps),
            "null_reference_reps": int(evaluator_fd_null_reps),
            "null_reference_seed": int(evaluator_fd_null_seed),
            "negative_controls": {
                "enabled": False,
                "enable_mode": str(negative_controls_enable_mode),
                "configured_enabled": negative_controls_cfg.get("enabled", "auto"),
                "controls_requested": list(negative_control_names),
                "match_generated_counts": bool(negative_controls_match_generated_counts),
                "seed": int(negative_controls_seed),
            },
        }
        feature_extractor: EvaluatorEmbeddingExtractor | None = None
        if evaluator_fd_enabled:
            try:
                feature_extractor = EvaluatorEmbeddingExtractor.from_run(
                    run_name=str(evaluator_fd_cfg.get("run_name", "evaluator_cat4plus_mild_rebalanced_s035")),
                    config_path=evaluator_fd_cfg.get("config_path"),
                    weights_path=evaluator_fd_cfg.get("weights_path"),
                    weights_name=evaluator_fd_cfg.get("weights_name", "best_tail"),
                    batch_size=int(evaluator_fd_cfg.get("batch_size", 32)),
                    embedding_layer=str(evaluator_fd_cfg.get("embedding_layer", "embedding_silu")),
                )
            except Exception as exc:
                if evaluator_fd_enable_mode == "auto":
                    evaluator_feature_metric_report["reason"] = (
                        "auto-disabled because the evaluator feature extractor could not be loaded: "
                        f"{exc}"
                    )
                    feature_extractor = None
                else:
                    raise

        if feature_extractor is not None:
            evaluator_feature_metric_report["enabled"] = True
            evaluator_feature_metric_report["extractor"] = feature_extractor.metadata()
            evaluator_feature_metric_report["input_preprocessing"] = (
                "clip to evaluator BT range, normalize to [-1, 1]"
            )
        elif "reason" not in evaluator_feature_metric_report:
            evaluator_feature_metric_report["reason"] = (
                (
                    "auto-disabled; default policy only enables this metric for cached-bank "
                    "full-test evals"
                    if evaluator_fd_enable_mode == "auto" and not evaluator_fd_enabled
                    else "disabled by configuration"
                )
                if not evaluator_fd_enabled
                else "evaluator feature extractor unavailable"
            )
        negative_controls_report = evaluator_feature_metric_report["negative_controls"]
        if feature_extractor is not None and negative_controls_enabled and negative_control_names:
            negative_controls_report["enabled"] = True
            negative_controls_report["source"] = "corrupted_real_test_images"
        elif not negative_controls_enabled:
            negative_controls_report["reason"] = (
                (
                    "auto-disabled because evaluator_feature_metric is not enabled"
                    if negative_controls_enable_mode == "auto"
                    else "disabled by configuration"
                )
            )
        elif feature_extractor is None:
            negative_controls_report["reason"] = "evaluator feature extractor unavailable"
        else:
            negative_controls_report["reason"] = "no negative controls requested"

        real_paths_by_class, real_wind_by_class = _select_real_by_class(
            cfg,
            n_per,
            ev["real_seed"],
            split,
            use_all=full_test,
        )
        candidate_generated_class_ids = (
            set(generated_bank.class_ids) if using_cached_generations and generated_bank is not None else set(range(num_classes))
        )
        valid_class_ids = [
            c for c in range(num_classes)
            if c in real_paths_by_class and c in candidate_generated_class_ids
        ]
        negative_controls_active = bool(
            feature_extractor is not None and negative_controls_enabled and negative_control_names
        )
        summary_tasks_per_class = 2 + int(feature_extractor is not None) + int(negative_controls_active)
        summary_total_tasks = summary_tasks_per_class * len(valid_class_ids)
        stage_names = []
        if not using_cached_generations:
            stage_names.append("sampling")
        stage_names.append("summaries")
        if bootstrap_reps > 0:
            stage_names.append("bootstrap")
        stage_lookup = {name: (idx + 1, len(stage_names)) for idx, name in enumerate(stage_names)}

        wind_targets_kt: Dict[int, float] = {}
        sampling_report = {}
        if using_cached_generations:
            if isinstance(sample_bank_info, dict):
                generation_meta = sample_bank_info.get("generation", {})
                if isinstance(generation_meta, dict):
                    sampling_report = {
                        "sampler": str(generation_meta.get("sampler", ev["sampler"])),
                        "guidance_scale": float(generation_meta.get("guidance_scale", ev["guidance_scale"])),
                        "sampling_steps": generation_meta.get("sampling_steps", ev.get("sampling_steps", ev.get("ddim_steps", None))),
                        "timestep_schedule": str(generation_meta.get("timestep_schedule", ev["timestep_schedule"])),
                        "ddim_eta": float(generation_meta.get("ddim_eta", ev.get("ddim_eta", 0.0))),
                        "sampling_guidance": dict(
                            generation_meta.get("sampling_guidance", sampling_guidance_cfg)
                        ),
                    }
                targets = sample_bank_info.get("conditioning_targets", {}).get("wind_kt_by_class", {})
                if isinstance(targets, dict):
                    wind_targets_kt = {int(k): float(v) for k, v in targets.items()}
            if not sampling_report:
                sampling_report = {
                    "sampler": str(ev["sampler"]),
                    "guidance_scale": float(ev["guidance_scale"]),
                    "sampling_steps": ev.get("sampling_steps", ev.get("ddim_steps", None)),
                    "timestep_schedule": str(ev["timestep_schedule"]),
                    "ddim_eta": float(ev.get("ddim_eta", 0.0)),
                    "sampling_guidance": dict(sampling_guidance_cfg),
                }
            if show_progress:
                real_mode_desc = (
                    "all available real test samples/class"
                    if full_test
                    else f"up to {n_per} sampled real references/class"
                )
                summary_stage_idx, stage_total = stage_lookup["summaries"]
                print(f"[eval] Reference mode: {real_mode_desc}. Generated source: cached bank ({n_per}/class).")
                stage_plan_parts = [f"stage {summary_stage_idx}/{stage_total} summaries={summary_total_tasks} class tasks"]
                if "bootstrap" in stage_lookup:
                    bootstrap_stage_idx, stage_total = stage_lookup["bootstrap"]
                    stage_plan_parts.append(f"stage {bootstrap_stage_idx}/{stage_total} bootstrap={bootstrap_reps} reps")
                print("[eval] Progress plan: " + "; ".join(stage_plan_parts) + ".")
        else:
            sampling_steps = ev.get("sampling_steps", ev.get("ddim_steps", None))
            sampling_timesteps = diffusion.get_sampling_timesteps(
                sampler=str(ev["sampler"]),
                num_sampling_steps=sampling_steps,
                timestep_schedule=str(ev["timestep_schedule"]),
            )
            sampling_report = {
                "sampler": str(ev["sampler"]),
                "guidance_scale": float(ev["guidance_scale"]),
                "sampling_steps": ev.get("sampling_steps", ev.get("ddim_steps", None)),
                "timestep_schedule": str(ev["timestep_schedule"]),
                "ddim_eta": float(ev.get("ddim_eta", 0.0)),
                "sampling_guidance": (
                    diffusion.get_sampling_guidance_report()
                    if hasattr(diffusion, "get_sampling_guidance_report")
                    else dict(sampling_guidance_cfg)
                ),
            }

            gen_raw_by_class: Dict[int, np.ndarray] = {}
            sampling_pbar = None
            if show_progress:
                real_mode_desc = (
                    "all available real test samples/class"
                    if full_test
                    else f"up to {n_per} sampled real references/class"
                )
                sampling_stage_idx, stage_total = stage_lookup["sampling"]
                summary_stage_idx, _ = stage_lookup["summaries"]
                print(f"[eval] Reference mode: {real_mode_desc}. Generated source: online sampling ({n_per}/class).")
                stage_plan_parts = [
                    f"stage {sampling_stage_idx}/{stage_total} sampling={num_classes} classes x {batches_per_class} "
                    f"batches/class x {len(sampling_timesteps)} reverse steps = "
                    f"{num_classes * batches_per_class * len(sampling_timesteps)} updates",
                    f"stage {summary_stage_idx}/{stage_total} summaries={summary_total_tasks} class tasks",
                ]
                if "bootstrap" in stage_lookup:
                    bootstrap_stage_idx, _ = stage_lookup["bootstrap"]
                    stage_plan_parts.append(f"stage {bootstrap_stage_idx}/{stage_total} bootstrap={bootstrap_reps} reps")
                print("[eval] Progress plan: " + "; ".join(stage_plan_parts) + ".")
                sampling_pbar = tqdm(
                    total=num_classes * batches_per_class * len(sampling_timesteps),
                    desc=f"Eval {sampling_stage_idx}/{stage_total}: sampling",
                    unit="step",
                    leave=True,
                )

            for c in range(num_classes):
                if use_wind_speed:
                    real_winds_c = real_wind_by_class.get(c)
                    if real_winds_c is not None and len(real_winds_c) > 0:
                        if len(real_winds_c) >= n_per:
                            wind_schedule = real_winds_c[:n_per]
                        else:
                            rng_w = np.random.default_rng(ev["seed"] + c)
                            idx = rng_w.choice(len(real_winds_c), size=n_per, replace=True)
                            wind_schedule = real_winds_c[idx]
                        wind_targets_kt[int(c)] = float(wind_schedule.mean())
                    else:
                        wind_schedule = np.full(n_per, _resolve_eval_wind_target_kt(cfg, c), dtype=np.float32)
                        wind_targets_kt[int(c)] = float(wind_schedule[0])
                else:
                    wind_schedule = None

                raw_chunks = []
                offset = 0
                remaining = n_per
                batch_idx = 0
                while remaining > 0:
                    bsz = min(gen_batch_size, remaining)
                    wind_batch = wind_schedule[offset:offset + bsz] if wind_schedule is not None else None
                    if sampling_pbar is not None:
                        sampling_pbar.set_postfix_str(
                            f"class {c} batch {batch_idx + 1}/{batches_per_class}",
                            refresh=False,
                        )
                    sample_outputs = diffusion.sample(
                        model,
                        batch_size=bsz,
                        image_size=image_size,
                        cond_value=c,
                        wind_value_kt=wind_batch,
                        guidance_scale=float(ev["guidance_scale"]),
                        sampler=str(ev["sampler"]),
                        num_sampling_steps=sampling_steps,
                        timestep_schedule=str(ev["timestep_schedule"]),
                        ddim_eta=float(ev.get("ddim_eta", 0.0)),
                        show_progress=bool(show_progress and sampling_pbar is None),
                        return_both=False,
                        progress_callback=sampling_pbar.update if sampling_pbar is not None else None,
                    )
                    raw_chunks.append(sample_outputs.numpy())
                    remaining -= bsz
                    offset += bsz
                    batch_idx += 1

                raw_norm = np.concatenate(raw_chunks, axis=0)
                gen_raw_by_class[c] = denorm_bt(raw_norm, bt_min_k, bt_max_k)[..., 0]

            if sampling_pbar is not None:
                sampling_pbar.close()

        real_counts_by_class = {str(c): int(len(paths)) for c, paths in sorted(real_paths_by_class.items())}
        if using_cached_generations:
            gen_counts_by_class = {str(c): int(n_per) for c in sorted(candidate_generated_class_ids)}
        else:
            gen_counts_by_class = {str(c): int(arr.shape[0]) for c, arr in sorted(gen_raw_by_class.items())}

        binner = PolarBinner(image_size, image_size, int(ev["profile_bins"]), 360)
        dav_computer = DAVComputer(
            image_size,
            image_size,
            pixel_size_km=dav_pixel_size_km,
            radius_km=dav_radius_km,
            center_region_size=dav_center_region_size,
        )

        class_caches: Dict[int, ClassMetricCache] = {}
        per_class_metrics = {}
        per_class_curves = {}
        class_scalars = {}
        real_evaluator_embeddings: Dict[int, np.ndarray] = {}
        gen_evaluator_embeddings: Dict[int, np.ndarray] = {}
        negative_control_embeddings: Dict[str, Dict[int, np.ndarray]] = {
            name: {} for name in negative_control_names
        } if negative_controls_active else {}
        negative_control_source_sampling_by_class: Dict[int, Dict[str, Any]] = {}
        pixel_shuffle_permutation: np.ndarray | None = None
        summary_pbar = None
        if show_progress and valid_class_ids:
            summary_stage_idx, stage_total = stage_lookup["summaries"]
            print(f"[eval] Stage {summary_stage_idx}/{stage_total}: computing class summaries and plots.")
            summary_pbar = tqdm(
                total=summary_total_tasks,
                desc=f"Eval {summary_stage_idx}/{stage_total}: class summaries",
                unit="task",
                leave=True,
            )

        for c in range(num_classes):
            real_paths_c = real_paths_by_class.get(c)
            if using_cached_generations:
                if generated_bank is None or c not in candidate_generated_class_ids:
                    gen_k_raw = None
                else:
                    gen_k_raw = generated_bank.load_bt_k(c, limit=n_per, mmap_mode="r")
            else:
                gen_k_raw = gen_raw_by_class.pop(c, None)
            if not real_paths_c or gen_k_raw is None:
                continue
            gen_k_raw = np.asarray(gen_k_raw, dtype=np.float32)
            real_k = _load_real_group_from_paths(
                real_backend,
                real_paths_c,
                bt_min_k=bt_min_k,
                bt_max_k=bt_max_k,
            )

            n_show = min(n_plot, real_k.shape[0], gen_k_raw.shape[0])
            if n_show > 0:
                save_real_generated_comparison_grid(
                    real_k=real_k,
                    gen_k=gen_k_raw,
                    path=str(eval_root / f"samples_class_{c}.png"),
                    n_show=n_show,
                    ncols=min(5, n_show),
                    real_title=f"Real (n={n_show})",
                    gen_title=f"Generated (n={n_show})",
                    suptitle=f"Class {c}: Real vs Generated Samples",
                )
            cache = _build_class_metric_cache(
                real_k=real_k,
                gen_k_raw=gen_k_raw,
                binner=binner,
                dav_computer=dav_computer,
                psd_bins=int(ev["psd_bins"]),
                bt_min_k=bt_min_k,
                bt_max_k=bt_max_k,
            )
            if feature_extractor is not None:
                real_evaluator_embeddings[c] = feature_extractor.encode_bt_k(real_k)
                gen_evaluator_embeddings[c] = feature_extractor.encode_bt_k(gen_k_raw)
                if summary_pbar is not None:
                    summary_pbar.set_postfix_str(f"features class {c}", refresh=False)
                    summary_pbar.update(1)
                if negative_controls_active:
                    draw_n = (
                        int(gen_k_raw.shape[0])
                        if negative_controls_match_generated_counts
                        else int(real_k.shape[0])
                    )
                    class_rng = np.random.default_rng(
                        int(negative_controls_seed + 7919 * (c + 1))
                    )
                    control_source_k, source_meta = _sample_real_images_for_negative_control(
                        real_k,
                        draw_n=draw_n,
                        rng=class_rng,
                    )
                    negative_control_source_sampling_by_class[c] = source_meta
                    if "gaussian_blur" in negative_control_embeddings:
                        blur_k = _apply_gaussian_blur_bt_k(
                            control_source_k,
                            sigma_px=negative_controls_blur_sigma,
                            kernel_size=negative_controls_blur_kernel_size,
                        )
                        negative_control_embeddings["gaussian_blur"][c] = feature_extractor.encode_bt_k(blur_k)
                    if "pixel_shuffle" in negative_control_embeddings:
                        if pixel_shuffle_permutation is None:
                            pixel_shuffle_permutation = _build_shared_pixel_shuffle_permutation(
                                image_shape=(int(control_source_k.shape[1]), int(control_source_k.shape[2])),
                                seed=int(negative_controls_seed),
                            )
                        shuffled_k = _apply_shared_pixel_shuffle_bt_k(
                            control_source_k,
                            permutation=pixel_shuffle_permutation,
                        )
                        negative_control_embeddings["pixel_shuffle"][c] = feature_extractor.encode_bt_k(shuffled_k)
                    if summary_pbar is not None:
                        summary_pbar.set_postfix_str(f"controls class {c}", refresh=False)
                        summary_pbar.update(1)
            del real_k
            del gen_k_raw
            class_caches[c] = cache
            metrics, curves = _compute_per_class_report(cache)
            per_class_metrics[c] = metrics
            per_class_curves[c] = curves
            class_scalars[c] = _compute_class_scalar_summary(cache)
            if summary_pbar is not None:
                summary_pbar.set_postfix_str(f"metrics class {c}", refresh=False)
                summary_pbar.update(1)

        plots_root = eval_root / "plots"
        for c, cur in per_class_curves.items():
            class_plot_dir = plots_root / f"class_{c}"

            bins = np.asarray(cur["bt_bins"], dtype=np.float32)
            hist_real = np.asarray(cur["hist_real"], dtype=np.float32)
            hist_gen = np.asarray(cur["hist_gen"], dtype=np.float32)
            plot_hist_overlay(
                out_path=class_plot_dir / "pixel_hist_bt.png",
                bins=bins,
                real_hist=hist_real,
                gen_hist=hist_gen,
                title=f"Class {c}: Pixel BT Histogram",
                xlabel="Brightness temperature [K]",
            )

            r_mean_real_mu = np.asarray(cur["radial_mean_real_mu"], dtype=np.float32)
            r_mean_real_sd = np.asarray(cur["radial_mean_real_sd"], dtype=np.float32)
            r_mean_gen_mu = np.asarray(cur["radial_mean_gen_mu"], dtype=np.float32)
            r_mean_gen_sd = np.asarray(cur["radial_mean_gen_sd"], dtype=np.float32)
            r = np.linspace(0.0, 1.0, num=r_mean_real_mu.shape[0], dtype=np.float32)
            plot_radial_profiles(
                out_path=class_plot_dir / "radial_mean.png",
                r=r,
                real_mean=r_mean_real_mu,
                real_std=r_mean_real_sd,
                gen_mean=r_mean_gen_mu,
                gen_std=r_mean_gen_sd,
                title=f"Class {c}: Radial Mean Profile",
                ylabel="BT [K]",
            )

            r_az_real_mu = np.asarray(cur["radial_azstd_real_mu"], dtype=np.float32)
            r_az_real_sd = np.asarray(cur["radial_azstd_real_sd"], dtype=np.float32)
            r_az_gen_mu = np.asarray(cur["radial_azstd_gen_mu"], dtype=np.float32)
            r_az_gen_sd = np.asarray(cur["radial_azstd_gen_sd"], dtype=np.float32)
            r_az = np.linspace(0.0, 1.0, num=r_az_real_mu.shape[0], dtype=np.float32)
            plot_radial_profiles(
                out_path=class_plot_dir / "radial_azstd.png",
                r=r_az,
                real_mean=r_az_real_mu,
                real_std=r_az_real_sd,
                gen_mean=r_az_gen_mu,
                gen_std=r_az_gen_sd,
                title=f"Class {c}: Radial Azimuthal Std",
                ylabel="Azimuthal std(BT) [K]",
            )

            psd_real_mu = np.asarray(cur["psd_real_mu"], dtype=np.float32)
            psd_real_sd = np.asarray(cur["psd_real_sd"], dtype=np.float32)
            psd_gen_mu = np.asarray(cur["psd_gen_mu"], dtype=np.float32)
            psd_gen_sd = np.asarray(cur["psd_gen_sd"], dtype=np.float32)
            k = np.linspace(0.0, 1.0, num=psd_real_mu.shape[0], dtype=np.float32)
            plot_psd(
                out_path=class_plot_dir / "psd_radial.png",
                k=k,
                real_mean=psd_real_mu,
                real_std=psd_real_sd,
                gen_mean=psd_gen_mu,
                gen_std=psd_gen_sd,
                title=f"Class {c}: Radial PSD",
            )
            if summary_pbar is not None:
                summary_pbar.set_postfix_str(f"plots class {c}", refresh=False)
                summary_pbar.update(1)

        if summary_pbar is not None:
            summary_pbar.close()

        evaluator_fd_summary: Dict[str, Any] | None = None
        if feature_extractor is not None and real_evaluator_embeddings and gen_evaluator_embeddings:
            evaluator_fd_summary = _compute_evaluator_fd(
                real_embeddings_by_class=real_evaluator_embeddings,
                gen_embeddings_by_class=gen_evaluator_embeddings,
                covariance_eps=evaluator_fd_covariance_eps,
            )
            evaluator_feature_metric_report.update(evaluator_fd_summary)
            if evaluator_fd_null_reps > 0:
                evaluator_feature_metric_report["real_vs_real_reference"] = _compute_evaluator_fd_real_vs_real_reference(
                    real_embeddings_by_class=real_evaluator_embeddings,
                    generated_counts_by_class={
                        int(c): int(arr.shape[0]) for c, arr in gen_evaluator_embeddings.items()
                    },
                    covariance_eps=evaluator_fd_covariance_eps,
                    reps=evaluator_fd_null_reps,
                    seed=evaluator_fd_null_seed,
                    observed_value=float(evaluator_fd_summary["value"]),
                )
            if negative_controls_active and negative_control_embeddings:
                negative_controls_report["controls"] = _compute_evaluator_fd_negative_controls(
                    real_embeddings_by_class=real_evaluator_embeddings,
                    control_embeddings_by_name=negative_control_embeddings,
                    source_sampling_by_class=negative_control_source_sampling_by_class,
                    covariance_eps=evaluator_fd_covariance_eps,
                    observed_value=float(evaluator_fd_summary["value"]),
                    negative_controls_cfg=negative_controls_cfg,
                )
        elif feature_extractor is not None:
            evaluator_feature_metric_report["reason"] = (
                "enabled, but no overlapping real/generated embeddings were available"
            )

        aggregate_primary_raw = _compute_aggregate_primary_raw(class_scalars)
        if evaluator_fd_summary is not None:
            aggregate_primary_raw["evaluator_fd"] = float(evaluator_fd_summary["value"])
        macro = _compute_macro_metrics(class_scalars)
        if show_progress and bootstrap_reps > 0:
            bootstrap_stage_idx, stage_total = stage_lookup["bootstrap"]
            print(f"[eval] Stage {bootstrap_stage_idx}/{stage_total}: bootstrap confidence intervals.")
        aggregate_primary_raw_ci, macro_ci = _bootstrap_ci_blocks(
            class_caches=class_caches,
            aggregate_primary_raw=aggregate_primary_raw,
            macro=macro,
            bootstrap_reps=bootstrap_reps,
            bootstrap_ci_level=bootstrap_ci_level,
            seed=int(ev["seed"]),
            show_progress=show_progress,
            progress_desc=(
                f"Eval {stage_lookup['bootstrap'][0]}/{stage_lookup['bootstrap'][1]}: bootstrap CIs"
                if "bootstrap" in stage_lookup
                else "Eval bootstrap CIs"
            ),
        )

        class_ids_for_report = sorted(int(c) for c in class_caches.keys())
        class_labels = _default_paper_ready_class_labels(class_ids_for_report) if class_ids_for_report else {}

        artifacts_root = eval_root / "artifacts"
        artifacts_manifest: Dict[str, Any] = {"manifest_version": 1}
        if per_class_metrics:
            per_class_csv_path = artifacts_root / "per_class_metrics.csv"
            _write_per_class_metrics_csv(
                per_class_csv_path,
                per_class_metrics=per_class_metrics,
                class_labels=class_labels,
                split=split,
                tag=tag,
                heavy=heavy,
                n_per_class=n_per,
            )
            artifacts_manifest["per_class_metrics_csv"] = {
                "storage": "sidecar_csv",
                "path": per_class_csv_path.relative_to(eval_root).as_posix(),
                "schema": PER_CLASS_METRICS_CSV_SCHEMA,
                "row_type": "per_class",
            }

        paper_ready_manifest: Dict[str, Any] = {}
        if class_caches:
            pixel_artifact_path = artifacts_root / "paper_ready" / "pixel_plausibility.npz"
            pixel_manifest = _write_paper_ready_pixel_plausibility_npz(
                pixel_artifact_path,
                class_caches=class_caches,
            )
            if pixel_manifest:
                paper_ready_manifest["pixel_plausibility"] = {
                    "storage": "sidecar_npz",
                    "path": pixel_artifact_path.relative_to(eval_root).as_posix(),
                    **pixel_manifest,
                }
            radial_artifact_path = artifacts_root / "paper_ready" / "radial_bt_profile.npz"
            radial_manifest = _write_paper_ready_radial_bt_profile_npz(
                radial_artifact_path,
                class_caches=class_caches,
            )
            if radial_manifest:
                paper_ready_manifest["radial_bt_profile"] = {
                    "storage": "sidecar_npz",
                    "path": radial_artifact_path.relative_to(eval_root).as_posix(),
                    **radial_manifest,
                }
            dav_artifact_path = artifacts_root / "paper_ready" / "dav.npz"
            dav_manifest = _write_paper_ready_dav_npz(
                dav_artifact_path,
                class_caches=class_caches,
                radius_km=dav_radius_km,
                pixel_size_km=dav_pixel_size_km,
                center_region_size=dav_center_region_size,
            )
            if dav_manifest:
                paper_ready_manifest["dav"] = {
                    "storage": "sidecar_npz",
                    "path": dav_artifact_path.relative_to(eval_root).as_posix(),
                    **dav_manifest,
                }
            cold_artifact_path = artifacts_root / "paper_ready" / "cold_cloud_fraction.npz"
            cold_manifest = _write_paper_ready_cold_cloud_fraction_npz(
                cold_artifact_path,
                class_caches=class_caches,
                threshold_k=200.0,
            )
            if cold_manifest:
                paper_ready_manifest["cold_cloud_fraction"] = {
                    "storage": "sidecar_npz",
                    "path": cold_artifact_path.relative_to(eval_root).as_posix(),
                    **cold_manifest,
                }

        report = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "tag": tag,
            "split": split,
            "full_test": full_test,
            "heavy": heavy,
            "n_per_class": n_per,
            "n_plot_per_group": n_plot,
            "gen_batch_size": gen_batch_size,
            "generated_source": (
                {
                    "mode": "sample_bank",
                    **generated_source_info,
                }
                if using_cached_generations
                else {"mode": "online_sampling"}
            ),
            "real_sampling": {
                "mode": "all_test_samples_per_class" if full_test else "sampled_per_class_cap",
                "seed": int(ev["real_seed"]),
            },
            "sample_counts": {
                "real_by_class": real_counts_by_class,
                "generated_by_class": gen_counts_by_class,
            },
            "sampling": sampling_report,
            "bootstrap": {
                "config_key": bootstrap_reps_key,
                "reps": bootstrap_reps,
                "ci_level": bootstrap_ci_level,
            },
            "dav_config": {
                "radius_km": dav_radius_km,
                "pixel_size_km": dav_pixel_size_km,
                "center_region_size": dav_center_region_size,
            },
            "aggregate_primary_raw": aggregate_primary_raw,
            "aggregate_primary_raw_ci": aggregate_primary_raw_ci,
            "evaluator_feature_metric": evaluator_feature_metric_report,
            "macro": macro,
            "macro_ci": macro_ci,
            "per_class": per_class_metrics,
            "paper_ready": paper_ready_manifest,
            "artifacts": artifacts_manifest,
        }
        if wind_targets_kt:
            report["conditioning_targets"] = {
                "wind_kt_by_class": {str(k): float(v) for k, v in sorted(wind_targets_kt.items())}
            }

        _write_json(eval_root / "metrics.json", report)
        return report
