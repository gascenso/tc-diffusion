# tc_diffusion/evaluation/evaluator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from .metrics import (
    PolarBinner,
    cold_cloud_fraction,
    denorm_bt,
    eye_contrast_proxy,
    flatten_features_for_diversity,
    js_divergence,
    psd_radial_batch,
    radial_profile_batch,
    rbf_mmd2,
    summary_stats,
    wasserstein1_from_hist,
)
from .plots import plot_hist_overlay, plot_psd, plot_radial_profiles
from ..data import build_data_backend, load_dataset_index, load_split_file_set
from ..diffusion import resolve_sampling_timestep_schedule
from ..plotting import save_real_generated_comparison_grid


PRIMARY_RAW_AGG_KEYS = (
    "pixel_hist_js",
    "pixel_hist_w1",
    "cold_cloud_fraction_200K_abs_gap",
    "eye_contrast_proxy_abs_gap",
    "psd_l2",
    "feature_mmd2",
    "diversity_feature_space_abs_gap",
    "gen_exceedance_rate_total",
)


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
    ev.setdefault("bootstrap_ci_level", 0.95)
    return ev


def _write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


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


def _sample_real_by_class(
    cfg: Dict[str, Any],
    n_per_class: int,
    seed: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load real BT images and their wind speeds, grouped by SS class."""
    data_cfg = cfg["data"]
    index_path = Path(data_cfg["dataset_index"])
    split_dir = Path(data_cfg["split_dir"])
    bt_min_k = float(data_cfg["bt_min_k"])
    bt_max_k = float(data_cfg["bt_max_k"])
    backend = build_data_backend(data_cfg)

    class_to_files, sample_meta = load_dataset_index(index_path, return_sample_meta=True)
    allowed = load_split_file_set(split_dir, split="val")

    rng = np.random.default_rng(seed)
    out_imgs: Dict[int, np.ndarray] = {}
    out_winds: Dict[int, np.ndarray] = {}

    for c, rels in sorted(class_to_files.items()):
        rels = [r for r in rels if r in allowed]
        if not rels:
            continue

        k = min(n_per_class, len(rels))
        pick = rng.choice(len(rels), size=k, replace=False)

        imgs = []
        winds = []
        for idx in pick:
            rel = rels[int(idx)]
            imgs.append(_load_one_bt_k(backend, rel, bt_min_k, bt_max_k))
            meta = sample_meta.get(rel, {})
            w = meta.get("wmo_wind_kt")
            if w is not None and np.isfinite(float(w)):
                winds.append(float(w))
            else:
                winds.append(_ss_class_midpoint_kt(c))

        out_imgs[int(c)] = np.stack(imgs, axis=0)
        out_winds[int(c)] = np.array(winds, dtype=np.float32)

    return out_imgs, out_winds


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
    if z.shape[0] < 2:
        return 0.0
    d = np.sqrt(((z[:, None] - z[None]) ** 2).sum(-1))
    return float(d.sum() / (z.shape[0] * (z.shape[0] - 1)))


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
    real_eye = _select_rows(cache.real_eye, real_idx)
    gen_eye = _select_rows(cache.gen_raw_eye, gen_idx)

    return {
        "pixel_hist_js": float(js_divergence(rh, gh)),
        "pixel_hist_w1": float(wasserstein1_from_hist(rh, gh, bin_edges=cache.bins)),
        "cold_cloud_fraction_200K_abs_gap": float(abs(real_cold.mean() - gen_cold.mean())),
        "eye_contrast_proxy_abs_gap": float(abs(real_eye.mean() - gen_eye.mean())),
        "psd_l2": float(((real_psd.mean(axis=0) - gen_psd.mean(axis=0)) ** 2).mean()),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space_abs_gap": float(abs(div_real - div_gen)),
        "gen_exceedance_rate_total": float(_select_rows(cache.gen_raw_exceed_total, gen_idx).mean()),
        "real_pixel_mean": float(_select_rows(cache.real_pixel_mean, real_idx).mean()),
        "gen_pixel_mean": float(_select_rows(cache.gen_raw_pixel_mean, gen_idx).mean()),
        "real_eye_mean": float(real_eye.mean()),
        "gen_eye_mean": float(gen_eye.mean()),
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
    real_eye = np.asarray([class_scalars[c]["real_eye_mean"] for c in class_ids], dtype=np.float64)
    gen_eye = np.asarray([class_scalars[c]["gen_eye_mean"] for c in class_ids], dtype=np.float64)

    return {
        "pixel_mean_bt": {
            "corr_class_real": _pearson(classes, real_mean),
            "corr_class_gen": _pearson(classes, gen_mean),
            "corr_real_gen": _pearson(real_mean, gen_mean),
            "mae_gen_vs_real": float(np.mean(np.abs(gen_mean - real_mean))),
        },
        "eye_contrast_proxy": {
            "corr_class_real": _pearson(classes, real_eye),
            "corr_class_gen": _pearson(classes, gen_eye),
            "corr_real_gen": _pearson(real_eye, gen_eye),
            "mae_gen_vs_real": float(np.mean(np.abs(gen_eye - real_eye))),
        },
    }


def _bootstrap_ci_blocks(
    *,
    class_caches: Dict[int, ClassMetricCache],
    aggregate_primary_raw: Dict[str, Any],
    macro: Dict[str, Any],
    bootstrap_reps: int,
    bootstrap_ci_level: float,
    seed: int,
    show_progress: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if bootstrap_reps <= 0 or not class_caches:
        return {}, {}

    rng = np.random.default_rng(seed)
    aggregate_samples: Dict[str, list[float]] = {}
    macro_samples: Dict[str, list[float]] = {}

    rep_iter = range(bootstrap_reps)
    if show_progress:
        rep_iter = tqdm(rep_iter, total=bootstrap_reps, desc="Eval: bootstrap", leave=True)

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
        model: tf.keras.Model,
        diffusion,
        out_dir: Path,
        tag: str,
        heavy: bool,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        ev = _default_eval_cfg(cfg)

        data_cfg = cfg["data"]
        bt_min_k = float(data_cfg["bt_min_k"])
        bt_max_k = float(data_cfg["bt_max_k"])
        image_size = int(data_cfg.get("image_size", 256))

        num_classes = int(cfg["conditioning"]["num_ss_classes"])
        use_wind_speed = bool(cfg.get("conditioning", {}).get("use_wind_speed", False))

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

        gen_batch_size_cfg = ev.get("gen_batch_size")
        if gen_batch_size_cfg is None:
            gen_batch_size = int(data_cfg.get("batch_size", n_per))
        else:
            gen_batch_size = int(gen_batch_size_cfg)
        if gen_batch_size <= 0:
            raise ValueError(f"evaluation.gen_batch_size must be > 0, got {gen_batch_size}")
        gen_batch_size = min(gen_batch_size, n_per)

        bootstrap_reps = int(ev.get("bootstrap_reps", 500))
        if bootstrap_reps < 0:
            raise ValueError(f"evaluation.bootstrap_reps must be >= 0, got {bootstrap_reps}")
        bootstrap_ci_level = float(ev.get("bootstrap_ci_level", 0.95))
        if not (0.0 < bootstrap_ci_level < 1.0):
            raise ValueError(
                f"evaluation.bootstrap_ci_level must lie strictly between 0 and 1, got {bootstrap_ci_level}"
            )

        tf.random.set_seed(ev["seed"])
        np.random.seed(ev["seed"])

        eval_root = out_dir / "eval" / tag
        eval_root.mkdir(parents=True, exist_ok=True)

        real_by_class, real_wind_by_class = _sample_real_by_class(cfg, n_per, ev["real_seed"])

        gen_raw_by_class: Dict[int, np.ndarray] = {}
        gen_post_by_class: Dict[int, np.ndarray] = {}
        wind_targets_kt: Dict[int, float] = {}

        class_iter = range(num_classes)
        if show_progress:
            class_iter = tqdm(class_iter, desc="Eval: generating", leave=True)

        for c in class_iter:
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
            post_chunks = []
            offset = 0
            remaining = n_per
            while remaining > 0:
                bsz = min(gen_batch_size, remaining)
                wind_batch = wind_schedule[offset:offset + bsz] if wind_schedule is not None else None
                sampling_steps = ev.get("sampling_steps", ev.get("ddim_steps", None))
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
                    show_progress=show_progress,
                    return_both=True,
                )
                raw_chunks.append(sample_outputs["raw_final"].numpy())
                post_chunks.append(sample_outputs["clipped_final"].numpy())
                remaining -= bsz
                offset += bsz

            raw_norm = np.concatenate(raw_chunks, axis=0)
            post_norm = np.concatenate(post_chunks, axis=0)
            gen_raw_by_class[c] = denorm_bt(raw_norm, bt_min_k, bt_max_k)[..., 0]
            gen_post_by_class[c] = denorm_bt(post_norm, bt_min_k, bt_max_k)[..., 0]

        for c in range(num_classes):
            if c not in real_by_class or c not in gen_raw_by_class:
                continue

            n_show = min(n_plot, real_by_class[c].shape[0], gen_raw_by_class[c].shape[0])
            if n_show <= 0:
                continue

            save_real_generated_comparison_grid(
                real_k=real_by_class[c],
                gen_k=gen_raw_by_class[c],
                path=str(eval_root / f"samples_class_{c}.png"),
                n_show=n_show,
                ncols=min(5, n_show),
                real_title=f"Real (n={n_show})",
                gen_title=f"Generated (n={n_show})",
                suptitle=f"Class {c}: Real vs Generated Samples",
            )

        binner = PolarBinner(image_size, image_size, int(ev["profile_bins"]), 360)

        class_caches: Dict[int, ClassMetricCache] = {}
        per_class_metrics = {}
        per_class_curves = {}
        class_scalars = {}

        for c in range(num_classes):
            if c not in real_by_class or c not in gen_raw_by_class or c not in gen_post_by_class:
                continue
            cache = _build_class_metric_cache(
                real_k=real_by_class[c],
                gen_k_raw=gen_raw_by_class[c],
                binner=binner,
                psd_bins=int(ev["psd_bins"]),
                bt_min_k=bt_min_k,
                bt_max_k=bt_max_k,
            )
            class_caches[c] = cache
            metrics, curves = _compute_per_class_report(cache)
            per_class_metrics[c] = metrics
            per_class_curves[c] = curves
            class_scalars[c] = _compute_class_scalar_summary(cache)

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

        aggregate_primary_raw = _compute_aggregate_primary_raw(class_scalars)
        macro = _compute_macro_metrics(class_scalars)
        aggregate_primary_raw_ci, macro_ci = _bootstrap_ci_blocks(
            class_caches=class_caches,
            aggregate_primary_raw=aggregate_primary_raw,
            macro=macro,
            bootstrap_reps=bootstrap_reps,
            bootstrap_ci_level=bootstrap_ci_level,
            seed=int(ev["seed"]),
            show_progress=show_progress,
        )

        report = {
            "tag": tag,
            "heavy": heavy,
            "n_per_class": n_per,
            "n_plot_per_group": n_plot,
            "gen_batch_size": gen_batch_size,
            "sampling": {
                "sampler": str(ev["sampler"]),
                "guidance_scale": float(ev["guidance_scale"]),
                "sampling_steps": ev.get("sampling_steps", ev.get("ddim_steps", None)),
                "timestep_schedule": str(ev["timestep_schedule"]),
                "ddim_eta": float(ev.get("ddim_eta", 0.0)),
            },
            "bootstrap": {
                "reps": bootstrap_reps,
                "ci_level": bootstrap_ci_level,
            },
            "aggregate_primary_raw": aggregate_primary_raw,
            "aggregate_primary_raw_ci": aggregate_primary_raw_ci,
            "macro": macro,
            "macro_ci": macro_ci,
            "per_class": per_class_metrics,
        }
        if wind_targets_kt:
            report["conditioning_targets"] = {
                "wind_kt_by_class": {str(k): float(v) for k, v in sorted(wind_targets_kt.items())}
            }

        _write_json(eval_root / "metrics.json", report)
        return report
