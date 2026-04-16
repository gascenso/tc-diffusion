# tc_diffusion/evaluation/evaluator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from .metrics import (
    denorm_bt,
    summary_stats,
    js_divergence,
    wasserstein1_from_hist,
    PolarBinner,
    radial_profile_batch,
    cold_cloud_fraction,
    eye_contrast_proxy,
    psd_radial_batch,
    flatten_features_for_diversity,
    rbf_mmd2,
)

from .plots import plot_radial_profiles, plot_psd, plot_hist_overlay
from ..plotting import save_real_generated_comparison_grid
from ..data import build_data_backend, load_dataset_index, load_split_file_set


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
    ev.setdefault("n_per_class_heavy", 50)
    ev.setdefault("gen_batch_size", None)
    ev.setdefault("guidance_scale", 0.0)
    ev.setdefault("sampler", "ddpm")
    ev.setdefault("seed", 123)
    ev.setdefault("real_seed", 123)
    ev.setdefault("profile_bins", 96)
    ev.setdefault("psd_bins", 96)

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


def _compute_macro_metrics(per_class_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if not per_class_metrics:
        return {}

    class_ids = sorted(int(c) for c in per_class_metrics.keys())
    classes = np.asarray(class_ids, dtype=np.float64)
    real_mean = np.asarray(
        [per_class_metrics[c]["pixel_stats"]["real"]["mean"] for c in class_ids],
        dtype=np.float64,
    )
    gen_mean = np.asarray(
        [per_class_metrics[c]["pixel_stats"]["gen_raw"]["mean"] for c in class_ids],
        dtype=np.float64,
    )
    real_eye = np.asarray(
        [per_class_metrics[c]["eye_contrast_proxy"]["real_mean"] for c in class_ids],
        dtype=np.float64,
    )
    gen_eye = np.asarray(
        [per_class_metrics[c]["eye_contrast_proxy"]["gen_mean"] for c in class_ids],
        dtype=np.float64,
    )

    out = {
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
    return out


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
    """Load real BT images and their wind speeds, grouped by SS class.

    Returns:
        images_by_class:  class → (N, H, W) float32 array of BT in Kelvin
        winds_by_class:   class → (N,) float32 array of wind speed in kt
                          (falls back to class midpoint when metadata absent)
    """
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
# per-class metric computation
# -----------------------------------------------------------------------------

def _hist_density(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, density=True)
    h = h.astype(np.float64)
    return h / (h.sum() + 1e-12)


def _compute_suite_for_class(
    *,
    real_k: np.ndarray,
    gen_k_raw: np.ndarray,
    binner: PolarBinner,
    profile_bins: int,
    psd_bins: int,
    bt_min_k: float,
    bt_max_k: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # ------------------------------------------------------------------
    # RAW exceedance diagnostics (no clipping)
    # ------------------------------------------------------------------
    rflat = real_k.reshape(-1)
    gflat_raw = gen_k_raw.reshape(-1)

    below = float(np.mean(gflat_raw < bt_min_k))
    above = float(np.mean(gflat_raw > bt_max_k))
    raw_min = float(np.min(gflat_raw))
    raw_max = float(np.max(gflat_raw))

    # ------------------------------------------------------------------
    # CLIPPED copy for stable structure / hist metrics
    # ------------------------------------------------------------------
    gen_k_clip = np.clip(gen_k_raw, bt_min_k, bt_max_k)
    gflat_clip = gen_k_clip.reshape(-1)

    # ------------------------------------------------------------------
    # pixel statistics
    # ------------------------------------------------------------------
    pix_real = summary_stats(rflat)
    pix_gen_raw = summary_stats(gflat_raw)
    pix_gen_clip = summary_stats(gflat_clip)

    bins = np.linspace(bt_min_k, bt_max_k, 129)
    rh = _hist_density(rflat, bins)
    gh = _hist_density(gflat_clip, bins)

    js = js_divergence(rh, gh)
    w1 = wasserstein1_from_hist(rh, gh, bin_edges=bins)

    # ------------------------------------------------------------------
    # radial / azimuthal structure (on clipped)
    # ------------------------------------------------------------------
    r_mean, r_azstd = radial_profile_batch(real_k, binner)
    g_mean, g_azstd = radial_profile_batch(gen_k_clip, binner)

    r_mean_mu, r_mean_sd = r_mean.mean(0), r_mean.std(0)
    g_mean_mu, g_mean_sd = g_mean.mean(0), g_mean.std(0)

    r_az_mu, r_az_sd = r_azstd.mean(0), r_azstd.std(0)
    g_az_mu, g_az_sd = g_azstd.mean(0), g_azstd.std(0)

    cold_r = cold_cloud_fraction(real_k, threshold_k=200.0)
    cold_g = cold_cloud_fraction(gen_k_clip, threshold_k=200.0)

    eye_r = eye_contrast_proxy(r_mean)
    eye_g = eye_contrast_proxy(g_mean)

    r_psd = psd_radial_batch(real_k, psd_bins)
    g_psd = psd_radial_batch(gen_k_clip, psd_bins)

    r_psd_mu, r_psd_sd = r_psd.mean(0), r_psd.std(0)
    g_psd_mu, g_psd_sd = g_psd.mean(0), g_psd.std(0)

    # Fit the scaler on real features only, then apply the same scaler to
    # generated features so both sets live in the same coordinate system.
    r_feat, feat_scaler = flatten_features_for_diversity(r_mean, r_psd)
    g_feat, _ = flatten_features_for_diversity(g_mean, g_psd, scaler=feat_scaler)

    mmd2 = rbf_mmd2(r_feat, g_feat)

    def _mean_pair_dist(Z):
        if Z.shape[0] < 2:
            return 0.0
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        return float(D.sum() / (Z.shape[0] * (Z.shape[0] - 1)))

    metrics = {
        "gen_exceedance_rate": {
            "below_bt_min": below,
            "above_bt_max": above,
            "total": below + above,
            "raw_min": raw_min,
            "raw_max": raw_max,
        },
        "pixel_stats": {
            "real": pix_real,
            "gen_raw": pix_gen_raw,
            "gen_clipped": pix_gen_clip,
        },
        "pixel_hist_js": js,
        "pixel_hist_w1": w1,
        "cold_cloud_fraction_200K": {
            "real_mean": float(cold_r.mean()),
            "real_std": float(cold_r.std()),
            "gen_mean": float(cold_g.mean()),
            "gen_std": float(cold_g.std()),
        },
        "eye_contrast_proxy": {
            "real_mean": float(eye_r.mean()),
            "real_std": float(eye_r.std()),
            "gen_mean": float(eye_g.mean()),
            "gen_std": float(eye_g.std()),
        },
        "psd_l2": float(((r_psd_mu - g_psd_mu) ** 2).mean()),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space": {
            "real": _mean_pair_dist(r_feat),
            "gen": _mean_pair_dist(g_feat),
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

        # Total generated samples per class is n_per; this only controls
        # micro-batch size to fit hardware limits.
        gen_batch_size_cfg = ev.get("gen_batch_size")
        if gen_batch_size_cfg is None:
            gen_batch_size = int(data_cfg.get("batch_size", n_per))
        else:
            gen_batch_size = int(gen_batch_size_cfg)
        if gen_batch_size <= 0:
            raise ValueError(f"evaluation.gen_batch_size must be > 0, got {gen_batch_size}")
        gen_batch_size = min(gen_batch_size, n_per)

        tf.random.set_seed(ev["seed"])
        np.random.seed(ev["seed"])

        eval_root = out_dir / "eval" / tag
        eval_root.mkdir(parents=True, exist_ok=True)

        # Load real samples first so we can match wind conditioning exactly.
        real_by_class, real_wind_by_class = _sample_real_by_class(cfg, n_per, ev["real_seed"])

        gen_by_class: Dict[int, np.ndarray] = {}
        wind_targets_kt: Dict[int, float] = {}

        class_iter = range(num_classes)
        if show_progress:
            class_iter = tqdm(class_iter, desc="Eval: generating", leave=True)

        for c in class_iter:
            # Determine per-sample wind speeds used for this class's generation.
            # If we have real wind speeds, use them directly so the generated
            # conditioning distribution matches the real one.  If there are fewer
            # real samples than n_per (rare classes), resample with replacement.
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
                    # No real samples for this class; fall back to fixed midpoint
                    wind_schedule = np.full(n_per, _resolve_eval_wind_target_kt(cfg, c), dtype=np.float32)
                    wind_targets_kt[int(c)] = float(wind_schedule[0])
            else:
                wind_schedule = None

            chunks = []
            offset = 0
            remaining = n_per
            while remaining > 0:
                bsz = min(gen_batch_size, remaining)
                wind_batch = wind_schedule[offset:offset + bsz] if wind_schedule is not None else None
                x_chunk = diffusion.sample(
                    model,
                    batch_size=bsz,
                    image_size=image_size,
                    cond_value=c,
                    wind_value_kt=wind_batch,
                    guidance_scale=float(ev["guidance_scale"]),
                    show_progress=show_progress,
                ).numpy()
                chunks.append(x_chunk)
                remaining -= bsz
                offset += bsz

            x = np.concatenate(chunks, axis=0)

            gen_k_raw = denorm_bt(x, bt_min_k, bt_max_k)[..., 0]
            gen_by_class[c] = gen_k_raw

        for c in range(num_classes):
            if c not in real_by_class or c not in gen_by_class:
                continue

            n_show = min(n_plot, real_by_class[c].shape[0], gen_by_class[c].shape[0])
            if n_show <= 0:
                continue

            save_real_generated_comparison_grid(
                real_k=real_by_class[c],
                gen_k=gen_by_class[c],
                path=str(eval_root / f"samples_class_{c}.png"),
                n_show=n_show,
                ncols=min(5, n_show),
                real_title=f"Real (n={n_show})",
                gen_title=f"Generated (n={n_show})",
                suptitle=f"Class {c}: Real vs Generated Samples",
            )

        binner = PolarBinner(image_size, image_size, ev["profile_bins"], 360)

        per_class_metrics = {}
        per_class_curves = {}

        for c in range(num_classes):
            if c not in real_by_class:
                continue
            m, cur = _compute_suite_for_class(
                real_k=real_by_class[c],
                gen_k_raw=gen_by_class[c],
                binner=binner,
                profile_bins=ev["profile_bins"],
                psd_bins=ev["psd_bins"],
                bt_min_k=bt_min_k,
                bt_max_k=bt_max_k,
            )
            per_class_metrics[c] = m
            per_class_curves[c] = cur

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

        report = {
            "tag": tag,
            "heavy": heavy,
            "n_per_class": n_per,
            "n_plot_per_group": n_plot,
            "gen_batch_size": gen_batch_size,
            "macro": _compute_macro_metrics(per_class_metrics),
            "per_class": per_class_metrics,
        }
        if wind_targets_kt:
            report["conditioning_targets"] = {
                "wind_kt_by_class": {str(k): float(v) for k, v in sorted(wind_targets_kt.items())}
            }

        _write_json(eval_root / "metrics.json", report)
        return report
