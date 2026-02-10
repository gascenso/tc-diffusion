# tc_diffusion/evaluation/evaluator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
import xarray as xr
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
from ..plotting import save_image_grid
from ..data import load_dataset_index, load_split_file_set


# -----------------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------------

def _default_eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = dict(cfg.get("evaluation", {}))
    ev.setdefault("enabled", True)
    ev.setdefault("every_epochs", 5)
    ev.setdefault("heavy_every_epochs", 25)
    ev.setdefault("n_per_class_light", 5)
    ev.setdefault("n_per_class_heavy", 50)
    ev.setdefault("gen_batch_size", None)
    ev.setdefault("guidance_scale", 0.0)
    ev.setdefault("sampler", "ddpm")
    ev.setdefault("seed", 123)
    ev.setdefault("real_seed", 123)
    ev.setdefault("profile_bins", 96)
    ev.setdefault("psd_bins", 96)

    pcfg = dict(ev.get("proxy_classifier", {}))
    pcfg.setdefault("enabled", True)
    pcfg.setdefault("max_train_per_class", 2000)
    pcfg.setdefault("C", 1.0)
    pcfg.setdefault("solver", "lbfgs")
    pcfg.setdefault("max_iter", 200)
    ev["proxy_classifier"] = pcfg
    return ev


def _write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# real data loading
# -----------------------------------------------------------------------------

def _load_one_bt_k(nc_path: Path, bt_min_k: float, bt_max_k: float) -> np.ndarray:
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        bt = ds["bt"].values.astype(np.float32)

    bt = np.nan_to_num(bt, nan=bt_min_k)
    bt = np.clip(bt, bt_min_k, bt_max_k)
    return bt


def _sample_real_by_class(
    cfg: Dict[str, Any],
    n_per_class: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    data_cfg = cfg["data"]
    data_root = Path(data_cfg["data_root"])
    index_path = Path(data_cfg["dataset_index"])
    split_dir = Path(data_cfg["split_dir"])
    bt_min_k = float(data_cfg["bt_min_k"])
    bt_max_k = float(data_cfg["bt_max_k"])

    class_to_files = load_dataset_index(index_path)
    allowed = load_split_file_set(split_dir, split="val")

    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}

    for c, rels in sorted(class_to_files.items()):
        rels = [r for r in rels if r in allowed]
        if not rels:
            continue

        k = min(n_per_class, len(rels))
        pick = rng.choice(len(rels), size=k, replace=False)

        imgs = []
        for idx in pick:
            imgs.append(_load_one_bt_k(data_root / rels[int(idx)], bt_min_k, bt_max_k))

        out[int(c)] = np.stack(imgs, axis=0)

    return out


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
    w1 = wasserstein1_from_hist(rh, gh)

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

    r_feat = flatten_features_for_diversity(r_mean, r_psd)
    g_feat = flatten_features_for_diversity(g_mean, g_psd)

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
        n_per = int(ev["n_per_class_heavy"] if heavy else ev["n_per_class_light"])
        if n_per <= 0:
            raise ValueError(f"evaluation n_per_class must be > 0, got {n_per}")

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

        gen_by_class: Dict[int, np.ndarray] = {}

        class_iter = range(num_classes)
        if show_progress:
            class_iter = tqdm(class_iter, desc="Eval: generating", leave=True)

        for c in class_iter:
            chunks = []
            remaining = n_per
            while remaining > 0:
                bsz = min(gen_batch_size, remaining)
                x_chunk = diffusion.sample(
                    model,
                    batch_size=bsz,
                    image_size=image_size,
                    cond_value=c,
                    guidance_scale=float(ev["guidance_scale"]),
                    show_progress=show_progress,
                ).numpy()
                chunks.append(x_chunk)
                remaining -= bsz

            x = np.concatenate(chunks, axis=0)

            gen_k_raw = denorm_bt(x, bt_min_k, bt_max_k)[..., 0]
            gen_by_class[c] = gen_k_raw

            save_image_grid(
                x,
                str(eval_root / f"samples_class_{c}.png"),
                bt_min_k,
                bt_max_k,
                ncols=min(5, n_per),
            )

        real_by_class = _sample_real_by_class(cfg, n_per, ev["real_seed"])

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
            "gen_batch_size": gen_batch_size,
            "macro": {},
            "per_class": per_class_metrics,
        }

        _write_json(eval_root / "metrics.json", report)
        return report
