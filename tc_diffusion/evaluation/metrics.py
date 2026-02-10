# tc_diffusion/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np


def denorm_bt(x: np.ndarray, bt_min_k: float, bt_max_k: float) -> np.ndarray:
    """
    x: [..., H, W, 1] in [-1, 1]
    returns Kelvin in [bt_min_k, bt_max_k] (not clipped)
    """
    x = np.asarray(x, dtype=np.float32)
    bt01 = (x + 1.0) * 0.5
    return bt01 * (bt_max_k - bt_min_k) + bt_min_k


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def wasserstein1_from_hist(
    p: np.ndarray,
    q: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
) -> float:
    """
    1D Wasserstein-1 distance computed from histograms on the same bins.
    If bin_edges are provided, returns distance in data units (e.g. Kelvin).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    cdf_gap = np.abs(cdf_p - cdf_q)

    if bin_edges is None:
        # fallback: assumes unit bin width
        return float(np.sum(cdf_gap))

    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.shape[0] != p.shape[0] + 1:
        raise ValueError(
            f"bin_edges must have length len(p)+1; got {edges.shape[0]} for len(p)={p.shape[0]}"
        )

    widths = np.diff(edges)
    return float(np.sum(cdf_gap * widths))


def summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skew": float(np.mean(((x - np.mean(x)) / (np.std(x) + 1e-12)) ** 3)),
        "kurtosis": float(np.mean(((x - np.mean(x)) / (np.std(x) + 1e-12)) ** 4) - 3.0),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


@dataclass
class PolarBinner:
    """
    Precomputes radius bins and theta bins for fast per-image radial / azimuthal stats.
    Assumes square images with center at (H/2, W/2).
    """
    H: int
    W: int
    r_bins: int = 96
    theta_bins: int = 360

    def __post_init__(self):
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        cy = (self.H - 1) / 2.0
        cx = (self.W - 1) / 2.0
        dy = yy - cy
        dx = xx - cx
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)  # [-pi, pi]
        theta = (theta + np.pi) / (2 * np.pi)  # [0,1)

        r_max = np.max(r)
        r_idx = np.floor((r / (r_max + 1e-12)) * self.r_bins).astype(np.int32)
        r_idx = np.clip(r_idx, 0, self.r_bins - 1)

        t_idx = np.floor(theta * self.theta_bins).astype(np.int32)
        t_idx = np.clip(t_idx, 0, self.theta_bins - 1)

        self.r_max = float(r_max)
        self._r_idx = r_idx.reshape(-1)
        self._t_idx = t_idx.reshape(-1)

    @property
    def r_idx_flat(self) -> np.ndarray:
        return self._r_idx

    @property
    def t_idx_flat(self) -> np.ndarray:
        return self._t_idx

    def radial_mean_and_azstd(self, img2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        img2d: (H,W) Kelvin or normalized, doesn't matter (just consistent)
        Returns:
          radial_mean: (r_bins,)
          az_std:      (r_bins,) = std over theta at each radius bin (computed via per-theta means)
        """
        x = np.asarray(img2d, dtype=np.float64).reshape(-1)
        r_idx = self._r_idx

        # radial mean
        sum_r = np.bincount(r_idx, weights=x, minlength=self.r_bins)
        cnt_r = np.bincount(r_idx, minlength=self.r_bins)
        mean_r = sum_r / (cnt_r + 1e-12)

        # azimuthal variability:
        # For each radius bin, bin by theta and compute mean over theta bins, then std over theta means.
        # This is robust and interpretable.
        t_idx = self._t_idx
        az_std = np.zeros(self.r_bins, dtype=np.float64)
        for rb in range(self.r_bins):
            mask = (r_idx == rb)
            if not np.any(mask):
                continue
            x_rb = x[mask]
            t_rb = t_idx[mask]
            sum_t = np.bincount(t_rb, weights=x_rb, minlength=self.theta_bins)
            cnt_t = np.bincount(t_rb, minlength=self.theta_bins)
            mean_t = sum_t / (cnt_t + 1e-12)

            # Only consider theta bins that had any pixels in this radius bin
            valid = cnt_t > 0
            if np.sum(valid) < 2:
                az_std[rb] = 0.0
            else:
                az_std[rb] = float(np.std(mean_t[valid]))

        return mean_r.astype(np.float32), az_std.astype(np.float32)


def radial_profile_batch(
    imgs: np.ndarray, binner: PolarBinner
) -> Tuple[np.ndarray, np.ndarray]:
    """
    imgs: (N,H,W) float
    Returns:
      mean_profiles: (N, r_bins)
      azstd_profiles:(N, r_bins)
    """
    N = imgs.shape[0]
    mean_prof = np.zeros((N, binner.r_bins), dtype=np.float32)
    azstd_prof = np.zeros((N, binner.r_bins), dtype=np.float32)
    for i in range(N):
        m, s = binner.radial_mean_and_azstd(imgs[i])
        mean_prof[i] = m
        azstd_prof[i] = s
    return mean_prof, azstd_prof


def cold_cloud_fraction(imgs_k: np.ndarray, threshold_k: float = 200.0) -> np.ndarray:
    """
    imgs_k: (N,H,W) in Kelvin
    returns (N,) fraction of pixels colder than threshold
    """
    return np.mean(imgs_k < float(threshold_k), axis=(1, 2)).astype(np.float32)


def eye_contrast_proxy(mean_profile_k: np.ndarray, inner_frac: float = 0.12, ring_frac: float = 0.25) -> np.ndarray:
    """
    Uses radial mean profile: higher contrast between warm inner core and cold ring suggests eye+eyewall structure.
    mean_profile_k: (N, r_bins)
    inner_frac: fraction of radius bins considered 'eye'
    ring_frac:  fraction of radius bins around which we seek cold ring
    """
    N, R = mean_profile_k.shape
    inner_bins = max(1, int(R * inner_frac))
    ring_center = int(R * ring_frac)
    ring_window = max(2, int(R * 0.05))

    eye_mean = np.mean(mean_profile_k[:, :inner_bins], axis=1)
    lo = max(0, ring_center - ring_window)
    hi = min(R, ring_center + ring_window)
    ring_min = np.min(mean_profile_k[:, lo:hi], axis=1)
    return (eye_mean - ring_min).astype(np.float32)


def psd_radial(img2d: np.ndarray, psd_bins: int = 96) -> np.ndarray:
    """
    Radially averaged 2D power spectral density.
    img2d: (H,W) float (Kelvin or normalized)
    returns: (psd_bins,) log10 power averaged in radial frequency bins
    """
    x = np.asarray(img2d, dtype=np.float64)
    H, W = x.shape
    x = x - np.mean(x)

    F = np.fft.fft2(x)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)

    yy, xx = np.mgrid[0:H, 0:W]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.max(r)

    ridx = np.floor((r / (r_max + 1e-12)) * psd_bins).astype(np.int32)
    ridx = np.clip(ridx, 0, psd_bins - 1).reshape(-1)

    Pflat = P.reshape(-1)
    sum_r = np.bincount(ridx, weights=Pflat, minlength=psd_bins)
    cnt_r = np.bincount(ridx, minlength=psd_bins)
    avg = sum_r / (cnt_r + 1e-12)

    avg = np.log10(avg + 1e-12)
    return avg.astype(np.float32)


def psd_radial_batch(imgs: np.ndarray, psd_bins: int = 96) -> np.ndarray:
    N = imgs.shape[0]
    out = np.zeros((N, psd_bins), dtype=np.float32)
    for i in range(N):
        out[i] = psd_radial(imgs[i], psd_bins=psd_bins)
    return out


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Squared MMD with RBF kernel.
    X: (n,d), Y: (m,d)
    gamma: kernel parameter; if None uses median heuristic.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # median heuristic for gamma
    if gamma is None:
        Z = np.vstack([X, Y])
        # sample pairs for median estimate (cheap)
        rng = np.random.default_rng(0)
        idx = rng.choice(Z.shape[0], size=min(200, Z.shape[0]), replace=False)
        Zs = Z[idx]
        d2 = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (2.0 * (med + 1e-12))

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)

    # unbiased-ish: remove diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    n = X.shape[0]
    m = Y.shape[0]
    mmd2 = (Kxx.sum() / (n * (n - 1) + 1e-12)
            + Kyy.sum() / (m * (m - 1) + 1e-12)
            - 2.0 * Kxy.mean())
    return float(mmd2)


def flatten_features_for_diversity(mean_prof: np.ndarray, psd_prof: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Construct a compact feature vector per sample, interpretable but useful for coverage/diversity metrics.
    """
    feats = [mean_prof, psd_prof]
    if extra is not None:
        feats.append(extra)
    X = np.concatenate(feats, axis=1)
    # standardize for distances
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    return X.astype(np.float32)
