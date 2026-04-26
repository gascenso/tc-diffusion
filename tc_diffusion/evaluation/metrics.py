# tc_diffusion/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

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
    # eps only in the denominator: zero-probability bins contribute 0*log(0)=0
    # exactly, without the small bias that adding eps to the numerator introduces.
    kl_pm_terms = np.zeros_like(p)
    kl_qm_terms = np.zeros_like(q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    kl_pm_terms[p_mask] = p[p_mask] * np.log(p[p_mask] / (m[p_mask] + eps))
    kl_qm_terms[q_mask] = q[q_mask] * np.log(q[q_mask] / (m[q_mask] + eps))
    kl_pm = np.sum(kl_pm_terms)
    kl_qm = np.sum(kl_qm_terms)
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


@dataclass
class DAVComputer:
    """
    Literature-style Deviation Angle Variance (DAV) helper.

    DAV is computed from Sobel image gradients inside a fixed-radius disk around
    the storm center. For each pixel, the deviation angle is the angle between
    the image-gradient vector and the radial vector extending from the center.
    The per-image DAV is the variance of those deviation angles, expressed in
    degree squared.

    To reduce sensitivity to small center-location errors, the literature often
    averages DAV over a small center neighborhood. We follow that approach with
    a default 3x3 center region.
    """

    H: int
    W: int
    pixel_size_km: float = 8.0
    radius_km: float = 300.0
    center_region_size: int = 3

    def __post_init__(self):
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"DAVComputer expects positive image dimensions, got H={self.H}, W={self.W}.")
        if self.pixel_size_km <= 0.0:
            raise ValueError(f"DAVComputer.pixel_size_km must be > 0, got {self.pixel_size_km}.")
        if self.radius_km <= 0.0:
            raise ValueError(f"DAVComputer.radius_km must be > 0, got {self.radius_km}.")
        if self.center_region_size <= 0 or (self.center_region_size % 2) != 1:
            raise ValueError(
                "DAVComputer.center_region_size must be a positive odd integer, "
                f"got {self.center_region_size}."
            )

        yy, xx = np.mgrid[0:self.H, 0:self.W]
        cy0 = (self.H - 1) / 2.0
        cx0 = (self.W - 1) / 2.0
        half = self.center_region_size // 2

        radial_x_rows = []
        radial_y_rows = []
        mask_rows = []
        for y_shift in range(-half, half + 1):
            for x_shift in range(-half, half + 1):
                cy = cy0 + float(y_shift)
                cx = cx0 + float(x_shift)
                dy = yy - cy
                dx = xx - cx
                radius_px = np.sqrt(dx * dx + dy * dy)
                radius_px_safe = np.maximum(radius_px, 1e-12)
                radial_x_rows.append((dx / radius_px_safe).reshape(-1).astype(np.float32))
                radial_y_rows.append((dy / radius_px_safe).reshape(-1).astype(np.float32))
                radius_km_grid = radius_px * float(self.pixel_size_km)
                mask = np.logical_and(radius_km_grid > 0.0, radius_km_grid <= float(self.radius_km))
                if not np.any(mask):
                    raise ValueError(
                        "DAVComputer radius selects no pixels. "
                        f"Got H={self.H}, W={self.W}, pixel_size_km={self.pixel_size_km}, "
                        f"radius_km={self.radius_km}, center_region_size={self.center_region_size}."
                    )
                mask_rows.append(mask.reshape(-1).astype(np.float32))

        self.radial_x = np.stack(radial_x_rows, axis=0)
        self.radial_y = np.stack(radial_y_rows, axis=0)
        self.mask = np.stack(mask_rows, axis=0)
        self.weight_sum = np.maximum(np.sum(self.mask, axis=1, keepdims=True), 1.0)

    def _sobel_gradient_components(self, img2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(img2d, dtype=np.float64)
        if x.shape != (self.H, self.W):
            raise ValueError(f"DAVComputer expected image shape {(self.H, self.W)}, got {x.shape}.")

        p = np.pad(x, pad_width=1, mode="reflect")
        x00 = p[:-2, :-2]
        x01 = p[:-2, 1:-1]
        x02 = p[:-2, 2:]
        x10 = p[1:-1, :-2]
        x12 = p[1:-1, 2:]
        x20 = p[2:, :-2]
        x21 = p[2:, 1:-1]
        x22 = p[2:, 2:]

        grad_x = (x00 + 2.0 * x10 + x20) - (x02 + 2.0 * x12 + x22)
        grad_y = (x00 + 2.0 * x01 + x02) - (x20 + 2.0 * x21 + x22)
        return grad_x, grad_y

    def per_image(self, img2d: np.ndarray) -> float:
        grad_x, grad_y = self._sobel_gradient_components(img2d)
        grad_x_flat = grad_x.reshape(1, -1)
        grad_y_flat = grad_y.reshape(1, -1)
        grad_norm = np.sqrt(grad_x_flat * grad_x_flat + grad_y_flat * grad_y_flat + 1e-8)

        cos_theta = (grad_x_flat * self.radial_x + grad_y_flat * self.radial_y) / (grad_norm + 1e-8)
        np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
        dev_angle_deg = np.arccos(cos_theta) * (180.0 / np.pi)

        mean_angle = np.sum(dev_angle_deg * self.mask, axis=1, keepdims=True) / self.weight_sum
        dav_by_center = np.sum(np.square(dev_angle_deg - mean_angle) * self.mask, axis=1) / self.weight_sum[:, 0]
        return float(np.mean(dav_by_center))

    def batch(self, imgs: np.ndarray) -> np.ndarray:
        arr = np.asarray(imgs, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (self.H, self.W):
            raise ValueError(
                f"DAVComputer.batch expects shape (N, {self.H}, {self.W}), got {arr.shape}."
            )
        out = np.zeros(arr.shape[0], dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = self.per_image(arr[i])
        return out


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


def eye_contrast_proxy(mean_profile_k: np.ndarray, inner_frac: float = 0.12, ring_frac: float = 0.20) -> np.ndarray:
    """
    Uses radial mean profile: higher contrast between warm inner core and cold eyewall ring.
    mean_profile_k: (N, r_bins)
    inner_frac: fraction of radius considered 'eye' (default 0.12)
    ring_frac:  fraction of radius marking the eyewall ring centre. The default
                is set far enough outside the inner-eye window that the ring
                search does not overlap the eye region on the common 96-bin
                setup, while still staying well inside the outer rainband region.

    Uses ring_mean instead of ring_min to avoid inflating the metric due to
    a single anomalously cold pixel anywhere in the ring window.
    """
    N, R = mean_profile_k.shape
    inner_bins = max(1, int(R * inner_frac))
    ring_center = int(R * ring_frac)
    ring_window = max(2, int(R * 0.05))

    eye_mean = np.mean(mean_profile_k[:, :inner_bins], axis=1)
    lo = max(inner_bins, ring_center - ring_window)
    lo = min(lo, max(R - 1, 0))
    hi = min(R, max(lo + 1, ring_center + ring_window))
    ring_mean = np.mean(mean_profile_k[:, lo:hi], axis=1)
    return (eye_mean - ring_mean).astype(np.float32)


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

    n = X.shape[0]
    m = Y.shape[0]

    block_size = 128

    def _rbf_block_sum(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        a_norm = np.sum(a * a, axis=1)
        b_norm = np.sum(b * b, axis=1)
        for i0 in range(0, a.shape[0], block_size):
            i1 = min(i0 + block_size, a.shape[0])
            ai = a[i0:i1]
            ai_norm = a_norm[i0:i1][:, None]
            for j0 in range(0, b.shape[0], block_size):
                j1 = min(j0 + block_size, b.shape[0])
                bj = b[j0:j1]
                bj_norm = b_norm[j0:j1][None, :]
                d2 = ai_norm + bj_norm - 2.0 * (ai @ bj.T)
                np.maximum(d2, 0.0, out=d2)
                total += float(np.exp(-gamma * d2).sum())
        return total

    def _rbf_block_sum_symmetric(a: np.ndarray) -> float:
        total = 0.0
        a_norm = np.sum(a * a, axis=1)
        for i0 in range(0, a.shape[0], block_size):
            i1 = min(i0 + block_size, a.shape[0])
            ai = a[i0:i1]
            ai_norm = a_norm[i0:i1][:, None]
            for j0 in range(i0, a.shape[0], block_size):
                j1 = min(j0 + block_size, a.shape[0])
                aj = a[j0:j1]
                aj_norm = a_norm[j0:j1][None, :]
                d2 = ai_norm + aj_norm - 2.0 * (ai @ aj.T)
                np.maximum(d2, 0.0, out=d2)
                block = np.exp(-gamma * d2)
                if i0 == j0:
                    total += float(block.sum() - np.trace(block))
                else:
                    total += 2.0 * float(block.sum())
        return total

    kxx_sum = _rbf_block_sum_symmetric(X)
    kyy_sum = _rbf_block_sum_symmetric(Y)
    kxy_sum = _rbf_block_sum(X, Y)

    mmd2 = (kxx_sum / (n * (n - 1) + 1e-12)
            + kyy_sum / (m * (m - 1) + 1e-12)
            - 2.0 * (kxy_sum / (n * m + 1e-12)))
    return float(mmd2)


def flatten_features_for_diversity(
    mean_prof: np.ndarray,
    psd_prof: np.ndarray,
    extra: Optional[np.ndarray] = None,
    scaler: Optional[tuple] = None,
) -> tuple:
    """Compact per-sample feature matrix for coverage/diversity metrics.

    Standardisation uses a scaler fitted on the *real* distribution so that
    both real and generated features live in the same coordinate system.

    Args:
        mean_prof: (N, radial_bins) radial-profile matrix.
        psd_prof:  (N, psd_bins)    PSD profile matrix.
        extra:     optional (N, K)  additional features.
        scaler:    optional (mean, std) tuple from a prior call.  When None,
                   the scaler is fitted on the data passed here (use for real
                   samples).  Pass the returned scaler when standardising
                   generated samples.

    Returns:
        (X, (mean, std)) — the standardised feature matrix and the scaler.
    """
    feats = [mean_prof, psd_prof]
    if extra is not None:
        feats.append(extra)
    X = np.concatenate(feats, axis=1).astype(np.float64)

    if scaler is None:
        feat_mean = X.mean(axis=0, keepdims=True)
        feat_std = X.std(axis=0, keepdims=True)
    else:
        feat_mean, feat_std = scaler

    X = (X - feat_mean) / (feat_std + 1e-12)
    return X.astype(np.float32), (feat_mean, feat_std)
