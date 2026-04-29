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


def pairwise_squared_distances_block(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Squared Euclidean distances for one block pair."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected rank-2 feature blocks, got {x.shape} and {y.shape}.")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimensions differ: {x.shape[1]} vs {y.shape[1]}.")

    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def nearest_neighbor_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    block_size: int = 256,
) -> np.ndarray:
    """
    Euclidean distance from each query row to its nearest reference row.

    The computation is blockwise to avoid materialising large full distance
    matrices for full-test and train-reference evaluations.
    """
    distances, _indices = nearest_neighbor_distances_and_indices(
        query,
        reference,
        block_size=block_size,
    )
    return distances


def nearest_neighbor_distances_and_indices(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean nearest-neighbor distance and reference-row index for each query row.

    This is the same blockwise computation as nearest_neighbor_distances, with
    the extra index output needed for qualitative nearest-neighbor inspection.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError(f"Expected rank-2 feature matrices, got {query.shape} and {reference.shape}.")
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if reference.shape[0] == 0:
        raise ValueError("nearest_neighbor_distances_and_indices requires at least one reference row.")
    if query.shape[1] != reference.shape[1]:
        raise ValueError(f"Feature dimensions differ: {query.shape[1]} vs {reference.shape[1]}.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    out_d2 = np.full((query.shape[0],), np.inf, dtype=np.float64)
    out_idx = np.full((query.shape[0],), -1, dtype=np.int64)
    for i0 in range(0, query.shape[0], block_size):
        i1 = min(i0 + block_size, query.shape[0])
        qi = query[i0:i1]
        best_d2 = np.full((i1 - i0,), np.inf, dtype=np.float64)
        best_idx = np.full((i1 - i0,), -1, dtype=np.int64)
        for j0 in range(0, reference.shape[0], block_size):
            j1 = min(j0 + block_size, reference.shape[0])
            d2 = pairwise_squared_distances_block(qi, reference[j0:j1])
            local_idx = np.argmin(d2, axis=1)
            local_d2 = d2[np.arange(i1 - i0), local_idx]
            improve = local_d2 < best_d2
            best_d2[improve] = local_d2[improve]
            best_idx[improve] = j0 + local_idx[improve]
        out_d2[i0:i1] = best_d2
        out_idx[i0:i1] = best_idx
    return np.sqrt(out_d2).astype(np.float32), out_idx


def kth_neighbor_distances_within_set(
    x: np.ndarray,
    *,
    k: int = 3,
    block_size: int = 256,
) -> np.ndarray:
    """
    Distance from each row to its kth nearest *other* row in the same set.

    This gives each real sample a local manifold radius for coverage. For very
    small sets, k is reduced to the largest valid neighbor rank.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 feature matrix, got {x.shape}.")
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)

    k_eff = min(max(int(k), 1), n - 1)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    out = np.zeros((n,), dtype=np.float64)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        d2_blocks = []
        xi = x[i0:i1]
        for j0 in range(0, n, block_size):
            j1 = min(j0 + block_size, n)
            d2 = pairwise_squared_distances_block(xi, x[j0:j1])
            if i0 == j0:
                local = np.arange(i1 - i0)
                d2[local, local] = np.inf
            d2_blocks.append(d2)
        d2_all = np.concatenate(d2_blocks, axis=1)
        kth = np.partition(d2_all, kth=k_eff - 1, axis=1)[:, k_eff - 1]
        out[i0:i1] = np.sqrt(kth)
    return out.astype(np.float32)


def mean_pairwise_distance(
    x: np.ndarray,
    *,
    block_size: int = 256,
) -> float:
    """Mean Euclidean distance over all unordered pairs in one feature set."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 feature matrix, got {x.shape}.")
    n = int(x.shape[0])
    if n < 2:
        return 0.0

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    total = 0.0
    count = 0
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        xi = x[i0:i1]
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            d2 = pairwise_squared_distances_block(xi, x[j0:j1])
            d = np.sqrt(d2)
            if i0 == j0:
                tri = np.triu_indices(i1 - i0, k=1)
                total += float(np.sum(d[tri]))
                count += int(tri[0].shape[0])
            else:
                total += float(np.sum(d))
                count += int(d.size)
    return float(total / max(count, 1))


def summarize_distances(values: np.ndarray) -> Dict[str, float]:
    """Compact summary for nearest-neighbor and pairwise distance distributions."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty distance distribution.")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def coverage_from_real_radii(
    *,
    real_to_gen_nn: np.ndarray,
    real_radii: np.ndarray,
) -> Dict[str, float]:
    """
    Fraction of real samples covered by generated samples.

    A real sample is covered when its nearest generated neighbor lies within
    that real sample's local real-manifold radius.
    """
    nn = np.asarray(real_to_gen_nn, dtype=np.float64).reshape(-1)
    radii = np.asarray(real_radii, dtype=np.float64).reshape(-1)
    if nn.shape != radii.shape:
        raise ValueError(f"real_to_gen_nn and real_radii shape mismatch: {nn.shape} vs {radii.shape}.")
    if nn.size == 0:
        raise ValueError("Cannot compute coverage from an empty reference set.")

    covered = nn <= radii
    return {
        "value": float(np.mean(covered)),
        "covered_count": int(np.sum(covered)),
        "reference_count": int(nn.size),
        "mean_real_radius": float(np.mean(radii)),
        "median_real_radius": float(np.median(radii)),
        "real_to_generated_nn": summarize_distances(nn),
        "real_radius": summarize_distances(radii),
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


def weighted_mean_and_cov(
    X: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    *,
    covariance_eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted empirical mean/covariance for FID-style Gaussian summaries.

    When sample_weights is None, this matches the usual unbiased sample
    covariance (normalisation by N-1). When weights are supplied, they are
    normalised to sum to 1 and a weighted Bessel correction is applied via
    1 - sum(w_i^2), which reduces to N-1 for uniform weights.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be rank-2 [N,D], got shape {X.shape}")
    n, d = X.shape
    if n <= 0:
        raise ValueError("X must contain at least one sample.")

    if sample_weights is None:
        mu = np.mean(X, axis=0)
        if n >= 2:
            xc = X - mu
            cov = (xc.T @ xc) / float(n - 1)
        else:
            cov = np.zeros((d, d), dtype=np.float64)
    else:
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(
                f"sample_weights must have length {n} to match X, got {w.shape[0]}"
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weights must be finite.")
        if np.any(w < 0.0):
            raise ValueError("sample_weights must be non-negative.")
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            raise ValueError("sample_weights must sum to a positive value.")
        w = w / wsum
        mu = np.sum(X * w[:, None], axis=0)
        xc = X - mu
        denom = 1.0 - float(np.sum(w * w))
        if denom > 1.0e-12:
            cov = ((xc * w[:, None]).T @ xc) / denom
        else:
            cov = np.zeros((d, d), dtype=np.float64)

    cov = 0.5 * (cov + cov.T)
    eps = float(max(covariance_eps, 0.0))
    if eps > 0.0:
        cov = cov + np.eye(d, dtype=np.float64) * eps
    return mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False)


def frechet_distance_from_stats(
    mu_real: np.ndarray,
    cov_real: np.ndarray,
    mu_gen: np.ndarray,
    cov_gen: np.ndarray,
) -> float:
    """
    Fréchet distance between two Gaussian summaries.

    This is the standard FID formula, implemented via PSD eigendecomposition
    rather than scipy.linalg.sqrtm to keep the dependency footprint small.
    """
    mu_real = np.asarray(mu_real, dtype=np.float64).reshape(-1)
    mu_gen = np.asarray(mu_gen, dtype=np.float64).reshape(-1)
    cov_real = np.asarray(cov_real, dtype=np.float64)
    cov_gen = np.asarray(cov_gen, dtype=np.float64)

    if mu_real.shape != mu_gen.shape:
        raise ValueError(
            f"Mean vectors must have identical shape, got {mu_real.shape} vs {mu_gen.shape}"
        )
    if cov_real.shape != cov_gen.shape:
        raise ValueError(
            f"Covariance matrices must have identical shape, got {cov_real.shape} vs {cov_gen.shape}"
        )
    if cov_real.shape[0] != cov_real.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got {cov_real.shape}")
    if cov_real.shape[0] != mu_real.shape[0]:
        raise ValueError(
            "Mean/covariance dimension mismatch: "
            f"mean has dim {mu_real.shape[0]}, covariance has shape {cov_real.shape}"
        )

    cov_real = 0.5 * (cov_real + cov_real.T)
    cov_gen = 0.5 * (cov_gen + cov_gen.T)
    diff = mu_real - mu_gen

    sqrt_cov_real = _psd_matrix_sqrt(cov_real)
    middle = sqrt_cov_real @ cov_gen @ sqrt_cov_real
    middle = 0.5 * (middle + middle.T)
    sqrt_middle = _psd_matrix_sqrt(middle)
    fd = float(diff @ diff + np.trace(cov_real + cov_gen - 2.0 * sqrt_middle))
    return float(max(fd, 0.0))


def _psd_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_eigvals = np.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.T


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
