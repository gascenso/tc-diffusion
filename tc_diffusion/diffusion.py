from collections.abc import Callable

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from .sampling_guidance import (
    load_or_build_sampling_guidance_target_bank,
    resolve_sampling_guidance_cfg,
    sampling_guidance_summary,
)


_VALID_BETA_SCHEDULES = ("linear", "cosine")
_VALID_TIMESTEP_SCHEDULES = ("linear", "leading", "trailing")
_ZERO_TERMINAL_SNR_SUFFIX = "_rescaled_zero_terminal_snr"


@tf.custom_gradient
def _real_to_complex_no_warning(x):
    y = tf.complex(x, tf.zeros_like(x))

    def grad(dy):
        return tf.math.real(dy)

    return y, grad


def _resolve_beta_schedule_spec(
    name: str,
    *,
    rescale_zero_terminal_snr: bool = False,
) -> tuple[str, bool]:
    schedule_name = str(name).strip().lower()
    alias_enables_rescale = False
    if schedule_name.endswith(_ZERO_TERMINAL_SNR_SUFFIX):
        schedule_name = schedule_name[: -len(_ZERO_TERMINAL_SNR_SUFFIX)]
        alias_enables_rescale = True
    if schedule_name not in _VALID_BETA_SCHEDULES:
        supported = list(_VALID_BETA_SCHEDULES) + [
            f"{base}{_ZERO_TERMINAL_SNR_SUFFIX}" for base in _VALID_BETA_SCHEDULES
        ]
        raise ValueError(
            "Unknown beta schedule "
            f"{name!r}. Expected one of: {', '.join(supported)}."
        )
    return schedule_name, bool(rescale_zero_terminal_snr or alias_enables_rescale)


def normalize_timestep_schedule_name(name: str) -> str:
    schedule_name = str(name).strip().lower()
    if schedule_name == "linspace":
        schedule_name = "linear"
    if schedule_name not in _VALID_TIMESTEP_SCHEDULES:
        raise ValueError(
            "Unknown timestep schedule "
            f"{name!r}. Expected one of: {', '.join(_VALID_TIMESTEP_SCHEDULES)}."
        )
    return schedule_name


def resolve_sampling_timestep_schedule(cfg: dict) -> str:
    sampling_cfg = cfg.get("sampling", {})
    if isinstance(sampling_cfg, dict):
        raw = sampling_cfg.get("timestep_schedule", None)
        if raw is not None:
            return normalize_timestep_schedule_name(raw)

    eval_cfg = cfg.get("evaluation", {})
    if isinstance(eval_cfg, dict):
        raw = eval_cfg.get("timestep_schedule", None)
        if raw is not None:
            return normalize_timestep_schedule_name(raw)

    return "linear"


def _ss_class_midpoint_kt_scalar(ss_cat: int) -> float:
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


class Diffusion:
    """
    Implements the forward diffusion process and training loss for a diffusion model.

    Args:
        cfg (dict): Configuration dictionary containing diffusion parameters.

    Methods:
        q_sample: Samples from the forward process q(x_t | x_0).
        loss: Computes the training loss for one diffusion step.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_steps = int(cfg["diffusion"]["num_steps"])
        self.beta_schedule_name, self.rescale_zero_terminal_snr = _resolve_beta_schedule_spec(
            cfg["diffusion"]["beta_schedule"],
            rescale_zero_terminal_snr=bool(
                cfg["diffusion"].get("rescale_zero_terminal_snr", False)
            ),
        )
        self.default_timestep_schedule_name = resolve_sampling_timestep_schedule(cfg)
        cond_cfg = cfg.get("conditioning", {})
        self.num_ss_classes = int(cond_cfg.get("num_ss_classes", 6))
        self.null_label = self.num_ss_classes  # must match model_unet.py
        self.use_wind_speed = bool(cond_cfg.get("use_wind_speed", False))
        self.wind_min_kt = float(cond_cfg.get("wind_min_kt", 35.0))
        self.wind_max_kt = float(cond_cfg.get("wind_max_kt", 170.0))
        self.null_wind_kt = float(cond_cfg.get("wind_null_kt", 0.0))
        self.prediction_type = self._parse_prediction_type(
            str(cfg["diffusion"].get("loss_type", "eps_mse"))
        )
        min_snr_gamma_cfg = cfg["diffusion"].get("min_snr_gamma", None)
        self.min_snr_gamma = None if min_snr_gamma_cfg is None else float(min_snr_gamma_cfg)
        if self.min_snr_gamma is not None and self.min_snr_gamma <= 0.0:
            raise ValueError(
                f"diffusion.min_snr_gamma must be > 0 when set, got {self.min_snr_gamma}"
            )

        betas = self._make_beta_schedule(
            self.beta_schedule_name,
            self.num_steps,
            rescale_zero_terminal_snr=self.rescale_zero_terminal_snr,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # alpha_bar_{t-1} with alpha_bar_{-1} := 1
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]).astype(np.float32)

        # posterior variance beta_tilde
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = posterior_variance.astype(np.float32)

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas = tf.constant(alphas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # sampling stabilization (high leverage for avoiding clipped extremes)
        self.dynamic_threshold = bool(cfg["diffusion"].get("dynamic_threshold", True))
        self.dynamic_threshold_p = float(cfg["diffusion"].get("dynamic_threshold_p", 0.995))

        phys_cfg = cfg.get("physics_loss", {})
        self.physics_enabled = bool(phys_cfg.get("enabled", False))
        self.physics_lambda = float(phys_cfg.get("lambda_phys", 0.0))
        physics_lambda_warmup_steps = phys_cfg.get("lambda_phys_warmup_steps", 0)
        physics_lambda_warmup_frac = phys_cfg.get("lambda_phys_warmup_frac", 0.0)
        self.physics_radial_weight = float(phys_cfg.get("radial_weight", 1.0))
        self.physics_dav_weight = float(phys_cfg.get("dav_weight", 0.0))
        self.physics_hist_weight = float(phys_cfg.get("hist_weight", 0.0))
        self.physics_cold_weight = float(phys_cfg.get("cold_weight", 1.0))
        self.physics_grad_weight = float(phys_cfg.get("grad_weight", 1.0))
        self.physics_snr_gate_max = float(phys_cfg.get("snr_gate_max", 0.8))
        self.physics_radial_bins = int(phys_cfg.get("radial_bins", 64))
        self.physics_dav_radius_km = float(phys_cfg.get("dav_radius_km", 300.0))
        self.physics_pixel_size_km = float(phys_cfg.get("pixel_size_km", 8.0))
        self.physics_hist_bins = int(phys_cfg.get("hist_bins", 32))
        self.physics_hist_softness_k = float(phys_cfg.get("hist_softness_k", 2.5))
        self.physics_cold_threshold_k = float(phys_cfg.get("cold_threshold_k", 235.0))
        self.physics_cold_softness_k = float(phys_cfg.get("cold_softness_k", 5.0))
        self.physics_charbonnier_eps = float(phys_cfg.get("charbonnier_eps", 1e-3))
        self.physics_lambda_warmup_steps = int(physics_lambda_warmup_steps)
        self.physics_lambda_warmup_frac = float(physics_lambda_warmup_frac)
        self.bt_min_k = float(cfg["data"]["bt_min_k"])
        self.bt_max_k = float(cfg["data"]["bt_max_k"])
        self.bt_range_k = self.bt_max_k - self.bt_min_k
        self.sampling_guidance_cfg = resolve_sampling_guidance_cfg(cfg)
        self.sampling_guidance_report = sampling_guidance_summary(cfg)
        self.sampling_guidance_enabled = bool(self.sampling_guidance_cfg["enabled"])
        self.sampling_guidance_active = (
            self.sampling_guidance_enabled
            and (
                float(self.sampling_guidance_cfg["radial_weight"]) > 0.0
                or float(self.sampling_guidance_cfg["dav_weight"]) > 0.0
                or float(self.sampling_guidance_cfg["hist_weight"]) > 0.0
                or float(self.sampling_guidance_cfg["cold_weight"]) > 0.0
                or float(self.sampling_guidance_cfg["eye_weight"]) > 0.0
                or float(self.sampling_guidance_cfg["psd_weight"]) > 0.0
            )
        )
        self._sampling_guidance_target_bank = None

        if self.physics_lambda < 0.0:
            raise ValueError(
                f"physics_loss.lambda_phys must be >= 0, got {self.physics_lambda}"
            )
        if self.physics_radial_weight < 0.0:
            raise ValueError(
                f"physics_loss.radial_weight must be >= 0, got {self.physics_radial_weight}"
            )
        if self.physics_dav_weight < 0.0:
            raise ValueError(
                f"physics_loss.dav_weight must be >= 0, got {self.physics_dav_weight}"
            )
        if self.physics_hist_weight < 0.0:
            raise ValueError(
                f"physics_loss.hist_weight must be >= 0, got {self.physics_hist_weight}"
            )
        if self.physics_cold_weight < 0.0:
            raise ValueError(
                f"physics_loss.cold_weight must be >= 0, got {self.physics_cold_weight}"
            )
        if self.physics_grad_weight < 0.0:
            raise ValueError(
                f"physics_loss.grad_weight must be >= 0, got {self.physics_grad_weight}"
            )
        if self.physics_snr_gate_max < 0.0:
            raise ValueError(
                f"physics_loss.snr_gate_max must be >= 0, got {self.physics_snr_gate_max}"
            )
        if self.physics_radial_bins <= 0:
            raise ValueError(
                f"physics_loss.radial_bins must be >= 1, got {self.physics_radial_bins}"
            )
        if self.physics_dav_radius_km <= 0.0:
            raise ValueError(
                "physics_loss.dav_radius_km must be > 0, "
                f"got {self.physics_dav_radius_km}"
            )
        if self.physics_pixel_size_km <= 0.0:
            raise ValueError(
                "physics_loss.pixel_size_km must be > 0, "
                f"got {self.physics_pixel_size_km}"
            )
        if self.physics_hist_bins < 2:
            raise ValueError(
                "physics_loss.hist_bins must be >= 2, "
                f"got {self.physics_hist_bins}"
            )
        if self.physics_hist_softness_k <= 0.0:
            raise ValueError(
                "physics_loss.hist_softness_k must be > 0, "
                f"got {self.physics_hist_softness_k}"
            )
        if self.physics_cold_softness_k <= 0.0:
            raise ValueError(
                "physics_loss.cold_softness_k must be > 0, "
                f"got {self.physics_cold_softness_k}"
            )
        if self.physics_charbonnier_eps <= 0.0:
            raise ValueError(
                "physics_loss.charbonnier_eps must be > 0, "
                f"got {self.physics_charbonnier_eps}"
            )
        if self.physics_lambda_warmup_steps < 0:
            raise ValueError(
                "physics_loss.lambda_phys_warmup_steps must be >= 0, "
                f"got {self.physics_lambda_warmup_steps}"
            )
        if self.physics_lambda_warmup_frac < 0.0 or self.physics_lambda_warmup_frac > 1.0:
            raise ValueError(
                "physics_loss.lambda_phys_warmup_frac must be in [0, 1], "
                f"got {self.physics_lambda_warmup_frac}"
            )
        if self.physics_lambda_warmup_steps > 0 and self.physics_lambda_warmup_frac > 0.0:
            raise ValueError(
                "Specify only one of physics_loss.lambda_phys_warmup_steps or "
                "physics_loss.lambda_phys_warmup_frac."
            )
        if self.bt_range_k <= 0.0:
            raise ValueError(
                "data.bt_max_k must be > data.bt_min_k to support physics_loss, "
                f"got bt_min_k={self.bt_min_k}, bt_max_k={self.bt_max_k}"
            )

        if self.physics_lambda_warmup_frac > 0.0:
            train_cfg = cfg.get("training", {})
            total_train_steps = int(train_cfg.get("num_epochs", 0)) * int(
                train_cfg.get("steps_per_epoch", 0)
            )
            if total_train_steps <= 0:
                raise ValueError(
                    "physics_loss.lambda_phys_warmup_frac requires positive "
                    "training.num_epochs and training.steps_per_epoch."
                )
            self.physics_lambda_warmup_steps = max(
                1,
                int(np.ceil(total_train_steps * self.physics_lambda_warmup_frac)),
            )

        self.physics_active = (
            self.physics_enabled
            and self.physics_lambda > 0.0
            and (
                self.physics_radial_weight > 0.0
                or self.physics_dav_weight > 0.0
                or self.physics_hist_weight > 0.0
                or self.physics_cold_weight > 0.0
                or self.physics_grad_weight > 0.0
            )
        )
        if self.physics_active:
            image_size = int(cfg["data"]["image_size"])
            if image_size <= 0:
                raise ValueError(
                    f"data.image_size must be >= 1 when physics_loss is enabled, got {image_size}"
                )
            radial_bin_index, radial_bin_counts = self._build_radial_bin_lookup(
                image_size=image_size,
                radial_bins=self.physics_radial_bins,
            )
            self.physics_image_size = image_size
            self.physics_radial_bin_index = tf.constant(radial_bin_index, dtype=tf.int32)
            self.physics_radial_bin_counts = tf.constant(radial_bin_counts, dtype=tf.float32)
            dav_radial_x, dav_radial_y, dav_mask = self._build_dav_lookup(
                image_size=image_size,
                pixel_size_km=self.physics_pixel_size_km,
                radius_km=self.physics_dav_radius_km,
            )
            self.physics_dav_radial_x = tf.constant(dav_radial_x, dtype=tf.float32)
            self.physics_dav_radial_y = tf.constant(dav_radial_y, dtype=tf.float32)
            self.physics_dav_mask = tf.constant(dav_mask, dtype=tf.float32)
            hist_edges_k = np.linspace(
                self.bt_min_k,
                self.bt_max_k,
                num=self.physics_hist_bins + 1,
                dtype=np.float32,
            )
            self.physics_hist_thresholds_k = tf.constant(hist_edges_k[1:-1], dtype=tf.float32)
            self.physics_hist_bin_width_k = float(hist_edges_k[1] - hist_edges_k[0])

        if self.sampling_guidance_active:
            image_size = int(cfg["data"]["image_size"])
            if image_size <= 0:
                raise ValueError(
                    "data.image_size must be >= 1 when sampling_guidance is enabled, "
                    f"got {image_size}"
                )
            sg_cfg = self.sampling_guidance_cfg
            radial_bin_index, radial_bin_counts = self._build_radial_bin_lookup(
                image_size=image_size,
                radial_bins=int(sg_cfg["radial_bins"]),
            )
            self.sg_image_size = image_size
            self.sg_radial_bins = int(sg_cfg["radial_bins"])
            self.sg_radial_bin_index = tf.constant(radial_bin_index, dtype=tf.int32)
            self.sg_radial_bin_counts = tf.constant(radial_bin_counts, dtype=tf.float32)
            (
                sg_dav_radial_x,
                sg_dav_radial_y,
                sg_dav_mask,
                sg_dav_weight_sum,
            ) = self._build_dav_lookup_multi_center(
                image_size=image_size,
                pixel_size_km=float(sg_cfg["pixel_size_km"]),
                radius_km=float(sg_cfg["dav_radius_km"]),
                center_region_size=int(sg_cfg["dav_center_region_size"]),
            )
            self.sg_dav_radial_x = tf.constant(sg_dav_radial_x, dtype=tf.float32)
            self.sg_dav_radial_y = tf.constant(sg_dav_radial_y, dtype=tf.float32)
            self.sg_dav_mask = tf.constant(sg_dav_mask, dtype=tf.float32)
            self.sg_dav_weight_sum = tf.constant(sg_dav_weight_sum, dtype=tf.float32)
            hist_edges_k = np.linspace(
                self.bt_min_k,
                self.bt_max_k,
                num=int(sg_cfg["hist_bins"]) + 1,
                dtype=np.float32,
            )
            self.sg_hist_thresholds_k = tf.constant(hist_edges_k[1:-1], dtype=tf.float32)
            psd_bin_index, psd_bin_counts = self._build_frequency_radial_bin_lookup(
                image_size=image_size,
                psd_bins=int(sg_cfg["psd_bins"]),
            )
            self.sg_psd_bins = int(sg_cfg["psd_bins"])
            self.sg_psd_bin_index = tf.constant(psd_bin_index, dtype=tf.int32)
            self.sg_psd_bin_counts = tf.constant(psd_bin_counts, dtype=tf.float32)

    def _default_wind_from_ss_tensor(self, ss_cat):
        ss_cat = tf.cast(ss_cat, tf.int32)
        if self.num_ss_classes == 6:
            table = tf.constant([49.0, 73.0, 89.0, 104.0, 124.5, 145.0], dtype=tf.float32)
            idx = tf.clip_by_value(ss_cat, 0, 5)
            wind = tf.gather(table, idx)
        else:
            idx = tf.cast(tf.clip_by_value(ss_cat, 0, self.num_ss_classes - 1), tf.float32)
            span = tf.maximum(self.wind_max_kt - self.wind_min_kt, 1e-6)
            wind = self.wind_min_kt + ((idx + 0.5) / float(self.num_ss_classes)) * span
        return tf.where(
            tf.equal(ss_cat, self.null_label),
            tf.fill(tf.shape(wind), tf.cast(self.null_wind_kt, tf.float32)),
            tf.cast(wind, tf.float32),
        )

    def _prepare_condition_tensors(self, cond, batch_size):
        if isinstance(cond, dict):
            ss_cat = tf.cast(cond["ss_cat"], tf.int32)
            if "wind_kt" in cond:
                wind_kt = tf.cast(cond["wind_kt"], tf.float32)
            else:
                wind_kt = self._default_wind_from_ss_tensor(ss_cat)
        elif isinstance(cond, (tuple, list)) and len(cond) == 2:
            ss_cat = tf.cast(cond[0], tf.int32)
            wind_kt = tf.cast(cond[1], tf.float32)
        else:
            ss_cat = tf.cast(cond, tf.int32)
            wind_kt = self._default_wind_from_ss_tensor(ss_cat)

        ss_cat = tf.reshape(ss_cat, [batch_size])
        wind_kt = tf.reshape(wind_kt, [batch_size])
        return ss_cat, wind_kt

    def _parse_prediction_type(self, loss_type: str) -> str:
        lt = str(loss_type).strip().lower()
        if lt in {"eps", "eps_mse", "epsilon", "epsilon_mse"}:
            return "eps"
        if lt in {"v", "v_mse", "vpred", "v_pred", "v_prediction"}:
            return "v"
        raise ValueError(
            "diffusion.loss_type must be one of: "
            "eps_mse|eps|epsilon_mse|epsilon|v_mse|v|vpred|v_pred|v_prediction. "
            f"Got: {loss_type}"
        )

    def _predict_x0_from_eps(self, x_t, alpha_bar_t, eps_theta):
        """
        Reconstruct x0 from epsilon prediction:
          x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        alpha_bar_t = tf.cast(alpha_bar_t, tf.float32)
        sqrt_ab = tf.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = tf.sqrt(1.0 - alpha_bar_t)
        return (x_t - sqrt_one_minus_ab * eps_theta) / (sqrt_ab + 1e-8)

    def _predict_x0_from_v(self, x_t, alpha_bar_t, v_theta):
        """
        Reconstruct x0 from v-prediction:
          x0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
        alpha_bar_t can be scalar or [B,1,1,1].
        """
        sqrt_ab = tf.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = tf.sqrt(1.0 - alpha_bar_t)
        return sqrt_ab * x_t - sqrt_one_minus_ab * v_theta

    def _predict_eps_from_v(self, x_t, alpha_bar_t, v_theta):
        """
        Reconstruct eps from v-prediction:
          eps = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v
        alpha_bar_t can be scalar or [B,1,1,1].
        """
        sqrt_ab = tf.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = tf.sqrt(1.0 - alpha_bar_t)
        return sqrt_one_minus_ab * x_t + sqrt_ab * v_theta

    def _dynamic_threshold_x0(self, x0, p=0.995, eps=1e-8):
        """
        Dynamic thresholding (per-sample).
        x0: [B,H,W,C] in normalized space.

        Returns:
          x0_thr: thresholded x0 in [-1,1] (approximately)
          s: per-sample scale [B,1,1,1]
        """
        b = tf.shape(x0)[0]
        x = tf.reshape(x0, [b, -1])
        absx = tf.abs(x)

        # percentile index
        n = tf.shape(absx)[1]
        k = tf.cast(tf.round(p * tf.cast(n - 1, tf.float32)), tf.int32)
        k = tf.clip_by_value(k, 0, n - 1)

        absx_sorted = tf.sort(absx, axis=1)
        s = tf.gather(absx_sorted, k, axis=1)  # [B]

        # Avoid amplifying small values
        s = tf.maximum(s, 1.0)
        s = tf.reshape(s, [-1, 1, 1, 1])

        x0_thr = tf.clip_by_value(x0, -s, s) / (s + eps)
        return x0_thr, s

    def _build_radial_bin_lookup(self, image_size: int, radial_bins: int):
        yy, xx = np.mgrid[0:image_size, 0:image_size]
        cy = (image_size - 1) / 2.0
        cx = (image_size - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius_norm = radius / np.maximum(np.max(radius), 1e-12)
        radial_bin_index = np.floor(radius_norm * radial_bins).astype(np.int32)
        radial_bin_index = np.clip(radial_bin_index, 0, radial_bins - 1).reshape(-1)
        radial_bin_counts = np.bincount(radial_bin_index, minlength=radial_bins).astype(np.float32)
        radial_bin_counts = np.maximum(radial_bin_counts, 1.0)
        return radial_bin_index, radial_bin_counts

    def _build_frequency_radial_bin_lookup(self, image_size: int, psd_bins: int):
        yy, xx = np.mgrid[0:image_size, 0:image_size]
        cy = (image_size - 1) / 2.0
        cx = (image_size - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius_norm = radius / np.maximum(np.max(radius), 1e-12)
        psd_bin_index = np.floor(radius_norm * int(psd_bins)).astype(np.int32)
        psd_bin_index = np.clip(psd_bin_index, 0, int(psd_bins) - 1).reshape(-1)
        psd_bin_counts = np.bincount(psd_bin_index, minlength=int(psd_bins)).astype(np.float32)
        psd_bin_counts = np.maximum(psd_bin_counts, 1.0)
        return psd_bin_index, psd_bin_counts

    def _build_dav_lookup(self, image_size: int, pixel_size_km: float, radius_km: float):
        yy, xx = np.mgrid[0:image_size, 0:image_size]
        cy = (image_size - 1) / 2.0
        cx = (image_size - 1) / 2.0
        dy = yy - cy
        dx = xx - cx
        radius_px = np.sqrt(dx * dx + dy * dy)
        radius_px_safe = np.maximum(radius_px, 1e-12)
        radial_x = (dx / radius_px_safe).reshape(-1).astype(np.float32)
        radial_y = (dy / radius_px_safe).reshape(-1).astype(np.float32)
        radius_km_grid = (radius_px * float(pixel_size_km)).reshape(-1)
        mask = np.logical_and(radius_km_grid > 0.0, radius_km_grid <= float(radius_km)).astype(np.float32)
        if not np.any(mask > 0.0):
            raise ValueError(
                "physics_loss.dav_radius_km selects no pixels. "
                f"Got image_size={image_size}, pixel_size_km={pixel_size_km}, "
                f"dav_radius_km={radius_km}."
            )
        return radial_x, radial_y, mask

    def _build_dav_lookup_multi_center(
        self,
        image_size: int,
        pixel_size_km: float,
        radius_km: float,
        center_region_size: int,
    ):
        yy, xx = np.mgrid[0:image_size, 0:image_size]
        cy0 = (image_size - 1) / 2.0
        cx0 = (image_size - 1) / 2.0
        half = int(center_region_size) // 2

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
                radius_km_grid = radius_px * float(pixel_size_km)
                mask = np.logical_and(
                    radius_km_grid > 0.0,
                    radius_km_grid <= float(radius_km),
                )
                if not np.any(mask):
                    raise ValueError(
                        "sampling_guidance.dav_radius_km selects no pixels. "
                        f"Got image_size={image_size}, pixel_size_km={pixel_size_km}, "
                        f"dav_radius_km={radius_km}, center_region_size={center_region_size}."
                    )
                mask_rows.append(mask.reshape(-1).astype(np.float32))

        radial_x = np.stack(radial_x_rows, axis=0)
        radial_y = np.stack(radial_y_rows, axis=0)
        mask = np.stack(mask_rows, axis=0)
        weight_sum = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0).astype(np.float32)
        return radial_x, radial_y, mask, weight_sum

    def _charbonnier(self, diff):
        diff = tf.cast(diff, tf.float32)
        eps = tf.cast(self.physics_charbonnier_eps, tf.float32)
        return tf.sqrt(tf.square(diff) + eps * eps) - eps

    def _denorm_bt_k(self, x):
        x = tf.cast(x, tf.float32)
        bt01 = (x + 1.0) * 0.5
        return bt01 * tf.cast(self.bt_range_k, tf.float32) + tf.cast(self.bt_min_k, tf.float32)

    def _radial_mean_profile(self, x):
        x = tf.cast(x, tf.float32)
        field = tf.reduce_mean(x, axis=-1)  # [B, H, W]
        flat = tf.reshape(field, [tf.shape(field)[0], -1])  # [B, HW]
        flat_t = tf.transpose(flat, [1, 0])  # [HW, B]
        radial_sum = tf.math.unsorted_segment_sum(
            flat_t,
            self.physics_radial_bin_index,
            self.physics_radial_bins,
        )  # [R, B]
        radial_mean = radial_sum / tf.reshape(self.physics_radial_bin_counts, [-1, 1])
        return tf.transpose(radial_mean, [1, 0])  # [B, R]

    def _soft_cold_cloud_fraction(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        field = tf.reduce_mean(x_bt_k, axis=-1)  # [B, H, W]
        threshold = tf.cast(self.physics_cold_threshold_k, tf.float32)
        softness = tf.cast(self.physics_cold_softness_k, tf.float32)
        cold_score = tf.nn.sigmoid((threshold - field) / softness)
        return tf.reduce_mean(cold_score, axis=[1, 2])  # [B]

    def _sobel_gradient_components(self, x):
        x = tf.cast(x, tf.float32)
        field = tf.reduce_mean(x, axis=-1, keepdims=True)
        sobel = tf.image.sobel_edges(field)  # [B, H, W, 1, 2]
        grad_y = sobel[..., 0]
        grad_x = sobel[..., 1]
        return grad_x, grad_y

    def _sobel_gradient_magnitude(self, x):
        grad_x, grad_y = self._sobel_gradient_components(x)
        return tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-8)  # [B, H, W, 1]

    def _dav_deviation_angle_deg(self, grad_x, grad_y, radial_x, radial_y, valid_mask):
        grad_x = tf.cast(grad_x, tf.float32)
        grad_y = tf.cast(grad_y, tf.float32)
        radial_x = tf.cast(radial_x, tf.float32)
        radial_y = tf.cast(radial_y, tf.float32)
        valid_mask = tf.cast(valid_mask, tf.bool)

        # Normalize gradient vectors before comparing directions. For invalid
        # pixels (outside the DAV mask or with near-zero gradients), swap in
        # the radial unit vector so atan2 never sees the undefined (0, 0) case.
        grad_norm = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))
        safe_grad_norm = tf.maximum(grad_norm, 1e-6)
        unit_grad_x = grad_x / safe_grad_norm
        unit_grad_y = grad_y / safe_grad_norm
        safe_grad_x = tf.where(valid_mask, unit_grad_x, radial_x)
        safe_grad_y = tf.where(valid_mask, unit_grad_y, radial_y)

        dot = safe_grad_x * radial_x + safe_grad_y * radial_y
        dot = tf.clip_by_value(dot, -1.0, 1.0)
        cross = safe_grad_x * radial_y - safe_grad_y * radial_x
        return tf.atan2(tf.abs(cross), dot) * (180.0 / np.pi)

    def _dav_per_sample(self, x):
        grad_x, grad_y = self._sobel_gradient_components(x)
        grad_x = tf.reshape(grad_x, [tf.shape(grad_x)[0], -1])
        grad_y = tf.reshape(grad_y, [tf.shape(grad_y)[0], -1])

        radial_x = tf.reshape(self.physics_dav_radial_x, [1, -1])
        radial_y = tf.reshape(self.physics_dav_radial_y, [1, -1])
        base_mask = tf.reshape(self.physics_dav_mask, [1, -1])
        grad_valid = tf.sqrt(tf.square(grad_x) + tf.square(grad_y)) > 1e-6
        valid_mask = tf.logical_and(base_mask > 0.0, grad_valid)
        mask = tf.cast(valid_mask, tf.float32)
        dev_angle_deg = self._dav_deviation_angle_deg(
            grad_x,
            grad_y,
            radial_x,
            radial_y,
            valid_mask,
        )

        weight_sum = tf.maximum(tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)
        mean_angle = tf.reduce_sum(dev_angle_deg * mask, axis=1, keepdims=True) / weight_sum
        return tf.reduce_sum(tf.square(dev_angle_deg - mean_angle) * mask, axis=1) / tf.squeeze(weight_sum, axis=1)

    def _soft_bt_cdf(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        flat = tf.reshape(x_bt_k, [tf.shape(x_bt_k)[0], -1, 1])
        thresholds = tf.reshape(self.physics_hist_thresholds_k, [1, 1, -1])
        softness = tf.cast(self.physics_hist_softness_k, tf.float32)
        return tf.reduce_mean(tf.nn.sigmoid((thresholds - flat) / softness), axis=1)

    def _sg_radial_mean_profile_bt_k(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        field = tf.reduce_mean(x_bt_k, axis=-1)
        flat = tf.reshape(field, [tf.shape(field)[0], -1])
        flat_t = tf.transpose(flat, [1, 0])
        radial_sum = tf.math.unsorted_segment_sum(
            flat_t,
            self.sg_radial_bin_index,
            self.sg_radial_bins,
        )
        radial_mean = radial_sum / tf.reshape(self.sg_radial_bin_counts, [-1, 1])
        return tf.transpose(radial_mean, [1, 0])

    def _sg_dav_per_sample_bt_k(self, x_bt_k):
        grad_x, grad_y = self._sobel_gradient_components(x_bt_k)
        grad_x = tf.reshape(grad_x, [tf.shape(grad_x)[0], 1, -1])
        grad_y = tf.reshape(grad_y, [tf.shape(grad_y)[0], 1, -1])

        num_centers = int(self.sg_dav_radial_x.shape[0])
        num_pixels = int(self.sg_dav_radial_x.shape[1])
        radial_x = tf.reshape(self.sg_dav_radial_x, [1, num_centers, num_pixels])
        radial_y = tf.reshape(self.sg_dav_radial_y, [1, num_centers, num_pixels])
        base_mask = tf.reshape(self.sg_dav_mask, [1, num_centers, num_pixels])
        grad_valid = tf.sqrt(tf.square(grad_x) + tf.square(grad_y)) > 1e-6
        valid_mask = tf.logical_and(base_mask > 0.0, grad_valid)
        mask = tf.cast(valid_mask, tf.float32)
        dev_angle_deg = self._dav_deviation_angle_deg(
            grad_x,
            grad_y,
            radial_x,
            radial_y,
            valid_mask,
        )

        weight_sum = tf.maximum(tf.reduce_sum(mask, axis=2, keepdims=True), 1.0)
        mean_angle = tf.reduce_sum(dev_angle_deg * mask, axis=2, keepdims=True) / weight_sum
        dav_by_center = tf.reduce_sum(
            tf.square(dev_angle_deg - mean_angle) * mask,
            axis=2,
        ) / tf.squeeze(weight_sum, axis=2)
        return tf.reduce_mean(dav_by_center, axis=1)

    def _sg_soft_bt_cdf(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        flat = tf.reshape(x_bt_k, [tf.shape(x_bt_k)[0], -1, 1])
        thresholds = tf.reshape(self.sg_hist_thresholds_k, [1, 1, -1])
        softness = tf.cast(self.sampling_guidance_cfg["hist_softness_k"], tf.float32)
        return tf.reduce_mean(tf.nn.sigmoid((thresholds - flat) / softness), axis=1)

    def _sg_soft_cold_cloud_fraction(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        field = tf.reduce_mean(x_bt_k, axis=-1)
        threshold = tf.cast(self.sampling_guidance_cfg["cold_threshold_k"], tf.float32)
        softness = tf.cast(self.sampling_guidance_cfg["cold_softness_k"], tf.float32)
        return tf.reduce_mean(tf.nn.sigmoid((threshold - field) / softness), axis=[1, 2])

    def _sg_eye_contrast_from_radial(self, radial_profile_k):
        radial_profile_k = tf.cast(radial_profile_k, tf.float32)
        radial_bins = int(self.sg_radial_bins)
        inner_frac = float(self.sampling_guidance_cfg["eye_inner_frac"])
        ring_frac = float(self.sampling_guidance_cfg["eye_ring_frac"])
        inner_bins = max(1, int(radial_bins * inner_frac))
        ring_center = int(radial_bins * ring_frac)
        ring_window = max(2, int(radial_bins * 0.05))
        lo = max(inner_bins, ring_center - ring_window)
        lo = min(lo, max(radial_bins - 1, 0))
        hi = min(radial_bins, max(lo + 1, ring_center + ring_window))
        eye_mean = tf.reduce_mean(radial_profile_k[:, :inner_bins], axis=1)
        ring_mean = tf.reduce_mean(radial_profile_k[:, lo:hi], axis=1)
        return eye_mean - ring_mean

    def _sg_psd_profile_bt_k(self, x_bt_k):
        x_bt_k = tf.cast(x_bt_k, tf.float32)
        field = tf.reduce_mean(x_bt_k, axis=-1)
        field = field - tf.reduce_mean(field, axis=[1, 2], keepdims=True)
        fft = tf.signal.fft2d(_real_to_complex_no_warning(field))
        power = tf.math.real(fft * tf.math.conj(fft))
        power = tf.signal.fftshift(power, axes=(1, 2))
        flat = tf.reshape(power, [tf.shape(power)[0], -1])
        flat_t = tf.transpose(flat, [1, 0])
        psd_sum = tf.math.unsorted_segment_sum(
            flat_t,
            self.sg_psd_bin_index,
            self.sg_psd_bins,
        )
        psd_mean = psd_sum / tf.reshape(self.sg_psd_bin_counts, [-1, 1])
        psd_mean = tf.transpose(psd_mean, [1, 0])
        return tf.math.log(psd_mean + 1e-12) / tf.math.log(tf.constant(10.0, tf.float32))

    def _sampling_guidance_band_penalty(self, current, target_mean, target_std, sigma_floor):
        current = tf.cast(current, tf.float32)
        target_mean = tf.cast(target_mean, tf.float32)
        target_std = tf.cast(target_std, tf.float32)
        sigma_floor = tf.cast(sigma_floor, tf.float32)
        band_width = tf.cast(self.sampling_guidance_cfg["band_width_sigma"], tf.float32)
        scale = tf.maximum(target_std, sigma_floor)
        excess = tf.nn.relu(tf.abs(current - target_mean) - band_width * scale)
        standardized_excess = excess / (scale + 1e-6)
        return self._charbonnier(standardized_excess)

    def _sampling_guidance_pull_penalty(self, current, target, target_std, sigma_floor):
        current = tf.cast(current, tf.float32)
        target = tf.cast(target, tf.float32)
        target_std = tf.cast(target_std, tf.float32)
        sigma_floor = tf.cast(sigma_floor, tf.float32)
        scale = tf.maximum(target_std, sigma_floor)
        standardized_gap = (current - target) / (scale + 1e-6)
        return self._charbonnier(standardized_gap)

    def _sampling_guidance_feature_penalty(self, current, targets, name: str, sigma_floor):
        penalty = self._sampling_guidance_band_penalty(
            current,
            targets[f"{name}_mean"],
            targets[f"{name}_std"],
            sigma_floor=sigma_floor,
        )
        pull_weight = float(self.sampling_guidance_cfg["target_pull_weight"])
        if pull_weight > 0.0:
            pull_penalty = self._sampling_guidance_pull_penalty(
                current,
                targets[f"{name}_target"],
                targets[f"{name}_std"],
                sigma_floor=sigma_floor,
            )
            penalty = penalty + tf.cast(pull_weight, tf.float32) * pull_penalty
        return penalty

    def _hist_cdf_loss_per_sample(self, x0_true, x0_hat):
        cdf_true = self._soft_bt_cdf(self._denorm_bt_k(x0_true))
        cdf_hat = self._soft_bt_cdf(self._denorm_bt_k(x0_hat))
        cdf_gap = self._charbonnier(cdf_hat - cdf_true)
        return tf.reduce_sum(cdf_gap, axis=1) * tf.cast(self.physics_hist_bin_width_k, tf.float32)

    def _physics_gate(self, alpha_bar_t):
        alpha_bar_t = tf.reshape(tf.cast(alpha_bar_t, tf.float32), [-1])
        snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
        gate = snr / (snr + 1.0)
        return tf.clip_by_value(gate, 0.0, tf.cast(self.physics_snr_gate_max, tf.float32))

    def _effective_physics_lambda(self, global_step=None):
        physics_lambda = tf.cast(self.physics_lambda, tf.float32)
        if self.physics_lambda_warmup_steps <= 0 or global_step is None:
            return physics_lambda

        step = tf.cast(global_step, tf.float32)
        warmup_steps = tf.cast(self.physics_lambda_warmup_steps, tf.float32)
        progress = tf.clip_by_value((step + 1.0) / warmup_steps, 0.0, 1.0)
        return physics_lambda * progress

    def _physics_loss_per_sample(self, x0_true, x0_hat):
        x0_true = tf.cast(x0_true, tf.float32)
        x0_hat = tf.cast(x0_hat, tf.float32)
        loss = tf.zeros([tf.shape(x0_hat)[0]], dtype=tf.float32)

        if self.physics_radial_weight > 0.0:
            radial_true = self._radial_mean_profile(x0_true)
            radial_hat = self._radial_mean_profile(x0_hat)
            radial_loss = tf.reduce_mean(self._charbonnier(radial_hat - radial_true), axis=1)
            loss = loss + tf.cast(self.physics_radial_weight, tf.float32) * radial_loss

        if self.physics_dav_weight > 0.0:
            dav_true = self._dav_per_sample(x0_true)
            dav_hat = self._dav_per_sample(x0_hat)
            dav_loss = self._charbonnier(dav_hat - dav_true)
            loss = loss + tf.cast(self.physics_dav_weight, tf.float32) * dav_loss

        if self.physics_hist_weight > 0.0:
            hist_loss = self._hist_cdf_loss_per_sample(x0_true, x0_hat)
            loss = loss + tf.cast(self.physics_hist_weight, tf.float32) * hist_loss

        if self.physics_cold_weight > 0.0:
            cold_true = self._soft_cold_cloud_fraction(self._denorm_bt_k(x0_true))
            cold_hat = self._soft_cold_cloud_fraction(self._denorm_bt_k(x0_hat))
            cold_loss = self._charbonnier(cold_hat - cold_true)
            loss = loss + tf.cast(self.physics_cold_weight, tf.float32) * cold_loss

        if self.physics_grad_weight > 0.0:
            grad_true = self._sobel_gradient_magnitude(x0_true)
            grad_hat = self._sobel_gradient_magnitude(x0_hat)
            grad_loss = tf.reduce_mean(self._charbonnier(grad_hat - grad_true), axis=[1, 2, 3])
            loss = loss + tf.cast(self.physics_grad_weight, tf.float32) * grad_loss

        return loss

    def _ensure_sampling_guidance_target_bank(self, show_progress: bool = False):
        if not self.sampling_guidance_active:
            return None
        if self._sampling_guidance_target_bank is None:
            self._sampling_guidance_target_bank = load_or_build_sampling_guidance_target_bank(
                self.cfg,
                show_progress=show_progress,
            )
        return self._sampling_guidance_target_bank

    def _prepare_sampling_guidance_targets(self, cond, batch_size, *, show_progress: bool = False):
        if not self.sampling_guidance_active:
            return None

        bank = self._ensure_sampling_guidance_target_bank(show_progress=show_progress)
        if bank is None:
            return None

        ss_cat, wind_kt = self._prepare_condition_tensors(cond, batch_size)
        ss_cat_np = np.asarray(ss_cat.numpy(), dtype=np.int32).reshape(-1)
        wind_kt_np = np.asarray(wind_kt.numpy(), dtype=np.float32).reshape(-1)
        if ss_cat_np.size == 0:
            return None
        if np.all(ss_cat_np == int(self.null_label)):
            return None
        if np.any(ss_cat_np == int(self.null_label)):
            raise ValueError(
                "sampling_guidance does not support batches that mix null-conditioning "
                "tokens with conditioned samples."
            )
        if not np.all(ss_cat_np == ss_cat_np[0]):
            raise ValueError(
                "sampling_guidance currently expects the batch to share a single SS class."
            )

        class_id = int(ss_cat_np[0])
        class_targets = bank.class_targets.get(class_id)
        if class_targets is None:
            raise ValueError(
                f"Sampling-guidance target bank has no entries for class {class_id}."
            )

        num_refs = int(class_targets.wind_kt.shape[0])
        if num_refs <= 0:
            raise ValueError(
                f"Sampling-guidance target bank has no usable reference winds for class {class_id}."
            )
        k = min(int(self.sampling_guidance_cfg["neighbor_k"]), num_refs)
        distances = np.abs(wind_kt_np[:, None] - class_targets.wind_kt[None, :])
        nn_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(ss_cat_np.shape[0], dtype=np.int64)
        target_mode = str(self.sampling_guidance_cfg["target_mode"])
        if target_mode == "neighbor_mean":
            prototype_idx = None
        elif target_mode == "nearest":
            nearest_rank = np.argmin(distances[row_idx[:, None], nn_idx], axis=1)
            prototype_idx = nn_idx[row_idx, nearest_rank]
        else:
            random_rank = np.random.randint(0, k, size=ss_cat_np.shape[0])
            prototype_idx = nn_idx[row_idx, random_rank]

        targets = {
            "class_id": class_id,
            "neighbor_k": k,
            "target_mode": target_mode,
        }
        def add_feature_targets(name: str, values: np.ndarray) -> None:
            neighbors = values[nn_idx]
            mean = neighbors.mean(axis=1).astype(np.float32, copy=False)
            std = neighbors.std(axis=1).astype(np.float32, copy=False)
            if prototype_idx is None:
                target = mean
            else:
                target = values[prototype_idx].astype(np.float32, copy=False)
            targets[f"{name}_mean"] = tf.constant(mean)
            targets[f"{name}_std"] = tf.constant(std)
            targets[f"{name}_target"] = tf.constant(target)

        if float(self.sampling_guidance_cfg["radial_weight"]) > 0.0:
            add_feature_targets("radial", class_targets.radial_profiles_k)
        if float(self.sampling_guidance_cfg["dav_weight"]) > 0.0:
            add_feature_targets("dav", class_targets.dav_deg2)
        if float(self.sampling_guidance_cfg["hist_weight"]) > 0.0:
            add_feature_targets("hist", class_targets.hist_cdf)
        if float(self.sampling_guidance_cfg["cold_weight"]) > 0.0:
            add_feature_targets("cold", class_targets.cold_fraction)
        if float(self.sampling_guidance_cfg["eye_weight"]) > 0.0:
            add_feature_targets("eye", class_targets.eye_contrast_k)
        if float(self.sampling_guidance_cfg["psd_weight"]) > 0.0:
            add_feature_targets("psd", class_targets.psd_profiles_log10)
        return targets

    def _sampling_guidance_energy_per_sample(self, x0_pred, targets):
        x0_pred = tf.cast(x0_pred, tf.float32)
        x_bt_k = self._denorm_bt_k(x0_pred)
        loss = tf.zeros([tf.shape(x0_pred)[0]], dtype=tf.float32)
        radial_cur = None

        if float(self.sampling_guidance_cfg["radial_weight"]) > 0.0:
            radial_cur = self._sg_radial_mean_profile_bt_k(x_bt_k)
            radial_penalty = self._sampling_guidance_feature_penalty(
                radial_cur,
                targets,
                "radial",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_radial_k"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["radial_weight"],
                tf.float32,
            ) * tf.reduce_mean(radial_penalty, axis=1)

        if float(self.sampling_guidance_cfg["dav_weight"]) > 0.0:
            dav_cur = self._sg_dav_per_sample_bt_k(x_bt_k)
            dav_penalty = self._sampling_guidance_feature_penalty(
                dav_cur,
                targets,
                "dav",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_dav_deg2"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["dav_weight"],
                tf.float32,
            ) * dav_penalty

        if float(self.sampling_guidance_cfg["hist_weight"]) > 0.0:
            hist_cur = self._sg_soft_bt_cdf(x_bt_k)
            hist_penalty = self._sampling_guidance_feature_penalty(
                hist_cur,
                targets,
                "hist",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_hist_cdf"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["hist_weight"],
                tf.float32,
            ) * tf.reduce_mean(hist_penalty, axis=1)

        if float(self.sampling_guidance_cfg["cold_weight"]) > 0.0:
            cold_cur = self._sg_soft_cold_cloud_fraction(x_bt_k)
            cold_penalty = self._sampling_guidance_feature_penalty(
                cold_cur,
                targets,
                "cold",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_cold_fraction"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["cold_weight"],
                tf.float32,
            ) * cold_penalty

        if float(self.sampling_guidance_cfg["eye_weight"]) > 0.0:
            if radial_cur is None:
                radial_cur = self._sg_radial_mean_profile_bt_k(x_bt_k)
            eye_cur = self._sg_eye_contrast_from_radial(radial_cur)
            eye_penalty = self._sampling_guidance_feature_penalty(
                eye_cur,
                targets,
                "eye",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_eye_k"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["eye_weight"],
                tf.float32,
            ) * eye_penalty

        if float(self.sampling_guidance_cfg["psd_weight"]) > 0.0:
            psd_cur = self._sg_psd_profile_bt_k(x_bt_k)
            psd_penalty = self._sampling_guidance_feature_penalty(
                psd_cur,
                targets,
                "psd",
                sigma_floor=float(self.sampling_guidance_cfg["sigma_floor_psd_log10"]),
            )
            loss = loss + tf.cast(
                self.sampling_guidance_cfg["psd_weight"],
                tf.float32,
            ) * tf.reduce_mean(psd_penalty, axis=1)

        return loss

    def _apply_sampling_guidance_to_x0(self, x0_pred, targets, guidance_step_scale: float):
        if targets is None or float(guidance_step_scale) <= 0.0:
            return x0_pred

        x0_guided = tf.cast(x0_pred, tf.float32)
        inner_steps = max(1, int(self.sampling_guidance_cfg["inner_steps"]))
        per_inner_step = float(guidance_step_scale) / float(inner_steps)
        for _ in range(inner_steps):
            with tf.GradientTape() as tape:
                tape.watch(x0_guided)
                energy_per_sample = self._sampling_guidance_energy_per_sample(x0_guided, targets)
                energy = tf.reduce_mean(energy_per_sample)

            grad = tape.gradient(energy, x0_guided)
            if grad is None:
                return x0_pred

            grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
            grad_rms = tf.sqrt(
                tf.reduce_mean(tf.square(grad), axis=[1, 2, 3], keepdims=True) + 1e-8
            )
            direction = grad / grad_rms
            x0_guided = x0_guided - tf.cast(per_inner_step, tf.float32) * direction
            if self.dynamic_threshold:
                x0_guided, _ = self._dynamic_threshold_x0(x0_guided, p=self.dynamic_threshold_p)
        return tf.cast(x0_guided, x0_pred.dtype)

    def _sampling_guidance_step_scale(self, step_index: int, total_steps: int) -> float:
        if not self.sampling_guidance_active or total_steps <= 0:
            return 0.0

        base_step_size = float(self.sampling_guidance_cfg["step_size"])
        if base_step_size <= 0.0:
            return 0.0

        start_frac = float(self.sampling_guidance_cfg["guide_start_step_frac"])
        start_index = int(np.floor(total_steps * start_frac))
        start_index = min(max(start_index, 0), max(total_steps - 1, 0))
        stop_frac = float(self.sampling_guidance_cfg["guide_stop_step_frac"])
        stop_index = int(np.ceil(total_steps * stop_frac)) - 1
        stop_index = min(max(stop_index, start_index), max(total_steps - 1, 0))
        if step_index < start_index or step_index > stop_index:
            return 0.0

        guided_steps = max(stop_index - start_index + 1, 1)
        progress = float(step_index - start_index + 1) / float(guided_steps)
        if str(self.sampling_guidance_cfg["schedule"]) == "cosine":
            ramp = 0.5 - 0.5 * np.cos(np.pi * progress)
        else:
            ramp = progress
        return base_step_size * ramp

    def get_sampling_guidance_report(self) -> dict:
        return dict(self.sampling_guidance_report)

    def _betas_for_alpha_bar(self, num_steps: int, alpha_bar_fn):
        t = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float64)
        alpha_bar = np.asarray(alpha_bar_fn(t), dtype=np.float64)
        if alpha_bar.shape != (num_steps + 1,):
            raise ValueError(
                f"alpha_bar_fn must return shape ({num_steps + 1},), got {alpha_bar.shape}."
            )
        alpha_bar0 = float(alpha_bar[0])
        if not np.isfinite(alpha_bar0) or alpha_bar0 <= 0.0:
            raise ValueError(f"alpha_bar[0] must be finite and > 0, got {alpha_bar0}.")
        alpha_bar = np.maximum(alpha_bar / alpha_bar0, 0.0)
        betas = 1.0 - (alpha_bar[1:] / np.maximum(alpha_bar[:-1], 1e-12))
        return betas.astype(np.float64, copy=False)

    def _rescale_betas_zero_terminal_snr(self, betas):
        betas = np.asarray(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas, axis=0)
        alpha_bar_sqrt = np.sqrt(alpha_bar)

        start = float(alpha_bar_sqrt[0])
        end = float(alpha_bar_sqrt[-1])
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("Cannot rescale betas with non-finite alpha_bar values.")
        if start <= end:
            raise ValueError(
                "Zero-terminal-SNR rescaling requires alpha_bar_sqrt[0] > alpha_bar_sqrt[-1]."
            )

        alpha_bar_sqrt = alpha_bar_sqrt - end
        alpha_bar_sqrt = alpha_bar_sqrt * (start / (start - end))

        alpha_bar_rescaled = np.square(alpha_bar_sqrt)
        alphas_rescaled = np.empty_like(alpha_bar_rescaled)
        alphas_rescaled[0] = alpha_bar_rescaled[0]
        alphas_rescaled[1:] = alpha_bar_rescaled[1:] / np.maximum(
            alpha_bar_rescaled[:-1], 1e-12
        )
        return 1.0 - alphas_rescaled

    def _validate_betas(self, betas, *, num_steps: int, schedule_name: str):
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape != (num_steps,):
            raise ValueError(
                f"{schedule_name} beta schedule must have shape ({num_steps},), got {betas.shape}."
            )
        if not np.all(np.isfinite(betas)):
            raise ValueError(f"{schedule_name} beta schedule contains non-finite values.")
        if np.any(betas <= 0.0) or np.any(betas >= 1.0):
            raise ValueError(
                f"{schedule_name} beta schedule must lie strictly inside (0, 1)."
            )

        alpha_bar = np.cumprod(1.0 - betas, axis=0)
        if not np.all(np.isfinite(alpha_bar)):
            raise ValueError(f"{schedule_name} alpha_bar contains non-finite values.")
        if np.any(alpha_bar <= 0.0) or np.any(alpha_bar > 1.0):
            raise ValueError(
                f"{schedule_name} alpha_bar must stay inside (0, 1], got "
                f"[{alpha_bar.min():.3e}, {alpha_bar.max():.3e}]."
            )
        if np.any(np.diff(alpha_bar) >= 0.0):
            raise ValueError(f"{schedule_name} alpha_bar must be strictly decreasing.")

    def _make_beta_schedule(
        self,
        name: str,
        num_steps: int,
        *,
        rescale_zero_terminal_snr: bool = False,
    ):
        """
        Create a beta schedule.

        Supported:
        - "linear": linear beta schedule (original DDPM)
        - "cosine": cosine alpha_bar schedule (Nichol & Dhariwal 2021)
        """
        if num_steps <= 0:
            raise ValueError(f"diffusion.num_steps must be >= 1, got {num_steps}.")

        schedule_name, rescale_zero_terminal_snr = _resolve_beta_schedule_spec(
            name,
            rescale_zero_terminal_snr=rescale_zero_terminal_snr,
        )

        if schedule_name == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
        elif schedule_name == "cosine":
            s = 0.008
            betas = self._betas_for_alpha_bar(
                num_steps,
                lambda t: np.cos(((t + s) / (1.0 + s)) * np.pi / 2.0) ** 2,
            )
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_name}")

        if rescale_zero_terminal_snr:
            betas = self._rescale_betas_zero_terminal_snr(betas)

        betas = np.clip(betas, 1e-8, 0.999).astype(np.float64, copy=False)
        self._validate_betas(
            betas,
            num_steps=num_steps,
            schedule_name=(
                f"{schedule_name}{_ZERO_TERMINAL_SNR_SUFFIX}"
                if rescale_zero_terminal_snr
                else schedule_name
            ),
        )
        return betas

    # --- for training ---
    def q_sample(self, x0, t, noise):
        """
        Forward process: q(x_t | x_0)
        x0: (B, H, W, C)
        t:  (B,) integer timesteps in [0, num_steps-1]
        noise: (B, H, W, C)
        """
        # Gather alpha_bar_t for each sample in batch
        alpha_bar_t = tf.gather(self.alphas_cumprod, t)  # (B,)
        alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1, 1))
        x_t = tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * noise
        return x_t

    def _min_snr_weights(self, t):
        """
        Min-SNR loss weights per sample.
        - eps prediction: w = min(snr, gamma) / snr
        - v prediction:   w = min(snr, gamma) / (snr + 1)
        """
        if self.min_snr_gamma is None:
            return tf.ones_like(tf.cast(t, tf.float32))

        alpha_bar_t = tf.gather(self.alphas_cumprod, t)  # (B,)
        snr = alpha_bar_t / tf.maximum(1.0 - alpha_bar_t, 1e-8)
        gamma = tf.cast(self.min_snr_gamma, tf.float32)
        snr_cap = tf.minimum(snr, gamma)

        if self.prediction_type == "eps":
            return snr_cap / tf.maximum(snr, 1e-8)
        return snr_cap / (snr + 1.0)

    def _seed_with_offset(self, seed, offset: int):
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        return tf.stack([seed[0], seed[1] + tf.cast(offset, tf.int32)])

    def _alpha_sigma_lambda(self, t_int: int):
        alpha_bar_t = tf.cast(self.alphas_cumprod[t_int], tf.float32)
        alpha_t = tf.sqrt(alpha_bar_t)
        sigma_t = tf.sqrt(1.0 - alpha_bar_t)
        lambda_t = tf.math.log(alpha_t + 1e-12) - tf.math.log(sigma_t + 1e-12)
        return alpha_t, sigma_t, lambda_t

    def _predict_x0_and_eps(self, model, x_t, t_int, cond, guidance_scale=0.0):
        """Run the denoiser once and return a stabilized x0 prediction plus eps."""
        bsz = tf.shape(x_t)[0]
        t = tf.fill([bsz], tf.cast(t_int, tf.int32))
        cond_ss, cond_wind = self._prepare_condition_tensors(cond, bsz)

        alpha_bar_t = tf.cast(self.alphas_cumprod[t_int], tf.float32)
        sqrt_one_minus_alpha_bar = tf.sqrt(1.0 - alpha_bar_t)
        gs = 0.0 if guidance_scale is None else float(guidance_scale)

        if gs > 0.0:
            cond_null_ss = tf.fill([bsz], tf.cast(self.null_label, tf.int32))
            cond_null_wind = tf.fill([bsz], tf.cast(self.null_wind_kt, tf.float32))

            x_in = tf.concat([x_t, x_t], axis=0)
            t_in = tf.concat([t, t], axis=0)
            ss_in = tf.concat([cond_ss, cond_null_ss], axis=0)
            wind_in = tf.concat([cond_wind, cond_null_wind], axis=0)

            pred_all = model([x_in, t_in, ss_in, wind_in], training=False)
            pred_cond, pred_uncond = tf.split(pred_all, num_or_size_splits=2, axis=0)
            pred_theta = pred_uncond + tf.cast(gs, tf.float32) * (pred_cond - pred_uncond)
        else:
            pred_theta = model([x_t, t, cond_ss, cond_wind], training=False)

        if self.prediction_type == "eps":
            eps_theta = pred_theta
            x0_pred = self._predict_x0_from_eps(x_t, alpha_bar_t, eps_theta)
        else:
            x0_pred = self._predict_x0_from_v(x_t, alpha_bar_t, pred_theta)
            eps_theta = self._predict_eps_from_v(x_t, alpha_bar_t, pred_theta)

        if self.dynamic_threshold:
            x0_pred, _ = self._dynamic_threshold_x0(x0_pred, p=self.dynamic_threshold_p)

        sqrt_alpha_bar = tf.sqrt(alpha_bar_t)
        eps_theta = (x_t - sqrt_alpha_bar * x0_pred) / (sqrt_one_minus_alpha_bar + 1e-8)
        return x0_pred, eps_theta

    def loss(self, model, x0, cond, training: bool = True, t=None, noise=None, seed=None, global_step=None):
        """
        One-step diffusion training loss.
        x0: (B, H, W, C) in [-1, 1]
        cond: int tensor (ss cat) or dict with {"ss_cat", "wind_kt"}
        training: passed to the model call; set False during validation so
                  CFG conditioning dropout is disabled and BN uses stored stats.
        t: optional int32 tensor of shape (B,) with explicit diffusion timesteps.
        noise: optional tensor with the same shape as x0 containing explicit noise.
        seed: optional stateless RNG seed of shape (2,) used when t/noise are not supplied.
        """
        batch_size = tf.shape(x0)[0]
        if t is None:
            if seed is None:
                # Uniform t from [0, num_steps-1]
                t = tf.random.uniform(
                    shape=(batch_size,), minval=0, maxval=self.num_steps, dtype=tf.int32
                )
            else:
                t = tf.random.stateless_uniform(
                    shape=(batch_size,),
                    seed=self._seed_with_offset(seed, 0),
                    minval=0,
                    maxval=self.num_steps,
                    dtype=tf.int32,
                )
        else:
            t = tf.convert_to_tensor(t, dtype=tf.int32)
            tf.debugging.assert_rank(t, 1, message="Diffusion.loss expects t to have shape (B,)")
            tf.debugging.assert_equal(
                tf.shape(t)[0],
                batch_size,
                message="Diffusion.loss expects len(t) to match the batch size.",
            )
            tf.debugging.assert_greater_equal(
                t,
                tf.zeros_like(t),
                message="Diffusion.loss expects all timesteps to be >= 0.",
            )
            tf.debugging.assert_less(
                t,
                tf.fill(tf.shape(t), tf.cast(self.num_steps, tf.int32)),
                message="Diffusion.loss expects all timesteps to be < num_steps.",
            )

        if noise is None:
            if seed is None:
                noise = tf.random.normal(shape=tf.shape(x0), dtype=x0.dtype)
            else:
                noise = tf.random.stateless_normal(
                    shape=tf.shape(x0),
                    seed=self._seed_with_offset(seed, 1),
                    dtype=x0.dtype,
                )
        else:
            noise = tf.convert_to_tensor(noise, dtype=x0.dtype)
            tf.debugging.assert_equal(
                tf.shape(noise),
                tf.shape(x0),
                message="Diffusion.loss expects noise to have the same shape as x0.",
            )

        x_t = self.q_sample(x0, t, noise)
        ss_cat, wind_kt = self._prepare_condition_tensors(cond, batch_size)
        pred = model([x_t, t, ss_cat, wind_kt], training=training)

        alpha_bar_t = tf.gather(self.alphas_cumprod, t)  # (B,)
        alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1, 1))
        sqrt_ab = tf.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = tf.sqrt(1.0 - alpha_bar_t)

        if self.prediction_type == "eps":
            target = noise
        else:
            target = sqrt_ab * noise - sqrt_one_minus_ab * x0

        mse_per_sample = tf.reduce_mean((pred - target) ** 2, axis=[1, 2, 3])  # (B,)
        weights = self._min_snr_weights(t)  # (B,)
        base_loss = tf.reduce_mean(mse_per_sample * weights)

        if not self.physics_active:
            return base_loss

        x_t_f32 = tf.cast(x_t, tf.float32)
        pred_f32 = tf.cast(pred, tf.float32)
        x0_f32 = tf.cast(x0, tf.float32)
        alpha_bar_t_f32 = tf.cast(alpha_bar_t, tf.float32)

        if self.prediction_type == "eps":
            x0_hat = self._predict_x0_from_eps(x_t_f32, alpha_bar_t_f32, pred_f32)
        else:
            x0_hat = self._predict_x0_from_v(x_t_f32, alpha_bar_t_f32, pred_f32)

        phys_loss_per_sample = self._physics_loss_per_sample(x0_f32, x0_hat)
        gate = self._physics_gate(alpha_bar_t_f32)
        physics_loss = tf.reduce_mean(gate * phys_loss_per_sample)
        physics_lambda = self._effective_physics_lambda(global_step if training else None)
        return base_loss + physics_lambda * physics_loss

    # --- for sampling ---
    def p_sample_step(self, model, x_t, t_int, cond, guidance_scale=0.0):
        """
        Single reverse step p_theta(x_{t-1} | x_t).

        t_int: Python int in [0, T-1], same for whole batch.
        cond: int tensor (ss cat) or dict with {"ss_cat", "wind_kt"}.
        """
        beta_t = self.betas[t_int]              # scalar
        alpha_t = self.alphas[t_int]            # scalar
        alpha_bar_t = self.alphas_cumprod[t_int]  # scalar

        beta_t = tf.cast(beta_t, tf.float32)
        alpha_t = tf.cast(alpha_t, tf.float32)
        alpha_bar_t = tf.cast(alpha_bar_t, tf.float32)

        sqrt_one_minus_alpha_bar = tf.sqrt(1.0 - alpha_bar_t)
        sqrt_recip_alpha_t = 1.0 / tf.sqrt(alpha_t)
        x0_pred, eps_theta = self._predict_x0_and_eps(
            model,
            x_t,
            t_int,
            cond,
            guidance_scale=guidance_scale,
        )

        # DDPM mean using stabilized eps (equivalently stabilized x0)
        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / (sqrt_one_minus_alpha_bar + 1e-8)) * eps_theta
        )
        
        posterior_var_t = self.posterior_variance[t_int]  # scalar

        if t_int > 0:
            noise = tf.random.normal(shape=tf.shape(x_t))
            x_prev = model_mean + tf.sqrt(posterior_var_t) * noise
            return x_prev

        # t == 0: return the stabilized x0 prediction without hard clipping so
        # evaluation can inspect out-of-range behavior honestly.
        return x0_pred

    def ddim_step(self, model, x_t, t_int, t_prev_int, cond, guidance_scale=0.0, eta=0.0):
        """Single DDIM reverse step x_t -> x_{t_prev}."""
        alpha_bar_t = tf.cast(self.alphas_cumprod[t_int], tf.float32)
        x0_pred, eps_theta = self._predict_x0_and_eps(
            model,
            x_t,
            t_int,
            cond,
            guidance_scale=guidance_scale,
        )

        if t_prev_int < 0:
            return x0_pred

        alpha_bar_prev = tf.cast(self.alphas_cumprod[t_prev_int], tf.float32)
        eta = tf.cast(eta, tf.float32)
        sigma_t = eta * tf.sqrt(
            tf.maximum((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), 0.0)
        ) * tf.sqrt(
            tf.maximum(1.0 - (alpha_bar_t / alpha_bar_prev), 0.0)
        )

        pred_dir = tf.sqrt(tf.maximum(1.0 - alpha_bar_prev - sigma_t ** 2, 0.0)) * eps_theta
        x_prev = tf.sqrt(alpha_bar_prev) * x0_pred + pred_dir

        if float(eta) > 0.0:
            noise = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)
            x_prev = x_prev + sigma_t * noise
        return x_prev

    def dpmpp_2m_step(
        self,
        model,
        x_t,
        t_int,
        t_prev_int,
        cond,
        guidance_scale=0.0,
        prev_x0_pred=None,
        prev_t_int=None,
        sampling_guidance_targets=None,
        sampling_guidance_step_scale: float = 0.0,
    ):
        """Single deterministic DPM-Solver++(2M, midpoint) step."""
        x0_pred, _ = self._predict_x0_and_eps(
            model,
            x_t,
            t_int,
            cond,
            guidance_scale=guidance_scale,
        )
        x0_pred = self._apply_sampling_guidance_to_x0(
            x0_pred,
            sampling_guidance_targets,
            guidance_step_scale=sampling_guidance_step_scale,
        )

        if t_prev_int < 0:
            return x0_pred, x0_pred

        alpha_t, sigma_t, lambda_t = self._alpha_sigma_lambda(int(t_prev_int))
        _, sigma_s0, lambda_s0 = self._alpha_sigma_lambda(int(t_int))
        h = lambda_t - lambda_s0

        expm1_neg_h = tf.exp(-h) - 1.0

        if prev_x0_pred is None or prev_t_int is None:
            x_prev = (sigma_t / sigma_s0) * x_t - alpha_t * expm1_neg_h * x0_pred
            return x_prev, x0_pred

        _, _, lambda_s1 = self._alpha_sigma_lambda(int(prev_t_int))
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / tf.maximum(h, 1e-12)
        d0 = x0_pred
        d1 = (x0_pred - prev_x0_pred) / tf.maximum(r0, 1e-12)
        x_prev = (
            (sigma_t / sigma_s0) * x_t
            - alpha_t * expm1_neg_h * d0
            - 0.5 * alpha_t * expm1_neg_h * d1
        )
        return x_prev, x0_pred

    def _validate_sampling_timesteps(
        self,
        timesteps,
        *,
        sampler_name: str,
        schedule_name: str,
        expected_steps: int,
    ):
        timesteps = np.asarray(timesteps, dtype=np.int32)
        if timesteps.shape != (expected_steps,):
            raise ValueError(
                f"{sampler_name.upper()} {schedule_name} timestep schedule must have "
                f"shape ({expected_steps},), got {timesteps.shape}."
            )
        if np.any(timesteps < 0) or np.any(timesteps >= self.num_steps):
            raise ValueError(
                f"{sampler_name.upper()} {schedule_name} timestep schedule produced "
                f"indices outside [0, {self.num_steps - 1}]."
            )
        if timesteps.size > 1 and np.any(np.diff(timesteps) >= 0):
            raise ValueError(
                f"{sampler_name.upper()} {schedule_name} timestep schedule must be "
                "strictly decreasing."
            )
        if timesteps.size > 1 and len(np.unique(timesteps)) != timesteps.size:
            raise ValueError(
                f"{sampler_name.upper()} {schedule_name} timestep schedule contains duplicates."
            )
        return list(timesteps.tolist())

    def _build_sampling_timesteps(
        self,
        num_sampling_steps,
        *,
        sampler_name: str,
        timestep_schedule: str | None = None,
    ):
        if num_sampling_steps is None:
            steps = self.num_steps
        else:
            steps = int(num_sampling_steps)
        if steps <= 0 or steps > self.num_steps:
            raise ValueError(
                f"{sampler_name.upper()} num_sampling_steps must be in [1, {self.num_steps}], got {steps}."
            )

        schedule_name = normalize_timestep_schedule_name(
            self.default_timestep_schedule_name if timestep_schedule is None else timestep_schedule
        )
        if steps == self.num_steps:
            return list(range(self.num_steps - 1, -1, -1))
        if steps == 1:
            return [self.num_steps - 1]

        if schedule_name == "linear":
            schedule = np.round(
                np.linspace(0, self.num_steps - 1, steps, dtype=np.float64)
            ).astype(np.int32)[::-1]
        elif schedule_name == "leading":
            step_ratio = self.num_steps // steps
            schedule = (
                np.arange(0, steps, dtype=np.int32) * np.int32(step_ratio)
            )[::-1]
        else:
            step_ratio = float(self.num_steps) / float(steps)
            schedule = np.round(
                np.arange(steps, 0, -1, dtype=np.float64) * step_ratio
            ).astype(np.int32) - 1

        return self._validate_sampling_timesteps(
            schedule,
            sampler_name=sampler_name,
            schedule_name=schedule_name,
            expected_steps=steps,
        )

    def sample(
        self,
        model,
        batch_size,
        image_size,
        cond_value=None,
        wind_value_kt=None,
        show_progress=True,
        guidance_scale=0.0,
        sampler: str = "dpmpp_2m",
        num_sampling_steps: int | None = None,
        timestep_schedule: str | None = None,
        ddim_eta: float = 0.0,
        return_both: bool = False,
        progress_callback: Callable[[int], None] | None = None,
    ):
        """
        Generate samples starting from pure noise.

        cond_value: integer SS category index (0..5), or None for unconditional.
        wind_value_kt: optional wind conditioning value. If None, uses class midpoint.
        sampler: reverse-process sampler to use. Supported: "ddpm", "ddim", "dpmpp_2m".
        num_sampling_steps: optional number of inference steps for DDIM / DPM-Solver++(2M).
        timestep_schedule: optional reduced-step timestep spacing. Supported:
                           "linear", "leading", "trailing".
        ddim_eta: DDIM stochasticity. 0.0 is deterministic DDIM.
        return_both: if True, return both the post-threshold raw final sample and
                     a hard-clipped sibling for diagnostic use.
        """
        x_t = tf.random.normal(
            shape=(batch_size, image_size, image_size, 1), dtype=tf.float32
        )

        if cond_value is None:
            cond = {
                "ss_cat": tf.fill([batch_size], tf.cast(self.null_label, tf.int32)),
                "wind_kt": tf.fill([batch_size], tf.cast(self.null_wind_kt, tf.float32)),
            }
        else:
            cls = int(cond_value)
            if wind_value_kt is None:
                # Single scalar: broadcast to whole batch
                wind_kt_t = tf.fill(
                    [batch_size],
                    tf.cast(_ss_class_midpoint_kt_scalar(cls), tf.float32),
                )
            elif np.ndim(wind_value_kt) == 0:
                # Scalar numpy value or Python float
                wind_kt_t = tf.fill(
                    [batch_size], tf.cast(float(wind_value_kt), tf.float32)
                )
            else:
                # Per-sample array: shape (batch_size,) — enables matched conditioning
                wind_kt_t = tf.cast(wind_value_kt, tf.float32)
            cond = {
                "ss_cat": tf.fill([batch_size], tf.cast(cls, tf.int32)),
                "wind_kt": wind_kt_t,
            }

        timesteps = self.get_sampling_timesteps(
            sampler=sampler,
            num_sampling_steps=num_sampling_steps,
            timestep_schedule=timestep_schedule,
        )
        sampler_name = str(sampler).strip().lower()
        if sampler_name == "dpm_solverpp_2m":
            sampler_name = "dpmpp_2m"
        if self.sampling_guidance_active and sampler_name != "dpmpp_2m":
            raise ValueError(
                "sampling_guidance is currently supported only with sampler='dpmpp_2m'."
            )

        t_iter = timesteps
        use_local_progress = bool(show_progress and progress_callback is None)
        if use_local_progress:
            t_iter = tqdm(
                t_iter,
                total=len(timesteps),
                desc=f"Sampling diffusion steps ({sampler_name.upper()})",
                leave=True,
            )

        if sampler_name == "ddpm":
            for t_int in t_iter:
                x_t = self.p_sample_step(model, x_t, t_int, cond, guidance_scale=guidance_scale)
                if progress_callback is not None:
                    progress_callback(1)
        elif sampler_name == "ddim":
            for idx, t_int in enumerate(t_iter):
                t_prev_int = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
                x_t = self.ddim_step(
                    model,
                    x_t,
                    t_int,
                    t_prev_int,
                    cond,
                    guidance_scale=guidance_scale,
                    eta=ddim_eta,
                )
                if progress_callback is not None:
                    progress_callback(1)
        else:
            sampling_guidance_targets = self._prepare_sampling_guidance_targets(
                cond,
                batch_size,
                show_progress=bool(show_progress),
            )
            prev_x0_pred = None
            prev_t_int = None
            for idx, t_int in enumerate(t_iter):
                t_prev_int = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
                sampling_guidance_step_scale = self._sampling_guidance_step_scale(
                    idx,
                    len(timesteps),
                )
                x_t, prev_x0_pred = self.dpmpp_2m_step(
                    model,
                    x_t,
                    t_int,
                    t_prev_int,
                    cond,
                    guidance_scale=guidance_scale,
                    prev_x0_pred=prev_x0_pred,
                    prev_t_int=prev_t_int,
                    sampling_guidance_targets=sampling_guidance_targets,
                    sampling_guidance_step_scale=sampling_guidance_step_scale,
                )
                prev_t_int = t_int
                if progress_callback is not None:
                    progress_callback(1)

        raw_final = x_t
        clipped_final = tf.clip_by_value(raw_final, -1.0, 1.0)
        if return_both:
            return {
                "raw_final": raw_final,
                "clipped_final": clipped_final,
            }
        return raw_final

    def get_sampling_timesteps(
        self,
        *,
        sampler: str = "dpmpp_2m",
        num_sampling_steps: int | None = None,
        timestep_schedule: str | None = None,
    ) -> list[int]:
        sampler_name = str(sampler).strip().lower()
        if sampler_name == "dpm_solverpp_2m":
            sampler_name = "dpmpp_2m"
        if sampler_name not in {"ddpm", "ddim", "dpmpp_2m"}:
            raise ValueError(
                f"Unsupported sampler '{sampler}'. Expected 'ddpm', 'ddim', or 'dpmpp_2m'."
            )

        if sampler_name == "ddpm":
            return list(range(self.num_steps - 1, -1, -1))

        return self._build_sampling_timesteps(
            num_sampling_steps,
            sampler_name=sampler_name,
            timestep_schedule=timestep_schedule,
        )
