import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm


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
        self.num_steps = int(cfg["diffusion"]["num_steps"])
        self.beta_schedule_name = cfg["diffusion"]["beta_schedule"]
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

        betas = self._make_beta_schedule(self.beta_schedule_name, self.num_steps)
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

    def _predict_x0_from_eps(self, x_t, t_int, eps_theta):
        """
        Reconstruct x0 from epsilon prediction:
          x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        alpha_bar_t = tf.cast(self.alphas_cumprod[t_int], tf.float32)  # scalar
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

    def _make_beta_schedule(self, name: str, num_steps: int):
        """
        Create a beta schedule.

        Supported:
        - "linear": linear beta schedule (original DDPM)
        - "cosine": cosine alpha_bar schedule (Nichol & Dhariwal 2021)
        """
        if name == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)

        elif name == "cosine":
            # --- cosine alpha_bar schedule (Nichol & Dhariwal, 2021) ---
            s = 0.008
            steps = num_steps

            # t in [0, T]
            t = np.linspace(0, steps, steps + 1, dtype=np.float64) / steps

            alpha_bar = np.cos(((t + s) / (1.0 + s)) * np.pi / 2.0) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1

            # betas from alpha_bar
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])

            # numerical safety
            betas = np.clip(betas, 1e-8, 0.999)

            return betas.astype(np.float64)

        else:
            raise ValueError(f"Unknown beta schedule: {name}")

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
            x0_pred = self._predict_x0_from_eps(x_t, t_int, eps_theta)
        else:
            x0_pred = self._predict_x0_from_v(x_t, alpha_bar_t, pred_theta)
            eps_theta = self._predict_eps_from_v(x_t, alpha_bar_t, pred_theta)

        if self.dynamic_threshold:
            x0_pred, _ = self._dynamic_threshold_x0(x0_pred, p=self.dynamic_threshold_p)

        sqrt_alpha_bar = tf.sqrt(alpha_bar_t)
        eps_theta = (x_t - sqrt_alpha_bar * x0_pred) / (sqrt_one_minus_alpha_bar + 1e-8)
        return x0_pred, eps_theta

    def loss(self, model, x0, cond, training: bool = True, t=None, noise=None, seed=None):
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
        loss = tf.reduce_mean(mse_per_sample * weights)
        return loss

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
    ):
        """Single deterministic DPM-Solver++(2M, midpoint) step."""
        x0_pred, _ = self._predict_x0_and_eps(
            model,
            x_t,
            t_int,
            cond,
            guidance_scale=guidance_scale,
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

    def _build_sampling_timesteps(self, num_sampling_steps, *, sampler_name: str):
        if num_sampling_steps is None:
            steps = self.num_steps
        else:
            steps = int(num_sampling_steps)
        if steps <= 0 or steps > self.num_steps:
            raise ValueError(
                f"{sampler_name.upper()} num_sampling_steps must be in [1, {self.num_steps}], got {steps}."
            )
        schedule = np.round(np.linspace(0, self.num_steps - 1, steps)).astype(np.int32)
        schedule = np.unique(schedule)
        return list(schedule[::-1])

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
        ddim_eta: float = 0.0,
        return_both: bool = False,
    ):
        """
        Generate samples starting from pure noise.

        cond_value: integer SS category index (0..5), or None for unconditional.
        wind_value_kt: optional wind conditioning value. If None, uses class midpoint.
        sampler: reverse-process sampler to use. Supported: "ddpm", "ddim", "dpmpp_2m".
        num_sampling_steps: optional number of inference steps for DDIM / DPM-Solver++(2M).
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

        sampler_name = str(sampler).strip().lower()
        if sampler_name == "dpm_solverpp_2m":
            sampler_name = "dpmpp_2m"
        if sampler_name not in {"ddpm", "ddim", "dpmpp_2m"}:
            raise ValueError(
                f"Unsupported sampler '{sampler}'. Expected 'ddpm', 'ddim', or 'dpmpp_2m'."
            )

        if sampler_name == "ddpm":
            timesteps = list(range(self.num_steps - 1, -1, -1))
        else:
            timesteps = self._build_sampling_timesteps(
                num_sampling_steps,
                sampler_name=sampler_name,
            )

        t_iter = timesteps
        if show_progress:
            t_iter = tqdm(
                t_iter,
                total=len(timesteps),
                desc=f"Sampling diffusion steps ({sampler_name.upper()})",
                leave=True,
            )

        if sampler_name == "ddpm":
            for t_int in t_iter:
                x_t = self.p_sample_step(model, x_t, t_int, cond, guidance_scale=guidance_scale)
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
        else:
            prev_x0_pred = None
            prev_t_int = None
            for idx, t_int in enumerate(t_iter):
                t_prev_int = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
                x_t, prev_x0_pred = self.dpmpp_2m_step(
                    model,
                    x_t,
                    t_int,
                    t_prev_int,
                    cond,
                    guidance_scale=guidance_scale,
                    prev_x0_pred=prev_x0_pred,
                    prev_t_int=prev_t_int,
                )
                prev_t_int = t_int

        raw_final = x_t
        clipped_final = tf.clip_by_value(raw_final, -1.0, 1.0)
        if return_both:
            return {
                "raw_final": raw_final,
                "clipped_final": clipped_final,
            }
        return raw_final
