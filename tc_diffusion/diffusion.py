import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

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
        self.num_ss_classes = int(cfg["conditioning"]["num_ss_classes"])
        self.null_label = self.num_ss_classes  # must match model_unet.py
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

    def loss(self, model, x0, cond):
        """
        One-step diffusion training loss.
        x0: (B, H, W, C) in [-1, 1]
        cond: (B,) scalar for now
        """
        batch_size = tf.shape(x0)[0]
        # Uniform t from [0, num_steps-1]
        t = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.num_steps, dtype=tf.int32
        )
        noise = tf.random.normal(shape=tf.shape(x0))

        x_t = self.q_sample(x0, t, noise)
        pred = model([x_t, t, cond], training=True)

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
        cond: (B,) scalar cond per sample (placeholder for now).
        """
        bsz = tf.shape(x_t)[0]
        t = tf.fill([bsz], tf.cast(t_int, tf.int32))  # (B,)

        beta_t = self.betas[t_int]              # scalar
        alpha_t = self.alphas[t_int]            # scalar
        alpha_bar_t = self.alphas_cumprod[t_int]  # scalar

        beta_t = tf.cast(beta_t, tf.float32)
        alpha_t = tf.cast(alpha_t, tf.float32)
        alpha_bar_t = tf.cast(alpha_bar_t, tf.float32)

        sqrt_one_minus_alpha_bar = tf.sqrt(1.0 - alpha_bar_t)
        sqrt_recip_alpha_t = 1.0 / tf.sqrt(alpha_t)

        gs = 0.0 if guidance_scale is None else float(guidance_scale)

        if gs > 0.0:
            # unconditional labels: the special null id
            cond_null = tf.fill([bsz], tf.cast(self.null_label, tf.int32))

            # One forward pass: [cond batch; uncond batch]
            x_in = tf.concat([x_t, x_t], axis=0)
            t_in = tf.concat([t, t], axis=0)
            c_in = tf.concat([cond, cond_null], axis=0)

            pred_all = model([x_in, t_in, c_in], training=False)
            pred_cond, pred_uncond = tf.split(pred_all, num_or_size_splits=2, axis=0)
            pred_theta = pred_uncond + tf.cast(gs, tf.float32) * (pred_cond - pred_uncond)
        else:
            pred_theta = model([x_t, t, cond], training=False)

        if self.prediction_type == "eps":
            eps_theta = pred_theta
            x0_pred = self._predict_x0_from_eps(x_t, t_int, eps_theta)
        else:
            x0_pred = self._predict_x0_from_v(x_t, alpha_bar_t, pred_theta)
            eps_theta = self._predict_eps_from_v(x_t, alpha_bar_t, pred_theta)

        if self.dynamic_threshold:
            x0_pred, _ = self._dynamic_threshold_x0(x0_pred, p=self.dynamic_threshold_p)

        # Recompute eps consistent with stabilized x0 (important)
        # eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
        sqrt_alpha_bar = tf.sqrt(alpha_bar_t)
        eps_theta = (x_t - sqrt_alpha_bar * x0_pred) / (sqrt_one_minus_alpha_bar + 1e-8)

        # DDPM mean using stabilized eps (equivalently stabilized x0)
        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / (sqrt_one_minus_alpha_bar + 1e-8)) * eps_theta
        )
        
        posterior_var_t = self.posterior_variance[t_int]  # scalar

        if t_int > 0:
            noise = tf.random.normal(shape=tf.shape(x_t))
            x_prev = model_mean + tf.sqrt(posterior_var_t) * noise
            return x_prev

        # t == 0: return x0 prediction (NOT model_mean), clipped to expected data range.
        return tf.clip_by_value(x0_pred, -1.0, 1.0)

    def sample(self, model, batch_size, image_size, cond_value=None, show_progress=True, guidance_scale=0.0):
        """
        Generate samples starting from pure noise.

        cond_value: integer SS category index (0..5).
        """
        x_t = tf.random.normal(
            shape=(batch_size, image_size, image_size, 1), dtype=tf.float32
        )

        if cond_value is None:
            # unconditional: always use null label
            cond = tf.fill([batch_size], tf.cast(self.null_label, tf.int32))
        else:
            # conditional: fixed SS category for whole batch
            cond = tf.fill([batch_size], tf.cast(int(cond_value), tf.int32))

        t_iter = reversed(range(self.num_steps))
        if show_progress:
            t_iter = tqdm(
                t_iter,
                total=self.num_steps,
                desc="Sampling diffusion steps",
                leave=True,
            )

        for t_int in t_iter:
            x_t = self.p_sample_step(model, x_t, t_int, cond, guidance_scale=guidance_scale)

        return x_t
