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

    def _make_beta_schedule(self, name, num_steps):
        if name == "linear":
            # Simple linear from 1e-4 to 0.02
            return np.linspace(1e-4, 2e-2, num_steps, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unknown beta_schedule {name}")

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

    def loss(self, model, x0, cond):
        """
        One-step diffusion training loss (epsilon MSE).
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
        eps_pred = model([x_t, t, cond], training=True)
        loss = tf.reduce_mean((noise - eps_pred) ** 2)
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

        if guidance_scale is None:
            guidance_scale = 0.0
        guidance_scale = tf.cast(guidance_scale, tf.float32)

        if guidance_scale > 0.0:
            # unconditional labels: the special null id
            cond_null = tf.fill([bsz], tf.cast(self.null_label, tf.int32))

            # One forward pass: [cond batch; uncond batch]
            x_in = tf.concat([x_t, x_t], axis=0)
            t_in = tf.concat([t, t], axis=0)
            c_in = tf.concat([cond, cond_null], axis=0)

            eps_all = model([x_in, t_in, c_in], training=False)
            eps_cond, eps_uncond = tf.split(eps_all, num_or_size_splits=2, axis=0)

            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_theta = model([x_t, t, cond], training=False)

        # DDPM mean
        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar) * eps_theta
        )
        
        posterior_var_t = self.posterior_variance[t_int]  # scalar

        if t_int > 0:
            noise = tf.random.normal(shape=tf.shape(x_t))
            x_prev = model_mean + tf.sqrt(posterior_var_t) * noise
        else:
            # t == 0: no noise
            x_prev = model_mean

        return x_prev

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