import tensorflow as tf
import numpy as np


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

        betas = self._make_beta_schedule(self.beta_schedule_name, self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas = tf.constant(alphas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)

    def _make_beta_schedule(self, name, num_steps):
        if name == "linear":
            # Simple linear from 1e-4 to 0.02
            return np.linspace(1e-4, 2e-2, num_steps, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unknown beta_schedule {name}")

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
        return tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * noise

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
