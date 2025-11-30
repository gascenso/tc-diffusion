# tc_diffusion/train_loop.py
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from .data import create_dataset
from .model_unet import build_unet
from .diffusion import Diffusion


def train(cfg):
    # Optionally control memory growth etc.
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    ds = create_dataset(cfg)
    model = build_unet(cfg)
    diffusion = Diffusion(cfg)

    lr = float(cfg["training"]["lr"])
    num_epochs = int(cfg["training"]["num_epochs"])
    log_interval = int(cfg["training"]["log_interval_steps"])

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Simple training loop (no EMA, no ckpt for now)
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch, (x0, cond) in enumerate(ds):
            with tf.GradientTape() as tape:
                loss = diffusion.loss(model, x0, cond)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            global_step += 1
            if global_step % log_interval == 0:
                print(f"  step {global_step:06d} | loss = {loss.numpy():.4f}")

        # Save simple checkpoint at end of epoch
        out_dir = Path(cfg["experiment"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"weights_epoch_{epoch+1:03d}.ckpt"
        model.save_weights(ckpt_path)
        print(f"Saved weights to {ckpt_path}")
