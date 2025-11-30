# tc_diffusion/train_loop.py
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm  # NEW

from .data import create_dataset
from .model_unet import build_unet
from .diffusion import Diffusion


def train(cfg):
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

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_epoch_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        epoch_loss_sum = 0.0
        epoch_batches = 0

        # ---- tqdm progress bar for the epoch ----
        pbar = tqdm(ds, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch, (x0, cond) in enumerate(pbar):
            with tf.GradientTape() as tape:
                loss = diffusion.loss(model, x0, cond)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            global_step += 1
            loss_value = float(loss.numpy())
            epoch_loss_sum += loss_value
            epoch_batches += 1

            if global_step % log_interval == 0:
                # update bar postfix instead of printing
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        pbar.close()

        epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"Epoch {epoch+1} mean loss: {epoch_loss:.6f}")

        # ---- save only if improved ----
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            ckpt_path = out_dir / "weights_best.weights.h5"
            model.save_weights(ckpt_path)
            print(f"  New best model, saved to {ckpt_path}")
        else:
            print("  (No improvement, not saving weights)")
