# tc_diffusion/train_loop.py
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from .data import create_dataset
from .model_unet import build_unet
from .diffusion import Diffusion


def train(cfg):
    # GPU memory growth
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

    # ----- early stopping config -----
    es_cfg = cfg["training"].get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    patience_epochs = int(es_cfg.get("patience_epochs", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    best_epoch_loss = float("inf")
    best_epoch_idx = -1
    epochs_without_improvement = 0

    global_step = 0

    # epoch-wise history
    epoch_indices = []
    epoch_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        epoch_loss_sum = 0.0
        epoch_batches = 0

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
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        pbar.close()

        # compute mean loss for this epoch
        epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        epoch_indices.append(epoch + 1)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch+1} mean loss: {epoch_loss:.6f}")

        # ----- check for improvement -----
        if best_epoch_loss - epoch_loss > min_delta:
            best_epoch_loss = epoch_loss
            best_epoch_idx = epoch + 1
            epochs_without_improvement = 0

            ckpt_path = out_dir / "weights_best.weights.h5"
            model.save_weights(ckpt_path)
            print(f"  New best model (epoch {best_epoch_idx}), saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No significant improvement ({epochs_without_improvement}/{patience_epochs})"
            )

        # ----- early stopping -----
        if es_enabled and epochs_without_improvement >= patience_epochs:
            print(
                f"\n Early stopping at epoch {epoch+1} "
                f"(best epoch: {best_epoch_idx}, with loss {best_epoch_loss:.6f})"
            )
            break

    # Return epoch-wise history
    return {
        "epoch": epoch_indices,
        "epoch_loss": epoch_losses,
        "best_epoch": best_epoch_idx,
        "best_epoch_loss": best_epoch_loss,
        "stopped_early": es_enabled and epochs_without_improvement >= patience_epochs,
    }
