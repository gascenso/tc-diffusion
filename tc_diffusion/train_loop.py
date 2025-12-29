# tc_diffusion/train_loop.py
import json
import glob
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from .data import create_dataset
from .model_unet import build_unet
from .diffusion import Diffusion


def _state_path(out_dir: Path) -> Path:
    return out_dir / "run_state.json"


def _find_latest_last_weights(out_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(out_dir / "weights_last.epoch_*.weights.h5")))
    if not files:
        return None
    return Path(files[-1])


def _delete_other_last_weights(out_dir: Path, keep: Path):
    for f in glob.glob(str(out_dir / "weights_last.epoch_*.weights.h5")):
        fp = Path(f)
        if fp != keep:
            try:
                fp.unlink()
            except Exception:
                pass


def _load_state(out_dir: Path) -> dict | None:
    sp = _state_path(out_dir)
    if not sp.exists():
        return None
    with sp.open("r") as f:
        return json.load(f)


def _save_state(out_dir: Path, state: dict):
    sp = _state_path(out_dir)
    tmp = out_dir / (sp.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(sp)


def train(cfg, resume: bool = False):
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

    # ----- resume state -----
    start_epoch = 0
    best_epoch_loss = float("inf")
    best_epoch_idx = -1
    epochs_without_improvement = 0
    global_step = 0

    epoch_indices = []
    epoch_losses = []

    if resume:
        state = _load_state(out_dir)
        last_w = _find_latest_last_weights(out_dir)

        if state is None or last_w is None:
            print(f"[resume] No state/last weights found in {out_dir}. Starting fresh.")
        else:
            print(f"[resume] Loading last weights: {last_w}")
            model.load_weights(str(last_w))

            # Continue from next epoch
            start_epoch = int(state.get("last_epoch", 0))
            best_epoch_loss = float(state.get("best_epoch_loss", best_epoch_loss))
            best_epoch_idx = int(state.get("best_epoch", best_epoch_idx))
            epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            global_step = int(state.get("global_step", 0))

            # If you want continuous loss curves across resumes:
            epoch_indices = list(state.get("epoch_indices", []))
            epoch_losses = list(state.get("epoch_losses", []))

            print(
                f"[resume] start_epoch={start_epoch}, best_epoch={best_epoch_idx}, "
                f"best_loss={best_epoch_loss:.6f}, global_step={global_step}"
            )

    for epoch in range(start_epoch, num_epochs):
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

        epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        epoch_indices.append(epoch + 1)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch+1} mean loss: {epoch_loss:.6f}")

        # ----- ALWAYS save "last" -----
        last_path = out_dir / f"weights_last.epoch_{epoch+1}.weights.h5"
        model.save_weights(last_path)
        _delete_other_last_weights(out_dir, keep=last_path)
        print(f"  Saved last weights to {last_path}")

        # ----- check for improvement (best) -----
        if best_epoch_loss - epoch_loss > min_delta:
            best_epoch_loss = epoch_loss
            best_epoch_idx = epoch + 1
            epochs_without_improvement = 0

            best_path = out_dir / "weights_best.weights.h5"
            model.save_weights(best_path)
            print(f"  New best model (epoch {best_epoch_idx}), saved to {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No significant improvement ({epochs_without_improvement}/{patience_epochs})")

        # ----- persist run state (for resuming) -----
        _save_state(
            out_dir,
            {
                "last_epoch": epoch + 1,  # completed epochs
                "global_step": global_step,
                "best_epoch": best_epoch_idx,
                "best_epoch_loss": best_epoch_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_indices": epoch_indices,
                "epoch_losses": epoch_losses,
            },
        )

        # ----- early stopping -----
        if es_enabled and epochs_without_improvement >= patience_epochs:
            print(
                f"\n Early stopping at epoch {epoch+1} "
                f"(best epoch: {best_epoch_idx}, with loss {best_epoch_loss:.6f})"
            )
            break

    return {
        "epoch": epoch_indices,
        "epoch_loss": epoch_losses,
        "best_epoch": best_epoch_idx,
        "best_epoch_loss": best_epoch_loss,
        "stopped_early": es_enabled and epochs_without_improvement >= patience_epochs,
    }
