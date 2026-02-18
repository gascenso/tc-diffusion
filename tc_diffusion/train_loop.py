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
from .evaluation.evaluator import TCEvaluator

class EMA:
    """Exponential Moving Average (EMA) over model weights.

    Updates after each optimizer step:
        ema = decay * ema + (1 - decay) * w
    """

    def __init__(self, model: tf.keras.Model, decay: float):
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.model = model
        self.decay = tf.constant(float(decay), dtype=tf.float32)

        # Shadow variables mirror model.weights (trainable + non-trainable)
        self.shadow_vars = [
            tf.Variable(w, trainable=False, dtype=w.dtype, name=f"ema/{w.name.split(':')[0]}")
            for w in model.weights
        ]

    def update(self):
        d = self.decay
        one_minus_d = 1.0 - d
        for s, w in zip(self.shadow_vars, self.model.weights):
            s.assign(d * s + one_minus_d * w)

    def copy_to_model(self):
        for s, w in zip(self.shadow_vars, self.model.weights):
            w.assign(s)

    def copy_from_model(self):
        for s, w in zip(self.shadow_vars, self.model.weights):
            s.assign(w)

    def get_model_weights_snapshot(self):
        return [w.numpy() for w in self.model.weights]

    def restore_model_weights_snapshot(self, snapshot):
        for arr, w in zip(snapshot, self.model.weights):
            w.assign(arr)

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

def _find_latest_ema_last_weights(out_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(out_dir / "weights_ema_last.epoch_*.weights.h5")))
    if not files:
        return None
    return Path(files[-1])

def _delete_other_ema_last_weights(out_dir: Path, keep: Path):
    for f in glob.glob(str(out_dir / "weights_ema_last.epoch_*.weights.h5")):
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

def evaluate_loss(diffusion, model, ds_val, val_steps: int | None = None):
    """Compute mean diffusion loss.

    If val_steps is None or <=0, iterate over the *entire* ds_val.
    This is the recommended mode for finite, deterministic validation sets.
    """
    full_pass = (val_steps is None) or (int(val_steps) <= 0)
    loss_sum = 0.0
    n = 0
    for batch, (x0, cond) in enumerate(ds_val):
        if (not full_pass) and batch >= int(val_steps):
            break
        loss = diffusion.loss(model, x0, cond)
        loss_sum += float(loss.numpy())
        n += 1
    return loss_sum / max(1, n)

def train(cfg, resume: bool = False):
    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # --- build train/val datasets ---
    data_cfg = cfg.setdefault("data", {})

    # Preserve eval_mode if present
    orig_eval_mode = data_cfg.get("eval_mode", "full")

    ds_train = create_dataset(cfg, split="train")

    # Validation: build two deterministic finite sets:
    #   1) full pass over the val split (micro average)
    #   2) fixed balanced subset (tail-sensitive but stable)
    data_cfg["eval_mode"] = "full"
    ds_val_full = create_dataset(cfg, split="val")

    data_cfg["eval_mode"] = "balanced_fixed"
    ds_val_balanced = create_dataset(cfg, split="val")

    # restore (so cfg isn't left in a surprising state)
    data_cfg["eval_mode"] = orig_eval_mode
    
    model = build_unet(cfg)
    diffusion = Diffusion(cfg)

    ema_decay = float(cfg.get("training", {}).get("ema_decay", 0.0))
    use_ema = (ema_decay is not None) and (ema_decay > 0.0)
    ema = EMA(model, ema_decay) if use_ema else None

    lr = float(cfg["training"]["lr"])
    num_epochs = int(cfg["training"]["num_epochs"])
    steps_per_epoch = int(cfg["training"].get("steps_per_epoch", 2000))
    val_every_epochs = int(cfg["training"].get("val_every_epochs", 1))
    if val_every_epochs <= 0:
        raise ValueError(f"training.val_every_epochs must be >= 1, got {val_every_epochs}")
    # If val_steps <= 0, we will do a full deterministic pass.
    val_steps = int(cfg["training"].get("val_steps", 0))
    log_interval = int(cfg["training"]["log_interval_steps"])
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- evaluation -----
    evaluator = TCEvaluator(cfg)

    # ----- early stopping config -----
    es_cfg = cfg["training"].get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    patience_epochs = int(es_cfg.get("patience_epochs", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    # ----- resume state -----
    start_epoch = 0
    best_epoch_loss = float("inf")
    best_epoch_balanced = float("inf")
    best_epoch_idx = -1
    epochs_without_improvement = 0
    global_step = 0

    epoch_indices = []
    epoch_losses = []
    val_losses = []
    val_losses_balanced = []

    if resume:
        state = _load_state(out_dir)
        last_w = _find_latest_last_weights(out_dir)
        last_ema_w = _find_latest_ema_last_weights(out_dir) if use_ema else None

        if state is None or last_w is None:
            print(f"[resume] No state/last weights found in {out_dir}. Starting fresh.")
        else:
            print(f"[resume] Loading last weights: {last_w}")
            model.load_weights(str(last_w))
            if use_ema and ema is not None:
                if last_ema_w is not None:
                    print(f"[resume] Loading last EMA weights: {last_ema_w}")
                    # Temporarily load EMA weights into model, copy into EMA shadow, then restore model weights.
                    model.load_weights(str(last_ema_w))
                    ema.copy_from_model()
                    model.load_weights(str(last_w))
                else:
                    print("[resume] No EMA weights found; initializing EMA from current model weights.")
                    ema.copy_from_model()
            # Continue from next epoch
            start_epoch = int(state.get("last_epoch", 0))
            best_epoch_loss = float(state.get("best_epoch_loss", best_epoch_loss))
            best_epoch_balanced = float(state.get("best_epoch_balanced", best_epoch_balanced))
            best_epoch_idx = int(state.get("best_epoch", best_epoch_idx))
            epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            global_step = int(state.get("global_step", 0))

            # If you want continuous loss curves across resumes:
            epoch_indices = list(state.get("epoch_indices", []))
            epoch_losses = list(state.get("epoch_losses", []))
            val_losses = list(state.get("val_losses", []))
            val_losses_balanced = list(state.get("val_losses_balanced", []))

            print(
                f"[resume] start_epoch={start_epoch}, best_epoch={best_epoch_idx}, "
                f"best_loss={best_epoch_loss:.6f}, global_step={global_step}"
            )

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        epoch_loss_sum = 0.0
        epoch_batches = 0

        pbar = tqdm(ds_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch, (x0, cond) in enumerate(pbar):
            if batch >= steps_per_epoch:
                break
            with tf.GradientTape() as tape:
                loss = diffusion.loss(model, x0, cond)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if use_ema and ema is not None:
                ema.update()
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
        val_loss = None
        val_loss_balanced = None
        if (epoch + 1) % val_every_epochs == 0:
            val_loss = evaluate_loss(diffusion, model, ds_val_full, val_steps=val_steps)
            val_loss_balanced = evaluate_loss(diffusion, model, ds_val_balanced, val_steps=val_steps)
            val_losses.append(val_loss)
            val_losses_balanced.append(val_loss_balanced)
            print(f"Epoch {epoch+1} val mean loss (full): {val_loss:.6f}")
            print(f"Epoch {epoch+1} val mean loss (balanced_fixed): {val_loss_balanced:.6f}")

        # ----- ALWAYS save "last" -----
        last_path = out_dir / f"weights_last.epoch_{epoch+1}.weights.h5"
        model.save_weights(last_path)
        _delete_other_last_weights(out_dir, keep=last_path)
        if use_ema and ema is not None:
            ema_last_path = out_dir / f"weights_ema_last.epoch_{epoch+1}.weights.h5"
            snap = ema.get_model_weights_snapshot()
            ema.copy_to_model()
            model.save_weights(ema_last_path)
            ema.restore_model_weights_snapshot(snap)
            _delete_other_ema_last_weights(out_dir, keep=ema_last_path)
            print(f"  Saved EMA last weights to {ema_last_path}")
        print(f"  Saved last weights to {last_path}")

        # ----- check for improvement (best) on FULL validation only -----
        if val_loss is not None:
            metric = val_loss
            improved = (best_epoch_loss - metric > min_delta)
            if improved:
                best_epoch_loss = metric
                best_epoch_idx = epoch + 1
                epochs_without_improvement = 0

                best_path = out_dir / "weights_best_val.weights.h5"
                model.save_weights(best_path)
                print(f"  New best model (epoch {best_epoch_idx}), saved to {best_path} (metric={best_epoch_loss:.6f})")

                if use_ema and ema is not None:
                    best_ema_path = out_dir / "weights_ema_best_val.weights.h5"
                    snap = ema.get_model_weights_snapshot()
                    ema.copy_to_model()
                    model.save_weights(best_ema_path)
                    ema.restore_model_weights_snapshot(snap)
                    print(f"  New best EMA model (epoch {best_epoch_idx}), saved to {best_ema_path}")
            else:
                epochs_without_improvement += 1
                print(f"  No significant improvement ({epochs_without_improvement}/{patience_epochs})")
        else:
            print("  Skipping early-stopping update (validation not run this epoch).")

        # Also track a best model on the balanced_fixed validation metric (does NOT drive early stopping).
        if val_loss_balanced is not None and (best_epoch_balanced - val_loss_balanced > min_delta):
            best_epoch_balanced = val_loss_balanced
            best_bal_path = out_dir / "weights_best_balanced_val.weights.h5"
            model.save_weights(best_bal_path)
            print(f"  New best BALANCED model, saved to {best_bal_path} (metric={best_epoch_balanced:.6f})")
            if use_ema and ema is not None:
                best_bal_ema_path = out_dir / "weights_ema_best_balanced_val.weights.h5"
                snap = ema.get_model_weights_snapshot()
                ema.copy_to_model()
                model.save_weights(best_bal_ema_path)
                ema.restore_model_weights_snapshot(snap)
                print(f"  New best BALANCED EMA model, saved to {best_bal_ema_path}")


        # ----- persist run state (for resuming) -----
        _save_state(
            out_dir,
            {
                "last_epoch": epoch + 1,  # completed epochs
                "global_step": global_step,
                "best_epoch": best_epoch_idx,
                "best_epoch_loss": best_epoch_loss,
                "best_epoch_balanced": best_epoch_balanced,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_indices": epoch_indices,
                "epoch_losses": epoch_losses,
                "val_losses": val_losses,
                "val_losses_balanced": val_losses_balanced,
            },
        )

        # ----- early stopping -----
        if es_enabled and epochs_without_improvement >= patience_epochs:
            print(
                f"\n Early stopping at epoch {epoch+1} "
                f"(best epoch: {best_epoch_idx}, with loss {best_epoch_loss:.6f})"
            )
            break
    

    
    # ----- post-training physics evaluation (always) -----
    heavy = True
    tag = "post_training"
    eval_weights = out_dir / "weights_best_val.weights.h5"
    if use_ema and ema is not None:
        ema_best = out_dir / "weights_ema_best_val.weights.h5"
        if ema_best.exists():
            eval_weights = ema_best

    if not eval_weights.exists():
        raise FileNotFoundError(
            f"Post-training evaluation requires best weights, but none were found at {eval_weights}."
        )

    model.load_weights(str(eval_weights))
    evaluator.run(model=model, diffusion=diffusion, out_dir=out_dir, tag=tag, heavy=heavy)
    print(f"  [eval] wrote eval/{tag} (heavy={heavy}) using {eval_weights.name}")

    return {
        "epoch": epoch_indices,
        "epoch_loss": epoch_losses,
        "best_epoch": best_epoch_idx,
        "best_epoch_loss": best_epoch_loss,
        "stopped_early": es_enabled and epochs_without_improvement >= patience_epochs,
    }
