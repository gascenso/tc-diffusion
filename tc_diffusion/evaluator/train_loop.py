from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from .data import (
    EvaluatorDataBundle,
    build_sampler_summary,
    build_evaluator_data,
    bundle_summary,
    make_batch_dataset,
    resolve_class_weight_alpha,
    resolve_sampler_config,
)
from .losses import evaluator_loss_components
from .metrics import compute_evaluator_report, make_loss_summary, to_builtin
from .model import build_evaluator_model


def train_evaluator(cfg: Dict[str, Any], *, resume: bool = False) -> Dict[str, Any]:
    configure_runtime(cfg)
    ev_cfg = cfg.setdefault("evaluator", {})
    train_cfg = _default_train_cfg(ev_cfg)
    loss_cfg = _default_loss_cfg(ev_cfg)
    metric_cfg = _default_metric_cfg(ev_cfg)
    ckpt_cfg = _default_checkpoint_cfg(ev_cfg)

    seed = int(train_cfg["seed"])
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[evaluator] Global seed set to {seed}")

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_evaluator_data(cfg, splits=("train", "val"))
    _save_startup_artifacts(out_dir=out_dir, bundle=bundle, cfg=cfg)
    _print_data_summary(bundle)
    sampler_cfg = resolve_sampler_config(ev_cfg)

    model = build_evaluator_model(cfg)
    optimizer = _make_adamw(
        learning_rate=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    use_mixed_precision = bool(train_cfg.get("mixed_precision", False))
    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch_var,
        global_step=global_step,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(out_dir / "train_checkpoints"),
        max_to_keep=int(ckpt_cfg["max_to_keep"]),
    )

    state = _load_state(out_dir)
    history = [] if state is None else list(state.get("history", []))
    best = {
        "epoch": -1,
        "metric": None,
        "monitor": ckpt_cfg["monitor"],
        "mode": ckpt_cfg["mode"],
    }
    if state is not None and isinstance(state.get("best"), dict):
        best.update(state["best"])

    start_epoch = 0
    if resume:
        latest = checkpoint_manager.latest_checkpoint
        if latest is None:
            print(f"[resume] No evaluator checkpoint found in {out_dir}; starting fresh.")
        else:
            print(f"[resume] Restoring evaluator checkpoint: {latest}")
            checkpoint.restore(latest).expect_partial()
            start_epoch = int(epoch_var.numpy())
            print(
                f"[resume] start_epoch={start_epoch}, "
                f"global_step={int(global_step.numpy())}, "
                f"best_epoch={best['epoch']}, best_metric={best['metric']}"
            )

    train_ds = make_batch_dataset(
        bundle,
        "train",
        shuffle=True,
        seed=seed,
        drop_remainder=bool(train_cfg.get("drop_remainder", False)),
        sampler_config=sampler_cfg,
    )
    train_ds._epoch_index = start_epoch
    val_ds = make_batch_dataset(bundle, "val", shuffle=False, seed=seed)

    scheduler = PlateauScheduler(
        optimizer=optimizer,
        **_default_scheduler_cfg(ev_cfg, ckpt_cfg),
    )
    if best.get("metric") is not None:
        scheduler.best = float(best["metric"])
    early = _default_early_stopping_cfg(ev_cfg)
    _print_training_control_summary(
        cfg=cfg,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        early_cfg=early,
        ckpt_cfg=ckpt_cfg,
    )
    epochs_without_improvement = int(
        state.get("epochs_without_improvement", 0) if state is not None else 0
    )

    jit_compile = bool(train_cfg.get("jit_compile", False))
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

    @tf.function(reduce_retracing=True, jit_compile=jit_compile)
    def train_step(x, labels, wind_z, sample_weight):
        with tf.GradientTape() as tape:
            outputs = model(x, training=True)
            comps = evaluator_loss_components(
                labels=labels,
                wind_z_true=wind_z,
                sample_weight=sample_weight,
                outputs=outputs,
                lambda_cls=float(loss_cfg["lambda_cls"]),
                lambda_wind=float(loss_cfg["lambda_wind"]),
                huber_delta=float(loss_cfg["huber_delta"]),
                label_smoothing=float(loss_cfg["label_smoothing"]),
            )
            loss = comps["total_loss"]
            if use_mixed_precision:
                if hasattr(optimizer, "get_scaled_loss"):
                    scaled_loss = optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = optimizer.scale_loss(loss)
            else:
                scaled_loss = loss

        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if use_mixed_precision and hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        finite = tf.math.is_finite(loss)
        grads = [
            None if g is None else tf.where(finite, g, tf.zeros_like(g))
            for g in grads
        ]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return comps, outputs

    num_epochs = int(train_cfg["num_epochs"])
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = ev_cfg.get("validation", {}).get("max_batches")
    max_val_batches = None if max_val_batches is None else int(max_val_batches)
    log_interval = int(train_cfg["log_interval_steps"])

    for epoch in range(start_epoch, num_epochs):
        epoch_idx = epoch + 1
        print(f"\n[evaluator] Epoch {epoch_idx}/{num_epochs}")
        _print_epoch_sampler_summary(
            train_ds,
            bundle=bundle,
            epoch=epoch_idx,
            max_batches=max_train_batches,
        )
        train_report = _run_train_epoch(
            train_step=train_step,
            dataset=train_ds,
            bundle=bundle,
            loss_cfg=loss_cfg,
            metric_cfg=metric_cfg,
            global_step=global_step,
            log_interval=log_interval,
            max_batches=max_train_batches,
        )

        val_report = evaluate_evaluator(
            model=model,
            dataset=val_ds,
            bundle=bundle,
            split_name="val",
            loss_cfg=loss_cfg,
            metric_cfg=metric_cfg,
            max_batches=max_val_batches,
            show_progress=bool(ev_cfg.get("validation", {}).get("show_progress", False)),
        )

        monitor_value = resolve_monitor_value(val_report, ckpt_cfg["monitor"])
        improved = _is_improved(
            current=monitor_value,
            best=best.get("metric"),
            mode=ckpt_cfg["mode"],
            min_delta=float(ckpt_cfg["min_delta"]),
        )

        lr_value = _get_lr(optimizer)
        record = {
            "epoch": epoch_idx,
            "global_step": int(global_step.numpy()),
            "lr": lr_value,
            "train_sampler": _epoch_sampler_record(
                train_ds,
                bundle=bundle,
                max_batches=max_train_batches,
            ),
            "monitor": ckpt_cfg["monitor"],
            "monitor_value": monitor_value,
            "best_improved": bool(improved),
            "train": train_report,
            "val": val_report,
        }
        _append_jsonl(out_dir / "metrics_epoch.jsonl", record)
        _write_json(out_dir / "history.json", {"epochs": history + [record]})
        _write_split_report(out_dir, "train", epoch_idx, train_report)
        _write_split_report(out_dir, "val", epoch_idx, val_report)

        print(
            "  train: "
            f"loss={train_report['loss']['total']:.5f}, "
            f"bal_acc={_fmt(train_report['classification']['balanced_accuracy'])}, "
            f"mae={_fmt(train_report['regression']['mae_kt'])} kt"
        )
        print(
            "  val:   "
            f"loss={val_report['loss']['total']:.5f}, "
            f"bal_acc={_fmt(val_report['classification']['balanced_accuracy'])}, "
            f"macro_f1={_fmt(val_report['classification']['macro_f1'])}, "
            f"{_format_tail_summary(val_report)}, "
            f"tail_score={_fmt(val_report['selection']['tail_score'])}"
        )

        raw_snapshot_source = None
        if bool(train_cfg.get("save_last_weights", True)):
            last_path = out_dir / "weights_last.weights.h5"
            model.save_weights(last_path)
            raw_snapshot_source = last_path

        if improved:
            best = {
                "epoch": epoch_idx,
                "metric": monitor_value,
                "monitor": ckpt_cfg["monitor"],
                "mode": ckpt_cfg["mode"],
            }
            best_path = out_dir / "weights_best_tail.weights.h5"
            if raw_snapshot_source is not None:
                shutil.copyfile(raw_snapshot_source, best_path)
            else:
                model.save_weights(best_path)
            model.save_weights(out_dir / "weights_best.weights.h5")
            _write_json(out_dir / "best_metrics.json", {"best": best, "val": val_report})
            epochs_without_improvement = 0
            print(
                f"  New best evaluator checkpoint: epoch={epoch_idx}, "
                f"{ckpt_cfg['monitor']}={monitor_value:.6f} "
                f"({_format_best_reason(val_report)})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  No best-checkpoint improvement "
                f"(current {ckpt_cfg['monitor']}={monitor_value:.6f}, "
                f"best={_fmt(best.get('metric'))}; "
                f"{epochs_without_improvement}/{early['patience_epochs']})"
            )

        scheduler.step(monitor_value)

        epoch_var.assign(epoch_idx)
        ckpt_path = checkpoint_manager.save(checkpoint_number=epoch_idx)
        print(f"  Saved train checkpoint to {ckpt_path}")

        history.append(record)
        _save_state(
            out_dir,
            {
                "last_epoch": epoch_idx,
                "global_step": int(global_step.numpy()),
                "best": best,
                "history": history,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )

        if bool(early["enabled"]) and epochs_without_improvement >= int(
            early["patience_epochs"]
        ):
            print(
                f"[evaluator] Early stopping at epoch {epoch_idx}; "
                f"patience={early['patience_epochs']}, "
                f"best epoch {best['epoch']} ({best['monitor']}={best['metric']})"
            )
            break

    return {
        "best": best,
        "history": history,
        "output_dir": str(out_dir),
    }


def evaluate_saved_evaluator(
    cfg: Dict[str, Any],
    *,
    weights_path: Path,
    split_name: str,
    out_dir: Path,
    tag: str,
    show_progress: bool = False,
) -> Dict[str, Any]:
    configure_runtime(cfg)
    split_name = str(split_name)
    bundle = build_evaluator_data(cfg, splits=("train", split_name))
    model = build_evaluator_model(cfg)
    model.load_weights(str(weights_path))
    ev_cfg = cfg.setdefault("evaluator", {})
    ds = make_batch_dataset(bundle, split_name, shuffle=False, seed=int(_default_train_cfg(ev_cfg)["seed"]))
    report = evaluate_evaluator(
        model=model,
        dataset=ds,
        bundle=bundle,
        split_name=split_name,
        loss_cfg=_default_loss_cfg(ev_cfg),
        metric_cfg=_default_metric_cfg(ev_cfg),
        max_batches=ev_cfg.get("validation", {}).get("max_batches"),
        show_progress=show_progress,
    )
    eval_root = Path(out_dir) / "eval_evaluator" / str(tag)
    _write_json(eval_root / f"{split_name}_metrics.json", report)
    _write_json(
        eval_root / f"{split_name}_confusion_matrix.json",
        {"matrix": report["classification"]["confusion_matrix"]},
    )
    _write_json(eval_root / f"{split_name}_per_class.json", report["classification"]["per_class"])
    return report


def evaluate_evaluator(
    *,
    model: keras.Model,
    dataset,
    bundle: EvaluatorDataBundle,
    split_name: str,
    loss_cfg: Dict[str, float],
    metric_cfg: Dict[str, Any],
    max_batches: int | None = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    y_true, y_pred, wind_true, wind_pred = [], [], [], []
    cls_num = 0.0
    reg_num = 0.0
    weight_sum = 0.0

    iterator = iter(dataset)
    total = len(dataset) if max_batches is None else min(len(dataset), int(max_batches))
    if show_progress:
        iterator = tqdm(iterator, total=total, desc=f"eval {split_name}", leave=False)

    for batch_idx, (x, targets) in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        outputs = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False)
        comps = evaluator_loss_components(
            labels=targets["class"],
            wind_z_true=targets["wind_z"],
            sample_weight=targets["sample_weight"],
            outputs=outputs,
            lambda_cls=float(loss_cfg["lambda_cls"]),
            lambda_wind=float(loss_cfg["lambda_wind"]),
            huber_delta=float(loss_cfg["huber_delta"]),
            label_smoothing=float(loss_cfg["label_smoothing"]),
        )
        logits = outputs["class_logits"].numpy()
        pred_z = tf.squeeze(outputs["wind_z"], axis=-1).numpy()
        y_true.append(targets["class"])
        y_pred.append(np.argmax(logits, axis=-1).astype(np.int32))
        wind_true.append(targets["wind_kt"])
        wind_pred.append(bundle.wind.inverse(pred_z))
        cls_num += float(comps["cls_num"].numpy())
        reg_num += float(comps["reg_num"].numpy())
        weight_sum += float(comps["weight_sum"].numpy())

    return _finalize_report(
        y_true=y_true,
        y_pred=y_pred,
        wind_true=wind_true,
        wind_pred=wind_pred,
        cls_num=cls_num,
        reg_num=reg_num,
        weight_sum=weight_sum,
        bundle=bundle,
        loss_cfg=loss_cfg,
        metric_cfg=metric_cfg,
    )


def configure_runtime(cfg: Dict[str, Any]):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    ev_train_cfg = cfg.get("evaluator", {}).get("training", {})
    use_mixed_precision = bool(ev_train_cfg.get("mixed_precision", False))
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[evaluator] Mixed precision enabled (mixed_float16)")


def resolve_monitor_value(report: Dict[str, Any], monitor: str) -> float:
    value: Any = report
    for part in str(monitor).split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Monitor '{monitor}' not found in evaluator report.")
        value = value[part]
    if value is None:
        raise ValueError(f"Monitor '{monitor}' resolved to None.")
    return float(value)


class PlateauScheduler:
    def __init__(
        self,
        *,
        optimizer,
        enabled: bool,
        monitor: str,
        mode: str,
        factor: float,
        patience_epochs: int,
        min_delta: float,
        min_lr: float,
        cooldown_epochs: int,
    ):
        self.optimizer = optimizer
        self.enabled = bool(enabled)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.factor = float(factor)
        self.patience_epochs = int(patience_epochs)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)
        self.cooldown_epochs = int(cooldown_epochs)
        self.best = None
        self.wait = 0
        self.cooldown = 0

    def step(self, value: float):
        if not self.enabled:
            return
        value = float(value)
        if _is_improved(value, self.best, mode=self.mode, min_delta=self.min_delta):
            self.best = value
            self.wait = 0
            return

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        self.wait += 1
        if self.wait < self.patience_epochs:
            return

        current_lr = _get_lr(self.optimizer)
        new_lr = max(self.min_lr, current_lr * self.factor)
        if new_lr < current_lr - 1.0e-12:
            _set_lr(self.optimizer, new_lr)
            print(
                f"  ReduceLROnPlateau: {self.monitor} plateaued; "
                f"lr {current_lr:.3e} -> {new_lr:.3e}"
            )
        self.wait = 0
        self.cooldown = self.cooldown_epochs


def _run_train_epoch(
    *,
    train_step,
    dataset,
    bundle: EvaluatorDataBundle,
    loss_cfg: Dict[str, float],
    metric_cfg: Dict[str, Any],
    global_step: tf.Variable,
    log_interval: int,
    max_batches: int | None,
) -> Dict[str, Any]:
    y_true, y_pred, wind_true, wind_pred = [], [], [], []
    cls_num = 0.0
    reg_num = 0.0
    weight_sum = 0.0

    total = len(dataset) if max_batches is None else min(len(dataset), int(max_batches))
    pbar = tqdm(total=total, desc="train evaluator", leave=True)
    for batch_idx, (x, targets) in enumerate(dataset):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        comps, outputs = train_step(
            tf.convert_to_tensor(x, dtype=tf.float32),
            tf.convert_to_tensor(targets["class"], dtype=tf.int32),
            tf.convert_to_tensor(targets["wind_z"], dtype=tf.float32),
            tf.convert_to_tensor(targets["sample_weight"], dtype=tf.float32),
        )
        global_step.assign_add(1)
        logits = outputs["class_logits"].numpy()
        pred_z = tf.squeeze(outputs["wind_z"], axis=-1).numpy()

        y_true.append(targets["class"])
        y_pred.append(np.argmax(logits, axis=-1).astype(np.int32))
        wind_true.append(targets["wind_kt"])
        wind_pred.append(bundle.wind.inverse(pred_z))
        cls_num += float(comps["cls_num"].numpy())
        reg_num += float(comps["reg_num"].numpy())
        weight_sum += float(comps["weight_sum"].numpy())

        if int(global_step.numpy()) % max(1, int(log_interval)) == 0:
            pbar.set_postfix(
                {
                    "loss": f"{float(comps['total_loss'].numpy()):.4f}",
                    "cls": f"{float(comps['cls_loss'].numpy()):.4f}",
                    "reg": f"{float(comps['reg_loss'].numpy()):.4f}",
                }
            )
        pbar.update(1)
    pbar.close()

    return _finalize_report(
        y_true=y_true,
        y_pred=y_pred,
        wind_true=wind_true,
        wind_pred=wind_pred,
        cls_num=cls_num,
        reg_num=reg_num,
        weight_sum=weight_sum,
        bundle=bundle,
        loss_cfg=loss_cfg,
        metric_cfg=metric_cfg,
    )


def _finalize_report(
    *,
    y_true,
    y_pred,
    wind_true,
    wind_pred,
    cls_num: float,
    reg_num: float,
    weight_sum: float,
    bundle: EvaluatorDataBundle,
    loss_cfg: Dict[str, float],
    metric_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not y_true:
        raise RuntimeError("No batches were processed; cannot compute evaluator metrics.")
    loss_summary = make_loss_summary(
        cls_num=cls_num,
        reg_num=reg_num,
        weight_sum=weight_sum,
        lambda_cls=float(loss_cfg["lambda_cls"]),
        lambda_wind=float(loss_cfg["lambda_wind"]),
    )
    return compute_evaluator_report(
        y_true=np.concatenate(y_true, axis=0),
        y_pred=np.concatenate(y_pred, axis=0),
        wind_true_kt=np.concatenate(wind_true, axis=0),
        wind_pred_kt=np.concatenate(wind_pred, axis=0),
        loss_summary=loss_summary,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        wind_std_kt=bundle.wind.std_kt,
        tail_classes=tuple(metric_cfg["tail_classes"]),
    )


def _print_training_control_summary(
    *,
    cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    loss_cfg: Dict[str, Any],
    early_cfg: Dict[str, Any],
    ckpt_cfg: Dict[str, Any],
):
    ev_cfg = cfg.get("evaluator", {})
    model_cfg = ev_cfg.get("model", {})
    spatial_dropout = float(model_cfg.get("spatial_dropout_rate", 0.0))
    embedding_dropout = float(
        model_cfg.get("embedding_dropout_rate", model_cfg.get("dropout", 0.0))
    )
    print(
        "[evaluator:control] "
        f"AdamW weight_decay={float(train_cfg['weight_decay']):.3g}, "
        f"label_smoothing={float(loss_cfg['label_smoothing']):.3g}, "
        f"class_weight_alpha={resolve_class_weight_alpha(ev_cfg):.3g}, "
        f"sampler={resolve_sampler_config(ev_cfg).mode}, "
        f"sampler_strength={resolve_sampler_config(ev_cfg).strength:.3g}"
    )
    print(
        "[evaluator:control] "
        f"spatial_dropout_rate={spatial_dropout:.3g}, "
        f"embedding_dropout_rate={embedding_dropout:.3g}, "
        f"best_monitor={ckpt_cfg['monitor']} ({ckpt_cfg['mode']}), "
        f"early_stopping_enabled={bool(early_cfg['enabled'])}, "
        f"early_stopping_patience={int(early_cfg['patience_epochs'])}"
    )


def _print_epoch_sampler_summary(
    dataset,
    *,
    bundle: EvaluatorDataBundle,
    epoch: int,
    max_batches: int | None,
):
    record = _epoch_sampler_record(dataset, bundle=bundle, max_batches=max_batches)
    expected = record["expected_samples_per_class"]
    probs = record["class_probabilities"]
    ratios = record["class_probability_ratio_vs_natural"]
    print(
        "[evaluator:sampler] "
        f"epoch={epoch}, mode={record['mode']}, strength={record['strength']:.3g}, "
        f"planned_samples={record['planned_samples']}"
    )
    for c in range(bundle.num_classes):
        print(
            "  "
            f"class {c}: p={float(probs[str(c)]):.5f}, "
            f"ratio_vs_natural={_fmt(ratios[str(c)])}, "
            f"expected_n={float(expected[str(c)]):.1f}"
        )


def _epoch_sampler_record(
    dataset,
    *,
    bundle: EvaluatorDataBundle,
    max_batches: int | None,
) -> Dict[str, Any]:
    desc = dataset.describe_sampler()
    planned_samples = int(dataset.num_samples_for_batches(max_batches))
    probabilities = desc["class_probabilities"]
    desc["planned_samples"] = planned_samples
    desc["expected_samples_per_class"] = {
        str(c): float(probabilities[str(c)]) * planned_samples
        for c in range(bundle.num_classes)
    }
    return desc


def _format_tail_summary(report: Dict[str, Any]) -> str:
    cls = report.get("classification", {})
    tail = report.get("tail", {})
    per_class = cls.get("per_class", {})
    cat45 = tail.get("cat45", {}) if isinstance(tail.get("cat45"), dict) else {}
    return (
        f"cat4_rec={_fmt(_nested(per_class, '4', 'recall'))}, "
        f"cat5_rec={_fmt(_nested(per_class, '5', 'recall'))}, "
        f"cat45_rec={_fmt(cat45.get('combined_recall'))}, "
        f"cat45_mae={_fmt(cat45.get('mae_kt'))} kt, "
        f"cat45_bias={_fmt(cat45.get('bias_kt'))} kt"
    )


def _format_best_reason(report: Dict[str, Any]) -> str:
    selection = report.get("selection", {})
    terms = selection.get("tail_score_terms", {})
    source = selection.get("tail_score_source", "unknown")
    return (
        f"source={source}, "
        f"mae/std={_fmt(terms.get('mae_over_train_std'))}, "
        f"1-recall={_fmt(terms.get('one_minus_recall'))}, "
        f"{_format_tail_summary(report)}"
    )


def _nested(obj: Dict[str, Any], *keys: str):
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _make_adamw(*, learning_rate: float, weight_decay: float):
    if hasattr(keras.optimizers, "AdamW"):
        return keras.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    experimental = getattr(keras.optimizers, "experimental", None)
    if experimental is not None and hasattr(experimental, "AdamW"):
        return experimental.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    raise RuntimeError("This TensorFlow/Keras installation does not provide AdamW.")


def _default_train_cfg(ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ev_cfg.get("training", {}))
    cfg.setdefault("seed", 42)
    cfg.setdefault("num_epochs", 50)
    cfg.setdefault("batch_size", None)
    cfg.setdefault("lr", 3.0e-4)
    cfg.setdefault("weight_decay", 5.0e-4)
    cfg.setdefault("log_interval_steps", 20)
    cfg.setdefault("mixed_precision", False)
    cfg.setdefault("jit_compile", False)
    cfg.setdefault("grad_clip_norm", 1.0)
    cfg.setdefault("drop_remainder", False)
    cfg.setdefault("save_last_weights", True)
    cfg.setdefault("max_train_batches", None)
    return cfg


def _default_loss_cfg(ev_cfg: Dict[str, Any]) -> Dict[str, float]:
    cfg = dict(ev_cfg.get("loss", {}))
    cfg.setdefault("lambda_cls", 1.0)
    cfg.setdefault("lambda_wind", 1.0)
    cfg.setdefault("huber_delta", 1.0)
    if "label_smoothing" in ev_cfg:
        cfg["label_smoothing"] = ev_cfg["label_smoothing"]
    else:
        cfg.setdefault("label_smoothing", 0.04)
    return cfg


def _default_metric_cfg(ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ev_cfg.get("metrics", {}))
    cfg.setdefault("tail_classes", [4, 5])
    return cfg


def _default_checkpoint_cfg(ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ev_cfg.get("checkpoint", {}))
    cfg.setdefault("monitor", "selection.tail_score")
    cfg.setdefault("mode", "min")
    cfg.setdefault("min_delta", 0.0)
    cfg.setdefault("max_to_keep", 1)
    if cfg["mode"] not in {"min", "max"}:
        raise ValueError("evaluator.checkpoint.mode must be 'min' or 'max'.")
    return cfg


def _default_scheduler_cfg(ev_cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ev_cfg.get("reduce_lr_on_plateau", {}))
    cfg.setdefault("enabled", True)
    cfg.setdefault("monitor", ckpt_cfg["monitor"])
    cfg.setdefault("mode", ckpt_cfg["mode"])
    cfg.setdefault("factor", 0.5)
    cfg.setdefault("patience_epochs", 5)
    cfg.setdefault("min_delta", ckpt_cfg.get("min_delta", 0.0))
    cfg.setdefault("min_lr", 1.0e-6)
    cfg.setdefault("cooldown_epochs", 0)
    return cfg


def _default_early_stopping_cfg(ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ev_cfg.get("early_stopping", {}))
    cfg.setdefault("enabled", True)
    if "early_stopping_patience" in ev_cfg:
        cfg["patience_epochs"] = ev_cfg["early_stopping_patience"]
    else:
        cfg.setdefault("patience_epochs", 6)
    return cfg


def _save_startup_artifacts(
    *,
    out_dir: Path,
    bundle: EvaluatorDataBundle,
    cfg: Dict[str, Any],
):
    summary = bundle_summary(bundle)
    _write_json(out_dir / "data_summary.json", summary)
    _write_json(out_dir / "wind_normalization.json", summary["wind_standardization"])
    _write_json(out_dir / "sampler_config.json", build_sampler_summary(cfg, bundle))
    _write_json(
        out_dir / "class_weights.json",
        {
            "alpha": resolve_class_weight_alpha(cfg.get("evaluator", {})),
            "weights": summary["class_weights"],
            "train_class_counts": summary["train_class_counts"],
        },
    )


def _print_data_summary(bundle: EvaluatorDataBundle):
    print(
        "[evaluator:data] "
        f"backend={getattr(bundle.backend, 'name', type(bundle.backend).__name__)}, "
        f"train={bundle.train.size}, val={bundle.val.size}, "
        f"wind_mean={bundle.wind.mean_kt:.3f} kt, wind_std={bundle.wind.std_kt:.3f} kt"
    )
    print("[evaluator:data] train class counts:")
    for c, n in enumerate(bundle.train_class_counts.tolist()):
        print(f"  class {c}: n={int(n)}, weight={float(bundle.class_weights[c]):.4f}")


def _write_split_report(out_dir: Path, split_name: str, epoch: int, report: Dict[str, Any]):
    root = out_dir / "metrics" / split_name / f"epoch_{epoch:04d}"
    _write_json(root / "metrics.json", report)
    _write_json(
        root / "confusion_matrix.json",
        {"matrix": report["classification"]["confusion_matrix"]},
    )
    _write_json(root / "per_class_classification.json", report["classification"]["per_class"])
    _write_json(root / "per_class_regression.json", report["regression"]["by_class"])


def _is_improved(current: float, best: float | None, *, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return float(best) - float(current) > float(min_delta)
    if mode == "max":
        return float(current) - float(best) > float(min_delta)
    raise ValueError(f"Unsupported monitor mode: {mode}")


def _get_base_optimizer(optimizer):
    return getattr(optimizer, "inner_optimizer", getattr(optimizer, "_optimizer", optimizer))


def _get_lr(optimizer) -> float:
    opt = _get_base_optimizer(optimizer)
    lr = opt.learning_rate
    if callable(lr):
        lr = lr(opt.iterations)
    return float(tf.keras.backend.get_value(lr))


def _set_lr(optimizer, value: float):
    opt = _get_base_optimizer(optimizer)
    lr = opt.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(float(value))
    else:
        opt.learning_rate = float(value)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _state_path(out_dir: Path) -> Path:
    return out_dir / "run_state.json"


def _load_state(out_dir: Path) -> Dict[str, Any] | None:
    path = _state_path(out_dir)
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _save_state(out_dir: Path, state: Dict[str, Any]):
    _write_json(_state_path(out_dir), state)


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as f:
        json.dump(to_builtin(obj), f, indent=2)
    tmp.replace(path)


def _append_jsonl(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(to_builtin(obj)) + "\n")
