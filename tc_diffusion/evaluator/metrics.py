from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def compute_evaluator_report(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wind_true_kt: np.ndarray,
    wind_pred_kt: np.ndarray,
    loss_summary: Dict[str, float],
    num_classes: int,
    class_names: Dict[int, str],
    wind_std_kt: float,
    tail_classes: Sequence[int] = (4, 5),
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    wind_true_kt = np.asarray(wind_true_kt, dtype=np.float32)
    wind_pred_kt = np.asarray(wind_pred_kt, dtype=np.float32)

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    cls = classification_report(
        cm,
        class_names=class_names,
    )
    reg = regression_report(
        labels=y_true,
        wind_true_kt=wind_true_kt,
        wind_pred_kt=wind_pred_kt,
        num_classes=num_classes,
        class_names=class_names,
    )
    tail = tail_report(
        labels=y_true,
        preds=y_pred,
        wind_true_kt=wind_true_kt,
        wind_pred_kt=wind_pred_kt,
        per_class_recall=cls["per_class_recall"],
        num_classes=num_classes,
        class_names=class_names,
        tail_classes=tail_classes,
    )
    selection = selection_report(
        classification=cls,
        regression=reg,
        tail=tail,
        wind_std_kt=wind_std_kt,
    )

    return to_builtin(
        {
            "num_samples": int(y_true.shape[0]),
            "loss": {
                "total": float(loss_summary["total_loss"]),
                "classification": float(loss_summary["classification_loss"]),
                "regression": float(loss_summary["regression_loss"]),
            },
            "classification": cls,
            "regression": reg,
            "tail": tail,
            "selection": selection,
        }
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    valid = (
        (y_true >= 0)
        & (y_true < int(num_classes))
        & (y_pred >= 0)
        & (y_pred < int(num_classes))
    )
    if np.any(valid):
        np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def classification_report(
    cm: np.ndarray,
    *,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    num_classes = int(cm.shape[0])
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    present = support > 0

    per_class = {}
    for c in range(num_classes):
        per_class[str(c)] = {
            "name": class_names.get(c, f"class_{c}"),
            "support": int(support[c]),
            "predicted": int(predicted[c]),
            "precision": _none_if_absent(precision[c], predicted[c] > 0),
            "recall": _none_if_absent(recall[c], support[c] > 0),
            "f1": _none_if_absent(f1[c], support[c] > 0),
        }

    balanced_accuracy = float(np.mean(recall[present])) if np.any(present) else None
    macro_f1 = float(np.mean(f1[present])) if np.any(present) else None
    accuracy = float(tp.sum() / max(1, cm.sum()))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "per_class_recall": {
            str(c): _none_if_absent(recall[c], support[c] > 0)
            for c in range(num_classes)
        },
    }


def regression_report(
    *,
    labels: np.ndarray,
    wind_true_kt: np.ndarray,
    wind_pred_kt: np.ndarray,
    num_classes: int,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    err = wind_pred_kt.astype(np.float64) - wind_true_kt.astype(np.float64)
    abs_err = np.abs(err)
    sq_err = np.square(err)

    by_class = {}
    for c in range(int(num_classes)):
        mask = labels == c
        by_class[str(c)] = _regression_subset_report(
            mask=mask,
            err=err,
            abs_err=abs_err,
            sq_err=sq_err,
            name=class_names.get(c, f"class_{c}"),
        )

    return {
        "mae_kt": float(np.mean(abs_err)) if abs_err.size else None,
        "rmse_kt": float(np.sqrt(np.mean(sq_err))) if sq_err.size else None,
        "bias_kt": float(np.mean(err)) if err.size else None,
        "by_class": by_class,
    }


def tail_report(
    *,
    labels: np.ndarray,
    preds: np.ndarray,
    wind_true_kt: np.ndarray,
    wind_pred_kt: np.ndarray,
    per_class_recall: Dict[str, float | None],
    num_classes: int,
    class_names: Dict[int, str],
    tail_classes: Sequence[int],
) -> Dict[str, Any]:
    available_tail = tuple(c for c in (int(x) for x in tail_classes) if 0 <= c < int(num_classes))
    strongest_present = sorted({int(c) for c in labels.tolist() if 0 <= int(c) < int(num_classes)})
    strongest_class = strongest_present[-1] if strongest_present else None

    out: Dict[str, Any] = {
        "tail_classes": list(available_tail),
        "strongest_class": strongest_class,
        "recall_for_strongest_classes": {
            str(c): per_class_recall.get(str(c)) for c in available_tail
        },
    }
    if strongest_class is not None and str(strongest_class) not in out["recall_for_strongest_classes"]:
        out["recall_for_strongest_classes"][str(strongest_class)] = per_class_recall.get(
            str(strongest_class)
        )

    if 4 < int(num_classes) and 5 < int(num_classes):
        cat45 = (4, 5)
        mask = np.isin(labels, cat45)
        pred_tail = np.isin(preds, cat45)
        err = wind_pred_kt.astype(np.float64) - wind_true_kt.astype(np.float64)
        abs_err = np.abs(err)
        n = int(np.sum(mask))
        out["cat45"] = {
            "classes": [4, 5],
            "n": n,
            "mae_kt": float(np.mean(abs_err[mask])) if n > 0 else None,
            "bias_kt": float(np.mean(err[mask])) if n > 0 else None,
            "combined_recall": float(np.sum(mask & pred_tail) / n) if n > 0 else None,
        }

    return out


def selection_report(
    *,
    classification: Dict[str, Any],
    regression: Dict[str, Any],
    tail: Dict[str, Any],
    wind_std_kt: float,
) -> Dict[str, Any]:
    std = max(float(wind_std_kt), 1.0e-6)
    cat45 = tail.get("cat45")
    if isinstance(cat45, dict) and cat45.get("n", 0) > 0:
        mae = cat45.get("mae_kt")
        recall = cat45.get("combined_recall")
        source = "cat45"
    else:
        mae = regression.get("mae_kt")
        recall = classification.get("balanced_accuracy")
        source = "overall_fallback"

    mae_term = 1.0 if mae is None else float(mae) / std
    recall_term = 1.0 if recall is None else 1.0 - float(recall)
    return {
        "tail_score": float(mae_term + recall_term),
        "tail_score_source": source,
        "tail_score_terms": {
            "mae_over_train_std": float(mae_term),
            "one_minus_recall": float(recall_term),
        },
    }


def make_loss_summary(
    *,
    cls_num: float,
    reg_num: float,
    weight_sum: float,
    lambda_cls: float,
    lambda_wind: float,
) -> Dict[str, float]:
    denom = max(float(weight_sum), 1.0e-8)
    cls_loss = float(cls_num) / denom
    reg_loss = float(reg_num) / denom
    return {
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
        "total_loss": float(lambda_cls) * cls_loss + float(lambda_wind) * reg_loss,
    }


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _regression_subset_report(
    *,
    mask: np.ndarray,
    err: np.ndarray,
    abs_err: np.ndarray,
    sq_err: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    n = int(np.sum(mask))
    if n <= 0:
        return {
            "name": name,
            "n": 0,
            "mae_kt": None,
            "rmse_kt": None,
            "bias_kt": None,
        }
    return {
        "name": name,
        "n": n,
        "mae_kt": float(np.mean(abs_err[mask])),
        "rmse_kt": float(np.sqrt(np.mean(sq_err[mask]))),
        "bias_kt": float(np.mean(err[mask])),
    }


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _none_if_absent(value: float, present: bool):
    if not present:
        return None
    return float(value)
