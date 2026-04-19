from __future__ import annotations

from typing import Dict

import tensorflow as tf


def evaluator_loss_components(
    *,
    labels,
    wind_z_true,
    sample_weight,
    outputs,
    lambda_cls: float = 1.0,
    lambda_wind: float = 1.0,
    huber_delta: float = 1.0,
    label_smoothing: float = 0.0,
) -> Dict[str, tf.Tensor]:
    logits = tf.cast(outputs["class_logits"], tf.float32)
    wind_z_pred = tf.squeeze(tf.cast(outputs["wind_z"], tf.float32), axis=-1)

    labels = tf.cast(labels, tf.int32)
    wind_z_true = tf.cast(wind_z_true, tf.float32)
    sample_weight = tf.cast(sample_weight, tf.float32)

    ce = classification_cross_entropy_per_sample(
        labels=labels,
        logits=logits,
        label_smoothing=float(label_smoothing),
    )
    reg = huber_per_sample(
        y_true=wind_z_true,
        y_pred=wind_z_pred,
        delta=float(huber_delta),
    )

    cls_loss, cls_num, weight_sum = weighted_mean(ce, sample_weight)
    reg_loss, reg_num, _ = weighted_mean(reg, sample_weight)
    total = float(lambda_cls) * cls_loss + float(lambda_wind) * reg_loss

    return {
        "total_loss": total,
        "cls_loss": cls_loss,
        "reg_loss": reg_loss,
        "cls_num": cls_num,
        "reg_num": reg_num,
        "weight_sum": weight_sum,
    }


def huber_per_sample(*, y_true, y_pred, delta: float = 1.0) -> tf.Tensor:
    err = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    abs_err = tf.abs(err)
    delta_t = tf.constant(float(delta), dtype=tf.float32)
    quadratic = tf.minimum(abs_err, delta_t)
    linear = abs_err - quadratic
    return 0.5 * tf.square(quadratic) + delta_t * linear


def classification_cross_entropy_per_sample(
    *,
    labels,
    logits,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    if smoothing <= 0.0:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.int32),
            logits=tf.cast(logits, tf.float32),
        )

    logits = tf.cast(logits, tf.float32)
    num_classes = tf.shape(logits)[-1]
    labels_one_hot = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=num_classes,
        dtype=tf.float32,
    )
    smooth = tf.constant(smoothing, dtype=tf.float32)
    num_classes_f = tf.cast(num_classes, tf.float32)
    labels_smoothed = labels_one_hot * (1.0 - smooth) + smooth / num_classes_f
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_smoothed,
        logits=logits,
    )


def weighted_mean(per_sample, sample_weight):
    per_sample = tf.cast(per_sample, tf.float32)
    sample_weight = tf.cast(sample_weight, tf.float32)
    numerator = tf.reduce_sum(per_sample * sample_weight)
    denominator = tf.maximum(tf.reduce_sum(sample_weight), tf.constant(1.0e-8, tf.float32))
    return numerator / denominator, numerator, denominator
