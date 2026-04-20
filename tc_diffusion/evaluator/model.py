from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import resolve_evaluator_num_classes
from ..model_unet import GroupNorm


def build_evaluator_model(cfg: Dict[str, Any]) -> keras.Model:
    ev_cfg = cfg.get("evaluator", {})
    model_cfg = ev_cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    image_size = int(model_cfg.get("image_size", data_cfg.get("image_size", 256)))
    input_shape = tuple(model_cfg.get("input_shape", [image_size, image_size, 1]))
    num_classes = resolve_evaluator_num_classes(cfg)
    stage_channels = tuple(int(c) for c in model_cfg.get("stage_channels", [32, 64, 128, 256]))
    blocks_per_stage = int(model_cfg.get("blocks_per_stage", 2))
    embedding_dim = int(model_cfg.get("embedding_dim", 256))
    norm = str(model_cfg.get("norm", "group")).strip().lower()
    gn_groups = int(model_cfg.get("gn_groups", 8))
    spatial_dropout_rate = float(model_cfg.get("spatial_dropout_rate", 0.0))
    embedding_dropout_rate = float(
        model_cfg.get("embedding_dropout_rate", model_cfg.get("dropout", 0.0))
    )

    if len(input_shape) != 3 or int(input_shape[-1]) != 1:
        raise ValueError(f"Evaluator expects input_shape [H, W, 1], got {input_shape}")
    if not stage_channels:
        raise ValueError("evaluator.model.stage_channels must contain at least one stage.")
    if blocks_per_stage <= 0:
        raise ValueError("evaluator.model.blocks_per_stage must be > 0.")
    _validate_dropout_rate(spatial_dropout_rate, "spatial_dropout_rate")
    _validate_dropout_rate(embedding_dropout_rate, "embedding_dropout_rate")

    inputs = keras.Input(shape=input_shape, name="bt")
    x = layers.Conv2D(
        stage_channels[0],
        3,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = _norm(x, norm=norm, groups=gn_groups, name="stem_norm")
    x = _silu(x, name="stem_silu")
    x = _spatial_dropout(x, spatial_dropout_rate, name="stem_spatial_dropout")

    for stage_idx, channels in enumerate(stage_channels):
        for block_idx in range(blocks_per_stage):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            x = _residual_block(
                x,
                channels=channels,
                stride=stride,
                norm=norm,
                groups=gn_groups,
                name=f"stage{stage_idx + 1}_block{block_idx + 1}",
            )
        if stage_idx == 1:
            x = _spatial_dropout(
                x,
                spatial_dropout_rate,
                name=f"stage{stage_idx + 1}_spatial_dropout",
            )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(embedding_dim, name="embedding_dense")(x)
    x = _silu(x, name="embedding_silu")
    if embedding_dropout_rate > 0.0:
        x = layers.Dropout(embedding_dropout_rate, name="embedding_dropout")(x)

    class_logits = layers.Dense(
        num_classes,
        dtype="float32",
        name="class_logits",
    )(x)
    wind = layers.Dense(1, dtype="float32", name="wind_z")(x)

    return keras.Model(
        inputs=inputs,
        outputs={"class_logits": class_logits, "wind_z": wind},
        name="tc_evaluator_resnet",
    )


def _residual_block(
    x,
    *,
    channels: int,
    stride: int,
    norm: str,
    groups: int,
    name: str,
):
    shortcut = x

    h = layers.Conv2D(
        channels,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    h = _norm(h, norm=norm, groups=groups, name=f"{name}_norm1")
    h = _silu(h, name=f"{name}_silu1")
    h = layers.Conv2D(
        channels,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(h)
    h = _norm(h, norm=norm, groups=groups, name=f"{name}_norm2")

    if stride != 1 or int(shortcut.shape[-1]) != int(channels):
        shortcut = layers.Conv2D(
            channels,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name}_skip_conv",
        )(shortcut)
        shortcut = _norm(shortcut, norm=norm, groups=groups, name=f"{name}_skip_norm")

    x = layers.Add(name=f"{name}_add")([shortcut, h])
    return _silu(x, name=f"{name}_out_silu")


def _norm(x, *, norm: str, groups: int, name: str):
    if norm in {"group", "groupnorm", "gn"}:
        return GroupNorm(groups=groups, name=name)(x)
    if norm in {"batch", "batchnorm", "bn"}:
        return layers.BatchNormalization(name=name)(x)
    if norm in {"none", "identity"}:
        return x
    raise ValueError(
        f"Unsupported evaluator.model.norm={norm!r}; expected 'group', 'batch', or 'none'."
    )


def _silu(x, *, name: str):
    return layers.Activation(tf.nn.silu, name=name)(x)


def _spatial_dropout(x, rate: float, *, name: str):
    if float(rate) <= 0.0:
        return x
    return layers.SpatialDropout2D(float(rate), name=name)(x)


def _validate_dropout_rate(rate: float, name: str):
    if not 0.0 <= float(rate) < 1.0:
        raise ValueError(f"evaluator.model.{name} must be in [0, 1), got {rate}")
