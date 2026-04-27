from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tc_diffusion.config import load_config

from .model import build_evaluator_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_optional_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def resolve_evaluator_run_dir(repo_root: Path, run_name: str) -> Path:
    run_dir = repo_root / "runs" / str(run_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"Evaluator run directory not found: {run_dir}")
    return run_dir


def resolve_evaluator_config_path(
    repo_root: Path,
    *,
    run_name: str,
    explicit_config_path: str | None = None,
) -> Path:
    path = _resolve_optional_path(repo_root, explicit_config_path)
    if path is None:
        path = resolve_evaluator_run_dir(repo_root, run_name) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Evaluator config not found: {path}")
    return path


def resolve_evaluator_weights_path(
    repo_root: Path,
    *,
    run_name: str,
    explicit_weights_path: str | None = None,
    weights_name: str | None = "best_tail",
) -> Path:
    path = _resolve_optional_path(repo_root, explicit_weights_path)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Evaluator weights not found: {path}")
        return path

    run_dir = resolve_evaluator_run_dir(repo_root, run_name)
    aliases = {
        "best_tail": "weights_best_tail.weights.h5",
        "best": "weights_best.weights.h5",
        "last": "weights_last.weights.h5",
    }
    if weights_name is None:
        names = ("best_tail", "best", "last")
    else:
        key = str(weights_name).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unsupported evaluator weights_name={weights_name!r}; "
                "expected one of {'best_tail', 'best', 'last'}."
            )
        names = (key,)

    candidates = [run_dir / aliases[name] for name in names]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find evaluator weights. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _ensure_channel_axis(bt: np.ndarray) -> np.ndarray:
    if bt.ndim == 3:
        return bt[..., None]
    if bt.ndim == 4 and bt.shape[-1] == 1:
        return bt
    raise ValueError(
        "Expected BT array shaped [N,H,W] or [N,H,W,1] for evaluator feature encoding, "
        f"got shape {bt.shape}."
    )


def _preprocess_bt_batch(bt_k: np.ndarray, bt_range: tuple[float, float]) -> np.ndarray:
    bt_min_k = float(bt_range[0])
    bt_max_k = float(bt_range[1])
    if bt_max_k <= bt_min_k:
        raise ValueError(
            f"Evaluator BT range must satisfy max > min, got ({bt_min_k}, {bt_max_k})."
        )

    batch = np.array(bt_k, dtype=np.float32, copy=True)
    np.nan_to_num(batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(batch, bt_min_k, bt_max_k, out=batch)
    batch = (batch - bt_min_k) / (bt_max_k - bt_min_k)
    batch = batch * 2.0 - 1.0
    return _ensure_channel_axis(batch)


@dataclass
class EvaluatorEmbeddingExtractor:
    run_name: str
    config_path: Path
    weights_path: Path
    bt_range: tuple[float, float]
    batch_size: int
    embedding_layer: str
    model: keras.Model
    feature_model: keras.Model
    embedding_dim: int

    @classmethod
    def from_run(
        cls,
        *,
        run_name: str,
        config_path: str | None = None,
        weights_path: str | None = None,
        weights_name: str | None = "best_tail",
        batch_size: int = 32,
        embedding_layer: str = "embedding_silu",
    ) -> "EvaluatorEmbeddingExtractor":
        if int(batch_size) <= 0:
            raise ValueError(f"Evaluator feature batch_size must be > 0, got {batch_size}")

        repo_root = _repo_root()
        resolved_config = resolve_evaluator_config_path(
            repo_root,
            run_name=run_name,
            explicit_config_path=config_path,
        )
        resolved_weights = resolve_evaluator_weights_path(
            repo_root,
            run_name=run_name,
            explicit_weights_path=weights_path,
            weights_name=weights_name,
        )

        cfg = load_config(str(resolved_config), overrides=[])
        model = build_evaluator_model(cfg)
        model.load_weights(str(resolved_weights))
        try:
            embedding_tensor = model.get_layer(str(embedding_layer)).output
        except ValueError as exc:
            raise ValueError(
                f"Evaluator embedding layer {embedding_layer!r} was not found in model "
                f"{model.name!r}."
            ) from exc

        feature_model = keras.Model(
            inputs=model.inputs,
            outputs=embedding_tensor,
            name=f"{model.name}_{embedding_layer}_features",
        )
        output_shape = feature_model.output_shape
        if not isinstance(output_shape, (tuple, list)) or len(output_shape) != 2:
            raise ValueError(
                f"Evaluator embedding layer {embedding_layer!r} must produce a rank-2 tensor, "
                f"got output_shape={output_shape!r}."
            )
        embedding_dim = int(output_shape[-1])
        data_cfg = cfg.get("data", {})
        bt_range = (
            float(data_cfg["bt_min_k"]),
            float(data_cfg["bt_max_k"]),
        )
        return cls(
            run_name=str(run_name),
            config_path=resolved_config,
            weights_path=resolved_weights,
            bt_range=bt_range,
            batch_size=int(batch_size),
            embedding_layer=str(embedding_layer),
            model=model,
            feature_model=feature_model,
            embedding_dim=embedding_dim,
        )

    def encode_bt_k(self, bt_k: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        arr = np.asarray(bt_k)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError(
                "Expected BT array shaped [N,H,W], [N,H,W,1], or [H,W] for evaluator "
                f"feature encoding, got shape {arr.shape}."
            )

        n = int(arr.shape[0])
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        bsz = int(self.batch_size if batch_size is None else batch_size)
        if bsz <= 0:
            raise ValueError(f"Evaluator feature batch_size must be > 0, got {bsz}")

        feats = []
        for start in range(0, n, bsz):
            stop = min(start + bsz, n)
            batch = _preprocess_bt_batch(arr[start:stop], self.bt_range)
            out = self.feature_model(tf.convert_to_tensor(batch, dtype=tf.float32), training=False)
            feats.append(np.asarray(out.numpy(), dtype=np.float32))
        return np.concatenate(feats, axis=0)

    def metadata(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config_path": str(self.config_path),
            "weights_path": str(self.weights_path),
            "embedding_layer": self.embedding_layer,
            "embedding_dim": int(self.embedding_dim),
            "bt_range_k": [float(self.bt_range[0]), float(self.bt_range[1])],
            "batch_size": int(self.batch_size),
        }
