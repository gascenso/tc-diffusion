from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from ..data import (
    _build_split_pairs,
    _ensure_channel_axis,
    build_data_backend,
    build_relpath_to_wind_kt,
    load_dataset_index,
    load_split_file_set,
    preprocess_bt,
)


@dataclass(frozen=True)
class WindStandardizer:
    mean_kt: float
    std_kt: float

    def transform(self, wind_kt: np.ndarray) -> np.ndarray:
        return ((wind_kt.astype(np.float32) - self.mean_kt) / self.std_kt).astype(
            np.float32
        )

    def inverse(self, wind_z: np.ndarray) -> np.ndarray:
        return (wind_z.astype(np.float32) * self.std_kt + self.mean_kt).astype(
            np.float32
        )

    def to_json(self) -> Dict[str, float]:
        return {
            "mean_kt": float(self.mean_kt),
            "std_kt": float(self.std_kt),
        }


@dataclass(frozen=True)
class EvaluatorSplit:
    name: str
    rel_paths: Tuple[str, ...]
    labels: np.ndarray
    wind_kt: np.ndarray
    wind_z: np.ndarray
    sample_weight: np.ndarray
    wind_metadata_fallbacks: int

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=int(num_classes)).astype(np.int64)


@dataclass(frozen=True)
class EvaluatorDataBundle:
    backend: Any
    bt_range: Tuple[float, float]
    batch_size: int
    num_classes: int
    class_names: Dict[int, str]
    class_weights: np.ndarray
    train_class_counts: np.ndarray
    wind: WindStandardizer
    splits: Dict[str, EvaluatorSplit]

    @property
    def train(self) -> EvaluatorSplit:
        return self.splits["train"]

    @property
    def val(self) -> EvaluatorSplit:
        return self.splits["val"]

    def split(self, name: str) -> EvaluatorSplit:
        return self.splits[str(name)]


@dataclass(frozen=True)
class EvaluatorSamplerConfig:
    mode: str = "natural"
    strength: float = 0.0


class NaturalEvaluatorBatchDataset:
    """Finite natural-order/shuffled batches for supervised evaluator training."""

    def __init__(
        self,
        *,
        split: EvaluatorSplit,
        backend: Any,
        bt_range: Tuple[float, float],
        batch_size: int,
        num_classes: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
    ):
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.split = split
        self.backend = backend
        self.bt_range = (float(bt_range[0]), float(bt_range[1]))
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_remainder = bool(drop_remainder)
        self._epoch_index = 0
        self.sampler_config = EvaluatorSamplerConfig(mode="natural", strength=0.0)
        self.class_probabilities = empirical_class_probabilities(
            split.class_counts(self.num_classes)
        )

    def __len__(self) -> int:
        if self.drop_remainder:
            return self.split.size // self.batch_size
        return int(math.ceil(self.split.size / float(self.batch_size)))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        idx = np.arange(self.split.size, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch_index)
            rng.shuffle(idx)
            self._epoch_index += 1

        for start in range(0, self.split.size, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            if self.drop_remainder and batch_idx.shape[0] < self.batch_size:
                break
            rel_paths = [self.split.rel_paths[int(i)] for i in batch_idx]
            bt = self.backend.load_bt_batch(rel_paths)
            bt = preprocess_bt(bt, self.bt_range)
            bt = _ensure_channel_axis(bt).astype(np.float32, copy=False)
            targets = {
                "class": self.split.labels[batch_idx].astype(np.int32, copy=False),
                "wind_z": self.split.wind_z[batch_idx].astype(np.float32, copy=False),
                "wind_kt": self.split.wind_kt[batch_idx].astype(np.float32, copy=False),
                "sample_weight": self.split.sample_weight[batch_idx].astype(
                    np.float32,
                    copy=False,
                ),
            }
            yield bt, targets

    def num_samples_for_batches(self, max_batches: int | None = None) -> int:
        n_batches = len(self)
        if max_batches is not None:
            n_batches = min(n_batches, max(0, int(max_batches)))
        if n_batches <= 0:
            return 0
        if self.drop_remainder:
            return int(n_batches * self.batch_size)
        if n_batches >= len(self):
            return int(self.split.size)
        return int(n_batches * self.batch_size)

    def describe_sampler(self) -> Dict[str, Any]:
        return sampler_description(
            mode=self.sampler_config.mode,
            strength=self.sampler_config.strength,
            probabilities=self.class_probabilities,
            class_counts=self.split.class_counts(self.num_classes),
        )


class MildRebalancedEvaluatorBatchDataset(NaturalEvaluatorBatchDataset):
    """Finite replacement sampler with class probabilities mildly shifted to tails.

    The class law is p_c proportional to empirical_p_c ** (1 - strength).
    strength=0 reproduces natural sampling; strength=1 is flat over present
    classes.  This is intentionally simple and mild by default.
    """

    def __init__(
        self,
        *,
        sampler_config: EvaluatorSamplerConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampler_config = sampler_config
        self.class_probabilities = compute_sampler_class_probabilities(
            counts=self.split.class_counts(self.num_classes),
            mode=sampler_config.mode,
            strength=sampler_config.strength,
        )
        self.classes = np.flatnonzero(self.class_probabilities > 0.0).astype(np.int64)
        if self.classes.size == 0:
            raise ValueError("Mild rebalanced sampler found no present classes.")
        self.class_probs_present = self.class_probabilities[self.classes]
        self.class_probs_present = self.class_probs_present / np.sum(
            self.class_probs_present
        )
        self.indices_by_class = {
            int(c): np.flatnonzero(self.split.labels == int(c)).astype(np.int64)
            for c in self.classes.tolist()
        }

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        rng = np.random.default_rng(self.seed + self._epoch_index)
        self._epoch_index += 1

        for start in range(0, self.split.size, self.batch_size):
            size = min(self.batch_size, self.split.size - start)
            if self.drop_remainder and size < self.batch_size:
                break
            sampled_classes = rng.choice(
                self.classes,
                size=size,
                replace=True,
                p=self.class_probs_present,
            )
            batch_idx = np.empty((size,), dtype=np.int64)
            for pos, cls in enumerate(sampled_classes.tolist()):
                candidates = self.indices_by_class[int(cls)]
                batch_idx[pos] = int(rng.choice(candidates))

            rel_paths = [self.split.rel_paths[int(i)] for i in batch_idx]
            bt = self.backend.load_bt_batch(rel_paths)
            bt = preprocess_bt(bt, self.bt_range)
            bt = _ensure_channel_axis(bt).astype(np.float32, copy=False)
            targets = {
                "class": self.split.labels[batch_idx].astype(np.int32, copy=False),
                "wind_z": self.split.wind_z[batch_idx].astype(np.float32, copy=False),
                "wind_kt": self.split.wind_kt[batch_idx].astype(np.float32, copy=False),
                "sample_weight": self.split.sample_weight[batch_idx].astype(
                    np.float32,
                    copy=False,
                ),
            }
            yield bt, targets


def build_evaluator_data(
    cfg: Dict[str, Any],
    *,
    splits: Sequence[str] = ("train", "val"),
) -> EvaluatorDataBundle:
    data_cfg = cfg["data"]
    ev_cfg = cfg.get("evaluator", {})
    ev_train_cfg = ev_cfg.get("training", {})

    num_classes = int(
        ev_cfg.get(
            "num_classes",
            cfg.get("conditioning", {}).get("num_ss_classes", 6),
        )
    )
    batch_size_cfg = ev_train_cfg.get("batch_size")
    if batch_size_cfg is None:
        batch_size_cfg = data_cfg.get("batch_size", 8)
    batch_size = int(batch_size_cfg)
    alpha = resolve_class_weight_alpha(ev_cfg)

    index_path = Path(data_cfg["dataset_index"])
    split_dir = Path(data_cfg["split_dir"])
    bt_range = (float(data_cfg["bt_min_k"]), float(data_cfg["bt_max_k"]))

    backend = build_data_backend(data_cfg)
    class_to_files, sample_meta = load_dataset_index(
        index_path,
        return_sample_meta=True,
    )
    class_names = _load_class_names(index_path=index_path, num_classes=num_classes)

    raw_splits: Dict[str, Dict[str, Any]] = {}
    for split_name in _dedupe_preserve_order(("train", *splits)):
        rel_paths, labels, wind_kt, n_fallback = _load_split_targets(
            split_name=split_name,
            split_dir=split_dir,
            class_to_files=class_to_files,
            sample_meta=sample_meta,
        )
        _validate_labels(labels, num_classes=num_classes, split_name=split_name)
        if rel_paths and n_fallback == len(rel_paths):
            raise RuntimeError(
                "Evaluator requires real per-sample wind metadata, but all "
                f"{split_name} samples fell back to class midpoints. Rebuild the "
                "dataset index with wind metadata."
            )
        raw_splits[split_name] = {
            "rel_paths": rel_paths,
            "labels": labels,
            "wind_kt": wind_kt,
            "fallbacks": int(n_fallback),
        }

    train_labels = raw_splits["train"]["labels"]
    train_wind = raw_splits["train"]["wind_kt"]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.int64)
    class_weights = compute_soft_class_weights(
        labels=train_labels,
        num_classes=num_classes,
        alpha=alpha,
    )
    wind = compute_wind_standardizer(train_wind)

    split_objs: Dict[str, EvaluatorSplit] = {}
    for split_name, raw in raw_splits.items():
        labels = raw["labels"]
        wind_kt = raw["wind_kt"]
        split_objs[split_name] = EvaluatorSplit(
            name=split_name,
            rel_paths=tuple(raw["rel_paths"]),
            labels=labels,
            wind_kt=wind_kt,
            wind_z=wind.transform(wind_kt),
            sample_weight=class_weights[labels].astype(np.float32),
            wind_metadata_fallbacks=int(raw["fallbacks"]),
        )

    return EvaluatorDataBundle(
        backend=backend,
        bt_range=bt_range,
        batch_size=batch_size,
        num_classes=num_classes,
        class_names=class_names,
        class_weights=class_weights,
        train_class_counts=class_counts,
        wind=wind,
        splits=split_objs,
    )


def make_batch_dataset(
    bundle: EvaluatorDataBundle,
    split_name: str,
    *,
    shuffle: bool,
    seed: int,
    drop_remainder: bool = False,
    sampler_config: EvaluatorSamplerConfig | None = None,
) -> NaturalEvaluatorBatchDataset:
    sampler_config = sampler_config or EvaluatorSamplerConfig()
    mode = str(sampler_config.mode).strip().lower()
    kwargs = {
        "split": bundle.split(split_name),
        "backend": bundle.backend,
        "bt_range": bundle.bt_range,
        "batch_size": bundle.batch_size,
        "num_classes": bundle.num_classes,
        "shuffle": shuffle,
        "seed": seed,
        "drop_remainder": drop_remainder,
    }
    if split_name == "train" and mode == "mild_rebalanced":
        return MildRebalancedEvaluatorBatchDataset(
            sampler_config=sampler_config,
            **kwargs,
        )
    if mode != "natural" and split_name == "train":
        raise ValueError(
            f"Unsupported evaluator sampler mode '{sampler_config.mode}'. "
            "Expected 'natural' or 'mild_rebalanced'."
        )
    return NaturalEvaluatorBatchDataset(
        **kwargs,
    )


def compute_wind_standardizer(wind_kt: np.ndarray) -> WindStandardizer:
    arr = np.asarray(wind_kt, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute wind normalization from an empty train split.")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std < 1.0e-6:
        std = 1.0
    return WindStandardizer(mean_kt=mean, std_kt=std)


def compute_soft_class_weights(
    *,
    labels: np.ndarray,
    num_classes: int,
    alpha: float,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        raise ValueError("Cannot compute class weights from an empty train split.")

    weights = np.zeros((int(num_classes),), dtype=np.float64)
    weights[present] = np.power(counts[present], -float(alpha))
    weights[present] /= np.mean(weights[present])
    return weights.astype(np.float32)


def resolve_class_weight_alpha(ev_cfg: Dict[str, Any]) -> float:
    if "class_weight_alpha" in ev_cfg:
        return float(ev_cfg["class_weight_alpha"])
    weight_cfg = ev_cfg.get("class_weights", {})
    if isinstance(weight_cfg, dict) and "alpha" in weight_cfg:
        return float(weight_cfg["alpha"])
    return 0.7


def resolve_sampler_config(ev_cfg: Dict[str, Any]) -> EvaluatorSamplerConfig:
    sampler_cfg = ev_cfg.get("sampler", {})
    if sampler_cfg is None:
        sampler_cfg = {}
    if not isinstance(sampler_cfg, dict):
        raise ValueError("evaluator.sampler must be a mapping when set.")

    mode = str(sampler_cfg.get("mode", ev_cfg.get("sampler_mode", "natural"))).strip().lower()
    default_strength = 0.35 if mode == "mild_rebalanced" else 0.0
    strength = float(
        sampler_cfg.get("strength", ev_cfg.get("sampler_strength", default_strength))
    )
    if mode not in {"natural", "mild_rebalanced"}:
        raise ValueError(
            f"Unsupported evaluator.sampler.mode={mode!r}; expected 'natural' or 'mild_rebalanced'."
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"evaluator.sampler.strength must be in [0, 1], got {strength}")
    if mode == "natural":
        strength = 0.0
    return EvaluatorSamplerConfig(mode=mode, strength=strength)


def empirical_class_probabilities(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("Cannot compute empirical class probabilities from empty counts.")
    return (counts / total).astype(np.float64)


def compute_sampler_class_probabilities(
    *,
    counts: np.ndarray,
    mode: str,
    strength: float,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0.0
    if not np.any(present):
        raise ValueError("Cannot compute sampler probabilities from empty counts.")

    empirical = empirical_class_probabilities(counts)
    mode = str(mode).strip().lower()
    if mode == "natural":
        return empirical
    if mode != "mild_rebalanced":
        raise ValueError(
            f"Unsupported sampler mode '{mode}'. Expected 'natural' or 'mild_rebalanced'."
        )

    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Sampler strength must be in [0, 1], got {strength}")
    exponent = 1.0 - strength
    probs = np.zeros_like(empirical, dtype=np.float64)
    probs[present] = np.power(empirical[present], exponent)
    probs[present] /= np.sum(probs[present])
    return probs


def sampler_description(
    *,
    mode: str,
    strength: float,
    probabilities: np.ndarray,
    class_counts: np.ndarray,
) -> Dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_counts = np.asarray(class_counts, dtype=np.int64)
    empirical = empirical_class_probabilities(class_counts)
    return {
        "mode": str(mode),
        "strength": float(strength),
        "class_probabilities": {
            str(i): float(v) for i, v in enumerate(probabilities.tolist())
        },
        "natural_class_probabilities": {
            str(i): float(v) for i, v in enumerate(empirical.tolist())
        },
        "class_probability_ratio_vs_natural": {
            str(i): (
                None
                if float(empirical[i]) <= 0.0
                else float(probabilities[i] / empirical[i])
            )
            for i in range(int(probabilities.shape[0]))
        },
    }


def build_sampler_summary(
    cfg: Dict[str, Any],
    bundle: EvaluatorDataBundle,
) -> Dict[str, Any]:
    sampler_config = resolve_sampler_config(cfg.get("evaluator", {}))
    probabilities = compute_sampler_class_probabilities(
        counts=bundle.train_class_counts,
        mode=sampler_config.mode,
        strength=sampler_config.strength,
    )
    return sampler_description(
        mode=sampler_config.mode,
        strength=sampler_config.strength,
        probabilities=probabilities,
        class_counts=bundle.train_class_counts,
    )


def bundle_summary(bundle: EvaluatorDataBundle) -> Dict[str, Any]:
    return {
        "backend": getattr(bundle.backend, "name", type(bundle.backend).__name__),
        "bt_range": [float(bundle.bt_range[0]), float(bundle.bt_range[1])],
        "batch_size": int(bundle.batch_size),
        "num_classes": int(bundle.num_classes),
        "class_names": {str(k): v for k, v in sorted(bundle.class_names.items())},
        "train_class_counts": {
            str(i): int(v) for i, v in enumerate(bundle.train_class_counts.tolist())
        },
        "class_weights": {
            str(i): float(v) for i, v in enumerate(bundle.class_weights.tolist())
        },
        "wind_standardization": bundle.wind.to_json(),
        "splits": {
            name: {
                "num_samples": split.size,
                "class_counts": {
                    str(i): int(v)
                    for i, v in enumerate(
                        split.class_counts(bundle.num_classes).tolist()
                    )
                },
                "wind_metadata_fallbacks": int(split.wind_metadata_fallbacks),
            }
            for name, split in sorted(bundle.splits.items())
        },
    }


def _load_split_targets(
    *,
    split_name: str,
    split_dir: Path,
    class_to_files: Dict[int, List[str]],
    sample_meta: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
    allowed = load_split_file_set(split_dir, split_name)
    class_to_split_files = {
        int(c): [p for p in paths if p in allowed]
        for c, paths in class_to_files.items()
    }
    rel_paths, labels_list = _build_split_pairs(class_to_split_files, allowed)
    if not rel_paths:
        raise RuntimeError(
            f"No files matched for split='{split_name}'. Check split manifests "
            "and dataset_index paths."
        )

    wind_lookup, n_fallback = build_relpath_to_wind_kt(
        rel_paths,
        labels_list,
        sample_meta,
    )
    labels = np.asarray(labels_list, dtype=np.int32)
    wind_kt = np.asarray([wind_lookup[p] for p in rel_paths], dtype=np.float32)
    return rel_paths, labels, wind_kt, int(n_fallback)


def _load_class_names(index_path: Path, num_classes: int) -> Dict[int, str]:
    defaults = {i: f"class_{i}" for i in range(int(num_classes))}
    try:
        with Path(index_path).open("r") as f:
            index = json.load(f)
    except Exception:
        return defaults

    raw = index.get("meta", {}).get("ss_definition", {})
    if not isinstance(raw, dict):
        return defaults

    out = dict(defaults)
    for key, value in raw.items():
        try:
            cls = int(key)
        except Exception:
            continue
        if 0 <= cls < int(num_classes):
            out[cls] = str(value)
    return out


def _validate_labels(labels: np.ndarray, *, num_classes: int, split_name: str):
    if labels.size == 0:
        raise RuntimeError(f"Split '{split_name}' is empty.")
    bad = labels[(labels < 0) | (labels >= int(num_classes))]
    if bad.size:
        uniq = sorted({int(x) for x in bad.tolist()})
        raise ValueError(
            f"Split '{split_name}' contains labels outside [0, {num_classes - 1}]: {uniq}"
        )


def _dedupe_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    out = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)
