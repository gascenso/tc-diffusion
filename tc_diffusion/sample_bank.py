from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


SAMPLE_BANK_SCHEMA = "tc_diffusion.sample_bank.v2"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_outputs_root(base_repo_root: Path, run_name: str) -> Path:
    return Path(base_repo_root) / "outputs" / str(run_name)


def model_eval_root(base_repo_root: Path, run_name: str) -> Path:
    return model_outputs_root(base_repo_root, run_name) / "eval"


def model_paper_ready_root(base_repo_root: Path, run_name: str) -> Path:
    return model_outputs_root(base_repo_root, run_name) / "paper_ready"


def model_sample_banks_root(base_repo_root: Path, run_name: str, split: str | None = None) -> Path:
    root = model_outputs_root(base_repo_root, run_name) / "sample_banks"
    if split is not None:
        root = root / _normalize_split(split)
    return root


def sample_bank_dir(base_repo_root: Path, run_name: str, split: str, bank_name: str) -> Path:
    return model_sample_banks_root(base_repo_root, run_name, split=split) / str(bank_name).strip()


def resolve_sample_bank_dir(base_repo_root: Path, run_name: str, split: str, bank_ref: str) -> Path:
    ref = Path(str(bank_ref).strip())
    if not str(ref):
        raise ValueError("Sample bank reference cannot be empty.")

    if ref.is_absolute():
        candidate = ref
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Sample bank directory not found: {candidate}")

    split = _normalize_split(split)
    outputs_root = model_outputs_root(base_repo_root, run_name)
    candidates = [
        outputs_root / "sample_banks" / split / ref,
        outputs_root / ref,
    ]
    if ref.parts and ref.parts[0] == "sample_banks":
        candidates.append(outputs_root / ref)
    if ref.parts and ref.parts[0] == split:
        candidates.append(outputs_root / "sample_banks" / ref)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(
        f"Could not locate sample bank {bank_ref!r} for run {run_name!r} on split {split!r}. Tried:\n{tried}"
    )


def format_float_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def default_sample_bank_name(
    *,
    guidance_scale: float,
    sampler: str,
    sampling_steps: int | None,
    timestep_schedule: str | None,
    n_per_class: int,
    seed: int,
    sampling_guidance: Dict[str, Any] | None = None,
) -> str:
    parts = [
        f"g{format_float_token(guidance_scale)}",
        str(sampler).strip().lower(),
        f"s{sampling_steps if sampling_steps is not None else 'full'}",
        str(timestep_schedule).strip().lower() if timestep_schedule else "schedule_default",
        f"n{int(n_per_class)}",
        f"seed{int(seed)}",
        _sampling_guidance_token(sampling_guidance),
    ]
    return "_".join(parts)


def _sampling_guidance_token(sampling_guidance: Dict[str, Any] | None) -> str:
    if not isinstance(sampling_guidance, dict) or not bool(sampling_guidance.get("enabled", False)):
        return "sgoff"

    payload = json.dumps(sampling_guidance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return f"sgon{digest}"


def repo_relative_or_abs(path: Path, base_repo_root: Path) -> str:
    path = Path(path)
    base_repo_root = Path(base_repo_root)
    try:
        return path.relative_to(base_repo_root).as_posix()
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SampleBankShard:
    part_index: int
    count: int
    bt_path: Path
    wind_path: Path | None = None
    wind_kt_mean: float | None = None


@dataclass(frozen=True)
class SampleBank:
    root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    run_name: str
    split: str
    name: str
    class_ids: tuple[int, ...]
    generated_counts_by_class: Dict[int, int]
    shards_by_class: Dict[int, tuple[SampleBankShard, ...]]
    generation: Dict[str, Any]
    conditioning: Dict[str, Any]
    conditioning_targets: Dict[str, Any]
    bt_dtype: str
    image_shape: tuple[int, int]
    total_n_per_class: int
    next_part_index: int
    suggested_next_seed: int
    suggested_next_real_seed: int

    @classmethod
    def from_dir(cls, root: Path) -> "SampleBank":
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Sample bank manifest not found: {manifest_path}")

        with manifest_path.open("r") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise ValueError(f"Sample bank manifest must be a JSON object: {manifest_path}")

        schema = str(manifest.get("schema", ""))
        if schema != SAMPLE_BANK_SCHEMA:
            raise ValueError(
                f"Unsupported sample bank schema in {manifest_path}: {schema!r}. "
                f"Expected {SAMPLE_BANK_SCHEMA!r}."
            )

        run_name = str(manifest.get("run_name") or "")
        split = _normalize_split(manifest.get("split", ""))
        if not run_name:
            raise ValueError(f"Sample bank manifest is missing run_name: {manifest_path}")

        shards_raw = manifest.get("shards_by_class")
        if not isinstance(shards_raw, dict) or not shards_raw:
            raise ValueError(f"Sample bank manifest is missing shards_by_class: {manifest_path}")

        class_ids = []
        shards_by_class: Dict[int, tuple[SampleBankShard, ...]] = {}
        generated_counts_by_class: Dict[int, int] = {}
        for raw_class_id, raw_shards in shards_raw.items():
            class_id = int(raw_class_id)
            class_ids.append(class_id)
            if not isinstance(raw_shards, list) or not raw_shards:
                raise ValueError(
                    f"Sample bank manifest shards_by_class[{class_id!r}] must be a non-empty list: {manifest_path}"
                )
            parsed_shards = []
            total_count = 0
            for raw_shard in raw_shards:
                if not isinstance(raw_shard, dict):
                    raise ValueError(
                        f"Sample bank manifest shard for class {class_id!r} must be an object: {manifest_path}"
                    )
                part_index = int(raw_shard.get("part_index"))
                count = int(raw_shard.get("count"))
                bt_rel = raw_shard.get("bt_k")
                if not bt_rel:
                    raise ValueError(
                        f"Sample bank manifest shard for class {class_id!r} is missing bt_k: {manifest_path}"
                    )
                wind_rel = raw_shard.get("wind_kt")
                parsed_shards.append(
                    SampleBankShard(
                        part_index=part_index,
                        count=count,
                        bt_path=root / str(bt_rel),
                        wind_path=None if not wind_rel else root / str(wind_rel),
                        wind_kt_mean=(
                            None
                            if raw_shard.get("wind_kt_mean") is None
                            else float(raw_shard.get("wind_kt_mean"))
                        ),
                    )
                )
                total_count += count
            parsed_shards = sorted(parsed_shards, key=lambda shard: shard.part_index)
            shards_by_class[class_id] = tuple(parsed_shards)
            generated_counts_by_class[class_id] = int(total_count)

        generation = manifest.get("generation", {})
        if not isinstance(generation, dict):
            generation = {}
        conditioning = manifest.get("conditioning", {})
        if not isinstance(conditioning, dict):
            conditioning = {}
        conditioning_targets = manifest.get("conditioning_targets", {})
        if not isinstance(conditioning_targets, dict):
            conditioning_targets = {}

        image_shape_raw = manifest.get("image_shape")
        if (
            not isinstance(image_shape_raw, list)
            or len(image_shape_raw) != 2
            or any(int(dim) <= 0 for dim in image_shape_raw)
        ):
            raise ValueError(f"Sample bank manifest has invalid image_shape: {manifest_path}")
        image_shape = (int(image_shape_raw[0]), int(image_shape_raw[1]))

        total_n_per_class = int(manifest.get("total_n_per_class") or min(generated_counts_by_class.values()))
        next_part_index = int(manifest.get("next_part_index") or max(_all_part_indices(shards_by_class), default=-1) + 1)
        suggested_next_seed = int(manifest.get("suggested_next_seed", 123))
        suggested_next_real_seed = int(manifest.get("suggested_next_real_seed", 123))

        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest=manifest,
            run_name=run_name,
            split=split,
            name=str(manifest.get("bank_name") or root.name),
            class_ids=tuple(sorted(class_ids)),
            generated_counts_by_class=generated_counts_by_class,
            shards_by_class=shards_by_class,
            generation=generation,
            conditioning=conditioning,
            conditioning_targets=conditioning_targets,
            bt_dtype=str(manifest.get("bt_dtype", "float32")),
            image_shape=image_shape,
            total_n_per_class=total_n_per_class,
            next_part_index=next_part_index,
            suggested_next_seed=suggested_next_seed,
            suggested_next_real_seed=suggested_next_real_seed,
        )

    @property
    def use_wind_speed(self) -> bool:
        return bool(self.conditioning.get("use_wind_speed", False))

    def available_n_per_class(self) -> int:
        if not self.generated_counts_by_class:
            return 0
        return int(min(self.generated_counts_by_class.values()))

    def load_bt_k(self, class_id: int, *, limit: int | None = None, mmap_mode: str | None = "r") -> np.ndarray:
        class_id = int(class_id)
        return self._load_sharded_array(class_id, kind="bt", limit=limit, mmap_mode=mmap_mode)

    def load_wind_kt(self, class_id: int, *, limit: int | None = None, mmap_mode: str | None = "r") -> np.ndarray | None:
        class_id = int(class_id)
        if not self.use_wind_speed:
            return None
        return self._load_sharded_array(class_id, kind="wind", limit=limit, mmap_mode=mmap_mode)

    def _load_sharded_array(
        self,
        class_id: int,
        *,
        kind: str,
        limit: int | None,
        mmap_mode: str | None,
    ) -> np.ndarray:
        shards = self.shards_by_class.get(int(class_id))
        if not shards:
            raise KeyError(f"Class {class_id} is not present in sample bank {self.name!r}.")

        available = self.generated_counts_by_class[int(class_id)]
        if limit is None:
            limit = available
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"Sample-bank limit must be > 0, got {limit}")
        if limit > available:
            raise ValueError(
                f"Sample bank {self.name!r} only has {available} samples for class {class_id}, "
                f"cannot take limit={limit}."
            )

        chunks = []
        remaining = limit
        for shard in shards:
            if remaining <= 0:
                break
            path = shard.bt_path if kind == "bt" else shard.wind_path
            if path is None:
                raise ValueError(
                    f"Sample bank {self.name!r} has no {kind} shard for class {class_id}, part {shard.part_index}."
                )
            arr = np.load(path, mmap_mode=mmap_mode)
            if kind == "bt" and arr.ndim != 3:
                raise ValueError(f"Expected BT shard {path} to be rank-3, found shape {arr.shape}.")
            if kind == "wind" and arr.ndim != 1:
                raise ValueError(f"Expected wind shard {path} to be rank-1, found shape {arr.shape}.")

            take = min(remaining, int(arr.shape[0]))
            view = arr[:take]
            chunks.append(view)
            remaining -= take

        if remaining != 0:
            raise ValueError(
                f"Sample bank {self.name!r} could not satisfy limit={limit} for class {class_id}; "
                f"{remaining} samples are missing across the listed shards."
            )

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)

    def load_generated_bt_k_by_class(
        self,
        *,
        limit_per_class: int | None = None,
        mmap_mode: str | None = "r",
    ) -> Dict[int, np.ndarray]:
        limit = self.available_n_per_class() if limit_per_class is None else int(limit_per_class)
        return {
            class_id: self.load_bt_k(class_id, limit=limit, mmap_mode=mmap_mode)
            for class_id in self.class_ids
        }


def write_sample_bank_manifest(root: Path, manifest: Dict[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _all_part_indices(shards_by_class: Dict[int, tuple[SampleBankShard, ...]]) -> list[int]:
    out = []
    for shards in shards_by_class.values():
        out.extend(shard.part_index for shard in shards)
    return out


def _normalize_split(split: Any) -> str:
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Expected split 'val' or 'test', got {split!r}.")
    return split
