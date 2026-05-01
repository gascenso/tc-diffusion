from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm.auto import tqdm

from .data import (
    build_data_backend,
    build_relpath_to_wind_kt,
    load_dataset_index,
    load_split_file_set,
    ss_class_midpoint_kt,
)
from .evaluation.metrics import DAVComputer, PolarBinner, radial_profile_batch
from .evaluation.metrics import cold_cloud_fraction, eye_contrast_proxy, psd_radial_batch


SAMPLING_GUIDANCE_BANK_SCHEMA = "tc_diffusion.sampling_guidance_bank.v2"
_DEFAULT_CACHE_ROOT = Path("outputs") / "sampling_guidance"


@dataclass(frozen=True)
class SamplingGuidanceClassTargets:
    wind_kt: np.ndarray
    radial_profiles_k: np.ndarray
    dav_deg2: np.ndarray
    hist_cdf: np.ndarray
    cold_fraction: np.ndarray
    eye_contrast_k: np.ndarray
    psd_profiles_log10: np.ndarray


@dataclass(frozen=True)
class SamplingGuidanceTargetBank:
    path: Path
    schema: str
    cache_key: str
    target_split: str
    image_size: int
    bt_min_k: float
    bt_max_k: float
    radial_bins: int
    hist_bins: int
    hist_softness_k: float
    hist_thresholds_k: np.ndarray
    cold_threshold_k: float
    eye_inner_frac: float
    eye_ring_frac: float
    psd_bins: int
    dav_radius_km: float
    pixel_size_km: float
    dav_center_region_size: int
    class_targets: Dict[int, SamplingGuidanceClassTargets]

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(class_id) for class_id in self.class_targets.keys()))


def resolve_sampling_guidance_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(cfg.get("sampling_guidance", {}))
    phys_cfg = cfg.get("physics_loss", {})
    eval_cfg = cfg.get("evaluation", {})

    out = {
        "enabled": bool(raw.get("enabled", False)),
        "target_split": str(raw.get("target_split", "train")).strip().lower(),
        "cache_path": raw.get("cache_path"),
        "build_batch_size": int(raw.get("build_batch_size", 16)),
        "neighbor_k": int(raw.get("neighbor_k", 64)),
        "target_mode": str(raw.get("target_mode", "random_neighbor")).strip().lower(),
        "guide_start_step_frac": float(raw.get("guide_start_step_frac", 0.65)),
        "guide_stop_step_frac": float(raw.get("guide_stop_step_frac", 1.0)),
        "schedule": str(raw.get("schedule", "linear")).strip().lower(),
        "step_size": float(raw.get("step_size", 0.01)),
        "inner_steps": int(raw.get("inner_steps", 1)),
        "band_width_sigma": float(raw.get("band_width_sigma", 1.0)),
        "target_pull_weight": float(raw.get("target_pull_weight", 0.25)),
        "radial_weight": float(raw.get("radial_weight", phys_cfg.get("radial_weight", 1.0))),
        "dav_weight": float(raw.get("dav_weight", phys_cfg.get("dav_weight", 0.0))),
        "hist_weight": float(raw.get("hist_weight", phys_cfg.get("hist_weight", 0.0))),
        "cold_weight": float(raw.get("cold_weight", 0.0)),
        "eye_weight": float(raw.get("eye_weight", 0.0)),
        "psd_weight": float(raw.get("psd_weight", 0.0)),
        "radial_bins": int(raw.get("radial_bins", phys_cfg.get("radial_bins", 64))),
        "hist_bins": int(raw.get("hist_bins", phys_cfg.get("hist_bins", 32))),
        "hist_softness_k": float(
            raw.get("hist_softness_k", phys_cfg.get("hist_softness_k", 2.5))
        ),
        "cold_threshold_k": float(raw.get("cold_threshold_k", 200.0)),
        "cold_softness_k": float(raw.get("cold_softness_k", 2.5)),
        "eye_inner_frac": float(raw.get("eye_inner_frac", 0.12)),
        "eye_ring_frac": float(raw.get("eye_ring_frac", 0.20)),
        "psd_bins": int(raw.get("psd_bins", eval_cfg.get("psd_bins", 96))),
        "dav_radius_km": float(
            raw.get(
                "dav_radius_km",
                eval_cfg.get("dav_radius_km", phys_cfg.get("dav_radius_km", 300.0)),
            )
        ),
        "pixel_size_km": float(
            raw.get(
                "pixel_size_km",
                eval_cfg.get("dav_pixel_size_km", phys_cfg.get("pixel_size_km", 8.0)),
            )
        ),
        "dav_center_region_size": int(
            raw.get("dav_center_region_size", eval_cfg.get("dav_center_region_size", 3))
        ),
        "sigma_floor_radial_k": float(raw.get("sigma_floor_radial_k", 2.0)),
        "sigma_floor_dav_deg2": float(raw.get("sigma_floor_dav_deg2", 100.0)),
        "sigma_floor_hist_cdf": float(raw.get("sigma_floor_hist_cdf", 0.02)),
        "sigma_floor_cold_fraction": float(raw.get("sigma_floor_cold_fraction", 0.002)),
        "sigma_floor_eye_k": float(raw.get("sigma_floor_eye_k", 1.0)),
        "sigma_floor_psd_log10": float(raw.get("sigma_floor_psd_log10", 0.10)),
    }
    _validate_sampling_guidance_cfg(out)
    return out


def sampling_guidance_summary(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sg_cfg = resolve_sampling_guidance_cfg(cfg)
    summary = {
        "enabled": bool(sg_cfg["enabled"]),
        "method": "target_bank_v2",
        "target_split": str(sg_cfg["target_split"]),
        "cache_key": sampling_guidance_cache_key(cfg),
    }
    if not bool(sg_cfg["enabled"]):
        return summary

    summary.update(
        {
            "neighbor_k": int(sg_cfg["neighbor_k"]),
            "target_mode": str(sg_cfg["target_mode"]),
            "guide_start_step_frac": float(sg_cfg["guide_start_step_frac"]),
            "guide_stop_step_frac": float(sg_cfg["guide_stop_step_frac"]),
            "schedule": str(sg_cfg["schedule"]),
            "step_size": float(sg_cfg["step_size"]),
            "inner_steps": int(sg_cfg["inner_steps"]),
            "band_width_sigma": float(sg_cfg["band_width_sigma"]),
            "target_pull_weight": float(sg_cfg["target_pull_weight"]),
            "radial_weight": float(sg_cfg["radial_weight"]),
            "dav_weight": float(sg_cfg["dav_weight"]),
            "hist_weight": float(sg_cfg["hist_weight"]),
            "cold_weight": float(sg_cfg["cold_weight"]),
            "eye_weight": float(sg_cfg["eye_weight"]),
            "psd_weight": float(sg_cfg["psd_weight"]),
            "radial_bins": int(sg_cfg["radial_bins"]),
            "hist_bins": int(sg_cfg["hist_bins"]),
            "hist_softness_k": float(sg_cfg["hist_softness_k"]),
            "cold_threshold_k": float(sg_cfg["cold_threshold_k"]),
            "cold_softness_k": float(sg_cfg["cold_softness_k"]),
            "eye_inner_frac": float(sg_cfg["eye_inner_frac"]),
            "eye_ring_frac": float(sg_cfg["eye_ring_frac"]),
            "psd_bins": int(sg_cfg["psd_bins"]),
            "dav_radius_km": float(sg_cfg["dav_radius_km"]),
            "pixel_size_km": float(sg_cfg["pixel_size_km"]),
            "dav_center_region_size": int(sg_cfg["dav_center_region_size"]),
            "sigma_floor_radial_k": float(sg_cfg["sigma_floor_radial_k"]),
            "sigma_floor_dav_deg2": float(sg_cfg["sigma_floor_dav_deg2"]),
            "sigma_floor_hist_cdf": float(sg_cfg["sigma_floor_hist_cdf"]),
            "sigma_floor_cold_fraction": float(sg_cfg["sigma_floor_cold_fraction"]),
            "sigma_floor_eye_k": float(sg_cfg["sigma_floor_eye_k"]),
            "sigma_floor_psd_log10": float(sg_cfg["sigma_floor_psd_log10"]),
        }
    )
    return summary


def sampling_guidance_cache_key(cfg: Dict[str, Any]) -> str:
    sg_cfg = resolve_sampling_guidance_cfg(cfg)
    data_cfg = cfg["data"]
    payload = {
        "schema": SAMPLING_GUIDANCE_BANK_SCHEMA,
        "target_split": str(sg_cfg["target_split"]),
        "image_size": int(data_cfg["image_size"]),
        "bt_min_k": float(data_cfg["bt_min_k"]),
        "bt_max_k": float(data_cfg["bt_max_k"]),
        "radial_bins": int(sg_cfg["radial_bins"]),
        "hist_bins": int(sg_cfg["hist_bins"]),
        "hist_softness_k": float(sg_cfg["hist_softness_k"]),
        "cold_threshold_k": float(sg_cfg["cold_threshold_k"]),
        "eye_inner_frac": float(sg_cfg["eye_inner_frac"]),
        "eye_ring_frac": float(sg_cfg["eye_ring_frac"]),
        "psd_bins": int(sg_cfg["psd_bins"]),
        "dav_radius_km": float(sg_cfg["dav_radius_km"]),
        "pixel_size_km": float(sg_cfg["pixel_size_km"]),
        "dav_center_region_size": int(sg_cfg["dav_center_region_size"]),
        "dataset_index": str(data_cfg.get("dataset_index", "")),
        "split_dir": str(data_cfg.get("split_dir", "")),
        "backend": str(data_cfg.get("backend", "")),
        "packed_manifest": str(data_cfg.get("packed_manifest", "")),
        "data_root": str(data_cfg.get("data_root", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def resolve_sampling_guidance_cache_path(cfg: Dict[str, Any]) -> Path:
    sg_cfg = resolve_sampling_guidance_cfg(cfg)
    raw = sg_cfg.get("cache_path")
    if raw is not None:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    return Path.cwd() / _DEFAULT_CACHE_ROOT / f"{sampling_guidance_cache_key(cfg)}.npz"


def load_or_build_sampling_guidance_target_bank(
    cfg: Dict[str, Any],
    *,
    show_progress: bool = False,
) -> SamplingGuidanceTargetBank:
    cache_path = resolve_sampling_guidance_cache_path(cfg)
    if cache_path.exists():
        bank = load_sampling_guidance_target_bank(cache_path)
        _validate_loaded_bank_against_cfg(bank, cfg)
        return bank

    return build_sampling_guidance_target_bank(
        cfg,
        out_path=cache_path,
        show_progress=show_progress,
    )


def load_sampling_guidance_target_bank(path: Path) -> SamplingGuidanceTargetBank:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    schema = str(data["schema"].item())
    if schema != SAMPLING_GUIDANCE_BANK_SCHEMA:
        raise ValueError(
            f"Unsupported sampling-guidance bank schema in {path}: {schema!r}. "
            f"Expected {SAMPLING_GUIDANCE_BANK_SCHEMA!r}."
        )

    class_ids = [int(v) for v in np.asarray(data["class_ids"], dtype=np.int32).tolist()]
    class_targets: Dict[int, SamplingGuidanceClassTargets] = {}
    for class_id in class_ids:
        class_targets[class_id] = SamplingGuidanceClassTargets(
            wind_kt=np.asarray(data[f"class_{class_id}_wind_kt"], dtype=np.float32),
            radial_profiles_k=np.asarray(
                data[f"class_{class_id}_radial_profiles_k"],
                dtype=np.float32,
            ),
            dav_deg2=np.asarray(data[f"class_{class_id}_dav_deg2"], dtype=np.float32),
            hist_cdf=np.asarray(data[f"class_{class_id}_hist_cdf"], dtype=np.float32),
            cold_fraction=np.asarray(
                data[f"class_{class_id}_cold_fraction"],
                dtype=np.float32,
            ),
            eye_contrast_k=np.asarray(
                data[f"class_{class_id}_eye_contrast_k"],
                dtype=np.float32,
            ),
            psd_profiles_log10=np.asarray(
                data[f"class_{class_id}_psd_profiles_log10"],
                dtype=np.float32,
            ),
        )

    return SamplingGuidanceTargetBank(
        path=path,
        schema=schema,
        cache_key=str(data["cache_key"].item()),
        target_split=str(data["target_split"].item()),
        image_size=int(np.asarray(data["image_size"]).item()),
        bt_min_k=float(np.asarray(data["bt_min_k"]).item()),
        bt_max_k=float(np.asarray(data["bt_max_k"]).item()),
        radial_bins=int(np.asarray(data["radial_bins"]).item()),
        hist_bins=int(np.asarray(data["hist_bins"]).item()),
        hist_softness_k=float(np.asarray(data["hist_softness_k"]).item()),
        hist_thresholds_k=np.asarray(data["hist_thresholds_k"], dtype=np.float32),
        cold_threshold_k=float(np.asarray(data["cold_threshold_k"]).item()),
        eye_inner_frac=float(np.asarray(data["eye_inner_frac"]).item()),
        eye_ring_frac=float(np.asarray(data["eye_ring_frac"]).item()),
        psd_bins=int(np.asarray(data["psd_bins"]).item()),
        dav_radius_km=float(np.asarray(data["dav_radius_km"]).item()),
        pixel_size_km=float(np.asarray(data["pixel_size_km"]).item()),
        dav_center_region_size=int(np.asarray(data["dav_center_region_size"]).item()),
        class_targets=class_targets,
    )


def build_sampling_guidance_target_bank(
    cfg: Dict[str, Any],
    *,
    out_path: Path,
    show_progress: bool = False,
) -> SamplingGuidanceTargetBank:
    sg_cfg = resolve_sampling_guidance_cfg(cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    bt_min_k = float(data_cfg["bt_min_k"])
    bt_max_k = float(data_cfg["bt_max_k"])
    image_size = int(data_cfg["image_size"])
    hist_bins = int(sg_cfg["hist_bins"])
    hist_edges_k = np.linspace(bt_min_k, bt_max_k, num=hist_bins + 1, dtype=np.float32)
    hist_thresholds_k = hist_edges_k[1:-1].astype(np.float32, copy=False)

    backend = build_data_backend(data_cfg)
    class_to_files, sample_meta = load_dataset_index(
        Path(data_cfg["dataset_index"]),
        return_sample_meta=True,
    )
    allowed = load_split_file_set(Path(data_cfg["split_dir"]), str(sg_cfg["target_split"]))

    file_to_class: Dict[str, int] = {}
    for class_id, rel_paths in class_to_files.items():
        for rel_path in rel_paths:
            file_to_class[str(rel_path)] = int(class_id)

    rel_paths = sorted(rel_path for rel_path in allowed if rel_path in file_to_class)
    labels = [int(file_to_class[rel_path]) for rel_path in rel_paths]
    if not rel_paths:
        raise RuntimeError(
            "Sampling-guidance target build found no usable samples after applying "
            f"split={sg_cfg['target_split']!r}."
        )

    if bool(cfg.get("conditioning", {}).get("use_wind_speed", False)):
        rel_path_to_wind_kt, _ = build_relpath_to_wind_kt(rel_paths, labels, sample_meta)
    else:
        rel_path_to_wind_kt = {
            rel_path: float(ss_class_midpoint_kt(label))
            for rel_path, label in zip(rel_paths, labels)
        }

    by_class_paths: Dict[int, list[str]] = {}
    by_class_winds: Dict[int, list[float]] = {}
    for rel_path, label in zip(rel_paths, labels):
        by_class_paths.setdefault(int(label), []).append(str(rel_path))
        by_class_winds.setdefault(int(label), []).append(float(rel_path_to_wind_kt[rel_path]))

    binner = PolarBinner(image_size, image_size, int(sg_cfg["radial_bins"]), 360)
    dav_computer = DAVComputer(
        image_size,
        image_size,
        pixel_size_km=float(sg_cfg["pixel_size_km"]),
        radius_km=float(sg_cfg["dav_radius_km"]),
        center_region_size=int(sg_cfg["dav_center_region_size"]),
    )

    class_targets: Dict[int, SamplingGuidanceClassTargets] = {}
    class_ids = sorted(int(class_id) for class_id in by_class_paths.keys())
    build_batch_size = int(sg_cfg["build_batch_size"])
    if build_batch_size <= 0:
        raise ValueError(
            f"sampling_guidance.build_batch_size must be > 0, got {build_batch_size}"
        )

    total_batches = sum(
        (len(by_class_paths[class_id]) + build_batch_size - 1) // build_batch_size
        for class_id in class_ids
    )
    pbar = None
    if bool(show_progress):
        pbar = tqdm(
            total=total_batches,
            desc="Building sampling-guidance targets",
            unit="batch",
            leave=True,
        )

    try:
        for class_id in class_ids:
            class_paths = by_class_paths[class_id]
            class_winds = np.asarray(by_class_winds[class_id], dtype=np.float32)
            radial_rows = []
            dav_rows = []
            hist_rows = []
            cold_rows = []
            eye_rows = []
            psd_rows = []

            for start in range(0, len(class_paths), build_batch_size):
                batch_paths = class_paths[start : start + build_batch_size]
                bt_batch = backend.load_bt_batch(batch_paths).astype(np.float32, copy=False)
                if bt_batch.ndim == 4 and bt_batch.shape[-1] == 1:
                    bt_batch = bt_batch[..., 0]
                if bt_batch.ndim != 3:
                    raise ValueError(
                        "Sampling-guidance target build expects backend batches with shape "
                        f"(N,H,W) or (N,H,W,1), got {bt_batch.shape}."
                    )
                bt_batch = np.nan_to_num(bt_batch, nan=bt_min_k, posinf=bt_max_k, neginf=bt_min_k)
                bt_batch = np.clip(bt_batch, bt_min_k, bt_max_k)

                mean_profiles, _ = radial_profile_batch(bt_batch, binner)
                dav_values = dav_computer.batch(bt_batch)
                hist_cdf = _soft_bt_cdf_numpy(
                    bt_batch,
                    thresholds_k=hist_thresholds_k,
                    softness_k=float(sg_cfg["hist_softness_k"]),
                )
                cold_values = cold_cloud_fraction(
                    bt_batch,
                    threshold_k=float(sg_cfg["cold_threshold_k"]),
                )
                eye_values = eye_contrast_proxy(
                    mean_profiles,
                    inner_frac=float(sg_cfg["eye_inner_frac"]),
                    ring_frac=float(sg_cfg["eye_ring_frac"]),
                )
                psd_profiles = psd_radial_batch(
                    bt_batch,
                    psd_bins=int(sg_cfg["psd_bins"]),
                )

                radial_rows.append(mean_profiles.astype(np.float32, copy=False))
                dav_rows.append(dav_values.astype(np.float32, copy=False))
                hist_rows.append(hist_cdf.astype(np.float32, copy=False))
                cold_rows.append(cold_values.astype(np.float32, copy=False))
                eye_rows.append(eye_values.astype(np.float32, copy=False))
                psd_rows.append(psd_profiles.astype(np.float32, copy=False))

                if pbar is not None:
                    pbar.set_postfix_str(f"class {class_id}", refresh=False)
                    pbar.update(1)

            class_targets[class_id] = SamplingGuidanceClassTargets(
                wind_kt=class_winds,
                radial_profiles_k=np.concatenate(radial_rows, axis=0),
                dav_deg2=np.concatenate(dav_rows, axis=0),
                hist_cdf=np.concatenate(hist_rows, axis=0),
                cold_fraction=np.concatenate(cold_rows, axis=0),
                eye_contrast_k=np.concatenate(eye_rows, axis=0),
                psd_profiles_log10=np.concatenate(psd_rows, axis=0),
            )
    finally:
        if pbar is not None:
            pbar.close()

    save_payload: Dict[str, Any] = {
        "schema": np.asarray(SAMPLING_GUIDANCE_BANK_SCHEMA),
        "cache_key": np.asarray(sampling_guidance_cache_key(cfg)),
        "target_split": np.asarray(str(sg_cfg["target_split"])),
        "image_size": np.asarray([image_size], dtype=np.int32),
        "bt_min_k": np.asarray([bt_min_k], dtype=np.float32),
        "bt_max_k": np.asarray([bt_max_k], dtype=np.float32),
        "radial_bins": np.asarray([int(sg_cfg["radial_bins"])], dtype=np.int32),
        "hist_bins": np.asarray([hist_bins], dtype=np.int32),
        "hist_softness_k": np.asarray([float(sg_cfg["hist_softness_k"])], dtype=np.float32),
        "hist_thresholds_k": hist_thresholds_k.astype(np.float32, copy=False),
        "cold_threshold_k": np.asarray([float(sg_cfg["cold_threshold_k"])], dtype=np.float32),
        "eye_inner_frac": np.asarray([float(sg_cfg["eye_inner_frac"])], dtype=np.float32),
        "eye_ring_frac": np.asarray([float(sg_cfg["eye_ring_frac"])], dtype=np.float32),
        "psd_bins": np.asarray([int(sg_cfg["psd_bins"])], dtype=np.int32),
        "dav_radius_km": np.asarray([float(sg_cfg["dav_radius_km"])], dtype=np.float32),
        "pixel_size_km": np.asarray([float(sg_cfg["pixel_size_km"])], dtype=np.float32),
        "dav_center_region_size": np.asarray(
            [int(sg_cfg["dav_center_region_size"])],
            dtype=np.int32,
        ),
        "class_ids": np.asarray(class_ids, dtype=np.int32),
    }
    for class_id, target in class_targets.items():
        save_payload[f"class_{class_id}_wind_kt"] = target.wind_kt.astype(np.float32, copy=False)
        save_payload[f"class_{class_id}_radial_profiles_k"] = target.radial_profiles_k.astype(
            np.float32,
            copy=False,
        )
        save_payload[f"class_{class_id}_dav_deg2"] = target.dav_deg2.astype(np.float32, copy=False)
        save_payload[f"class_{class_id}_hist_cdf"] = target.hist_cdf.astype(np.float32, copy=False)
        save_payload[f"class_{class_id}_cold_fraction"] = target.cold_fraction.astype(
            np.float32,
            copy=False,
        )
        save_payload[f"class_{class_id}_eye_contrast_k"] = target.eye_contrast_k.astype(
            np.float32,
            copy=False,
        )
        save_payload[f"class_{class_id}_psd_profiles_log10"] = target.psd_profiles_log10.astype(
            np.float32,
            copy=False,
        )

    np.savez_compressed(out_path, **save_payload)
    return load_sampling_guidance_target_bank(out_path)


def _soft_bt_cdf_numpy(
    bt_batch_k: np.ndarray,
    *,
    thresholds_k: np.ndarray,
    softness_k: float,
) -> np.ndarray:
    flat = np.reshape(np.asarray(bt_batch_k, dtype=np.float32), [bt_batch_k.shape[0], -1, 1])
    thresholds = np.reshape(np.asarray(thresholds_k, dtype=np.float32), [1, 1, -1])
    logits = (thresholds - flat) / max(float(softness_k), 1.0e-6)
    return np.mean(1.0 / (1.0 + np.exp(-logits)), axis=1).astype(np.float32)


def _validate_sampling_guidance_cfg(sg_cfg: Dict[str, Any]) -> None:
    split = str(sg_cfg["target_split"]).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(
            "sampling_guidance.target_split must be one of 'train', 'val', or 'test', "
            f"got {sg_cfg['target_split']!r}."
        )
    if int(sg_cfg["build_batch_size"]) <= 0:
        raise ValueError(
            "sampling_guidance.build_batch_size must be > 0, "
            f"got {sg_cfg['build_batch_size']}."
        )
    if int(sg_cfg["neighbor_k"]) <= 0:
        raise ValueError(
            "sampling_guidance.neighbor_k must be > 0, "
            f"got {sg_cfg['neighbor_k']}."
        )
    if str(sg_cfg["target_mode"]) not in {"neighbor_mean", "nearest", "random_neighbor"}:
        raise ValueError(
            "sampling_guidance.target_mode must be one of 'neighbor_mean', "
            f"'nearest', or 'random_neighbor', got {sg_cfg['target_mode']!r}."
        )
    if float(sg_cfg["guide_start_step_frac"]) < 0.0 or float(sg_cfg["guide_start_step_frac"]) > 1.0:
        raise ValueError(
            "sampling_guidance.guide_start_step_frac must be in [0, 1], "
            f"got {sg_cfg['guide_start_step_frac']}."
        )
    if float(sg_cfg["guide_stop_step_frac"]) < 0.0 or float(sg_cfg["guide_stop_step_frac"]) > 1.0:
        raise ValueError(
            "sampling_guidance.guide_stop_step_frac must be in [0, 1], "
            f"got {sg_cfg['guide_stop_step_frac']}."
        )
    if float(sg_cfg["guide_stop_step_frac"]) < float(sg_cfg["guide_start_step_frac"]):
        raise ValueError(
            "sampling_guidance.guide_stop_step_frac must be >= guide_start_step_frac, "
            f"got start={sg_cfg['guide_start_step_frac']} and stop={sg_cfg['guide_stop_step_frac']}."
        )
    if str(sg_cfg["schedule"]) not in {"linear", "cosine"}:
        raise ValueError(
            "sampling_guidance.schedule must be one of 'linear' or 'cosine', "
            f"got {sg_cfg['schedule']!r}."
        )
    if float(sg_cfg["step_size"]) < 0.0:
        raise ValueError(
            "sampling_guidance.step_size must be >= 0, "
            f"got {sg_cfg['step_size']}."
        )
    if int(sg_cfg["inner_steps"]) <= 0:
        raise ValueError(
            "sampling_guidance.inner_steps must be > 0, "
            f"got {sg_cfg['inner_steps']}."
        )
    if float(sg_cfg["band_width_sigma"]) < 0.0:
        raise ValueError(
            "sampling_guidance.band_width_sigma must be >= 0, "
            f"got {sg_cfg['band_width_sigma']}."
        )
    if float(sg_cfg["target_pull_weight"]) < 0.0:
        raise ValueError(
            "sampling_guidance.target_pull_weight must be >= 0, "
            f"got {sg_cfg['target_pull_weight']}."
        )
    if int(sg_cfg["radial_bins"]) <= 0:
        raise ValueError(
            "sampling_guidance.radial_bins must be > 0, "
            f"got {sg_cfg['radial_bins']}."
        )
    if int(sg_cfg["hist_bins"]) < 2:
        raise ValueError(
            "sampling_guidance.hist_bins must be >= 2, "
            f"got {sg_cfg['hist_bins']}."
        )
    if float(sg_cfg["hist_softness_k"]) <= 0.0:
        raise ValueError(
            "sampling_guidance.hist_softness_k must be > 0, "
            f"got {sg_cfg['hist_softness_k']}."
        )
    if float(sg_cfg["cold_threshold_k"]) <= 0.0:
        raise ValueError(
            "sampling_guidance.cold_threshold_k must be > 0, "
            f"got {sg_cfg['cold_threshold_k']}."
        )
    if float(sg_cfg["cold_softness_k"]) <= 0.0:
        raise ValueError(
            "sampling_guidance.cold_softness_k must be > 0, "
            f"got {sg_cfg['cold_softness_k']}."
        )
    for key in ("eye_inner_frac", "eye_ring_frac"):
        if float(sg_cfg[key]) <= 0.0 or float(sg_cfg[key]) >= 1.0:
            raise ValueError(f"sampling_guidance.{key} must be in (0, 1), got {sg_cfg[key]}.")
    if int(sg_cfg["psd_bins"]) <= 0:
        raise ValueError(
            "sampling_guidance.psd_bins must be > 0, "
            f"got {sg_cfg['psd_bins']}."
        )
    if float(sg_cfg["dav_radius_km"]) <= 0.0:
        raise ValueError(
            "sampling_guidance.dav_radius_km must be > 0, "
            f"got {sg_cfg['dav_radius_km']}."
        )
    if float(sg_cfg["pixel_size_km"]) <= 0.0:
        raise ValueError(
            "sampling_guidance.pixel_size_km must be > 0, "
            f"got {sg_cfg['pixel_size_km']}."
        )
    center_region_size = int(sg_cfg["dav_center_region_size"])
    if center_region_size <= 0 or (center_region_size % 2) != 1:
        raise ValueError(
            "sampling_guidance.dav_center_region_size must be a positive odd integer, "
            f"got {center_region_size}."
        )
    for key in (
        "radial_weight",
        "dav_weight",
        "hist_weight",
        "cold_weight",
        "eye_weight",
        "psd_weight",
    ):
        if float(sg_cfg[key]) < 0.0:
            raise ValueError(f"sampling_guidance.{key} must be >= 0, got {sg_cfg[key]}.")
    for key in (
        "sigma_floor_radial_k",
        "sigma_floor_dav_deg2",
        "sigma_floor_hist_cdf",
        "sigma_floor_cold_fraction",
        "sigma_floor_eye_k",
        "sigma_floor_psd_log10",
    ):
        if float(sg_cfg[key]) <= 0.0:
            raise ValueError(f"sampling_guidance.{key} must be > 0, got {sg_cfg[key]}.")


def _validate_loaded_bank_against_cfg(
    bank: SamplingGuidanceTargetBank,
    cfg: Dict[str, Any],
) -> None:
    sg_cfg = resolve_sampling_guidance_cfg(cfg)
    data_cfg = cfg["data"]
    expected_key = sampling_guidance_cache_key(cfg)
    if bank.cache_key != expected_key:
        raise ValueError(
            f"Sampling-guidance bank cache key mismatch for {bank.path}.\n"
            f"Expected: {expected_key}\n"
            f"Found:    {bank.cache_key}"
        )
    if bank.target_split != str(sg_cfg["target_split"]):
        raise ValueError(
            f"Sampling-guidance bank split mismatch for {bank.path}: "
            f"expected {sg_cfg['target_split']!r}, found {bank.target_split!r}."
        )
    if int(bank.image_size) != int(data_cfg["image_size"]):
        raise ValueError(
            f"Sampling-guidance bank image_size mismatch for {bank.path}: "
            f"expected {int(data_cfg['image_size'])}, found {bank.image_size}."
        )
    if not np.isclose(float(bank.bt_min_k), float(data_cfg["bt_min_k"])):
        raise ValueError(
            f"Sampling-guidance bank bt_min_k mismatch for {bank.path}: "
            f"expected {float(data_cfg['bt_min_k'])}, found {bank.bt_min_k}."
        )
    if not np.isclose(float(bank.bt_max_k), float(data_cfg["bt_max_k"])):
        raise ValueError(
            f"Sampling-guidance bank bt_max_k mismatch for {bank.path}: "
            f"expected {float(data_cfg['bt_max_k'])}, found {bank.bt_max_k}."
        )
    if int(bank.radial_bins) != int(sg_cfg["radial_bins"]):
        raise ValueError(
            f"Sampling-guidance bank radial_bins mismatch for {bank.path}: "
            f"expected {int(sg_cfg['radial_bins'])}, found {bank.radial_bins}."
        )
    if int(bank.hist_bins) != int(sg_cfg["hist_bins"]):
        raise ValueError(
            f"Sampling-guidance bank hist_bins mismatch for {bank.path}: "
            f"expected {int(sg_cfg['hist_bins'])}, found {bank.hist_bins}."
        )
    if not np.isclose(float(bank.hist_softness_k), float(sg_cfg["hist_softness_k"])):
        raise ValueError(
            f"Sampling-guidance bank hist_softness_k mismatch for {bank.path}: "
            f"expected {float(sg_cfg['hist_softness_k'])}, found {bank.hist_softness_k}."
        )
    if not np.isclose(float(bank.cold_threshold_k), float(sg_cfg["cold_threshold_k"])):
        raise ValueError(
            f"Sampling-guidance bank cold_threshold_k mismatch for {bank.path}: "
            f"expected {float(sg_cfg['cold_threshold_k'])}, found {bank.cold_threshold_k}."
        )
    if not np.isclose(float(bank.eye_inner_frac), float(sg_cfg["eye_inner_frac"])):
        raise ValueError(
            f"Sampling-guidance bank eye_inner_frac mismatch for {bank.path}: "
            f"expected {float(sg_cfg['eye_inner_frac'])}, found {bank.eye_inner_frac}."
        )
    if not np.isclose(float(bank.eye_ring_frac), float(sg_cfg["eye_ring_frac"])):
        raise ValueError(
            f"Sampling-guidance bank eye_ring_frac mismatch for {bank.path}: "
            f"expected {float(sg_cfg['eye_ring_frac'])}, found {bank.eye_ring_frac}."
        )
    if int(bank.psd_bins) != int(sg_cfg["psd_bins"]):
        raise ValueError(
            f"Sampling-guidance bank psd_bins mismatch for {bank.path}: "
            f"expected {int(sg_cfg['psd_bins'])}, found {bank.psd_bins}."
        )
    if not np.isclose(float(bank.dav_radius_km), float(sg_cfg["dav_radius_km"])):
        raise ValueError(
            f"Sampling-guidance bank dav_radius_km mismatch for {bank.path}: "
            f"expected {float(sg_cfg['dav_radius_km'])}, found {bank.dav_radius_km}."
        )
    if not np.isclose(float(bank.pixel_size_km), float(sg_cfg["pixel_size_km"])):
        raise ValueError(
            f"Sampling-guidance bank pixel_size_km mismatch for {bank.path}: "
            f"expected {float(sg_cfg['pixel_size_km'])}, found {bank.pixel_size_km}."
        )
    if int(bank.dav_center_region_size) != int(sg_cfg["dav_center_region_size"]):
        raise ValueError(
            f"Sampling-guidance bank dav_center_region_size mismatch for {bank.path}: "
            f"expected {int(sg_cfg['dav_center_region_size'])}, "
            f"found {bank.dav_center_region_size}."
        )
