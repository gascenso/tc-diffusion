from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tc_diffusion.config import load_config
from tc_diffusion.diffusion import Diffusion, resolve_sampling_timestep_schedule
from tc_diffusion.evaluation.evaluator import _resolve_eval_wind_target_kt, _select_real_by_class
from tc_diffusion.evaluation.metrics import denorm_bt
from tc_diffusion.model_unet import build_unet
from tc_diffusion.sample_bank import (
    SAMPLE_BANK_SCHEMA,
    SampleBank,
    default_sample_bank_name,
    repo_relative_or_abs,
    repo_root,
    sample_bank_dir,
    utc_now_iso,
    write_sample_bank_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or append to a split-specific reusable sample bank for a trained model. "
            "Each invocation writes one shard per class, so banks can grow incrementally."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to runs/<name>/config.yaml saved at training time.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--name", type=str, required=True, help="Run name under runs/ to load weights from.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Reference split used for wind-conditioning schedules. Default is test.",
    )
    parser.add_argument(
        "--bank_name",
        type=str,
        default=None,
        help="Optional sample-bank folder name. Defaults to a name derived from the sampling settings.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional explicit sample-bank directory. Defaults to outputs/<name>/sample_banks/<split>/<bank_name>.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append a new shard to an existing bank instead of creating a fresh one.",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=500,
        help="Number of generated samples to add per class in this invocation. Default is 500.",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=None,
        help="Generation batch size. Defaults to evaluation.gen_batch_size or data.batch_size.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Defaults to evaluation.guidance_scale from the saved config.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["ddpm", "ddim", "dpmpp_2m"],
        help="Sampler to use. Defaults to evaluation.sampler from the saved config.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Number of reverse-process sampling steps. Defaults to evaluation.sampling_steps.",
    )
    parser.add_argument(
        "--timestep_schedule",
        type=str,
        default=None,
        choices=["linear", "leading", "trailing"],
        help="Reduced-step timestep schedule. Defaults to evaluation.timestep_schedule.",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="DDIM eta value. Ignored by other samplers. Defaults to evaluation.ddim_eta.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling RNG seed. Defaults to evaluation.seed for a new bank, or next suggested seed when appending.",
    )
    parser.add_argument(
        "--real_seed",
        type=int,
        default=None,
        help=(
            "Seed used to select split-specific reference wind schedules. Defaults to evaluation.real_seed for a new "
            "bank, or next suggested real seed when appending."
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, require EMA weights instead of preferring EMA automatically.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1],
        default=1,
        help="Verbosity level. Use 1 to show progress bars (default) or 0 for quiet mode.",
    )
    return parser.parse_args()


def configure_runtime(cfg):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    use_mixed_precision = bool(cfg.get("training", {}).get("mixed_precision", False))
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[sample_bank] Mixed precision enabled (mixed_float16)")


def _legacy_model_cfg(cfg):
    legacy_cfg = deepcopy(cfg)
    legacy_cfg.setdefault("model", {})
    legacy_cfg["model"]["decoder_skip_mode"] = "per_level"
    legacy_cfg["model"]["output_head_pre_norm"] = False
    return legacy_cfg


def _load_model_with_arch_fallback(cfg, weights_path: Path):
    model_cfg = cfg.get("model", {})
    has_explicit_arch_keys = (
        "decoder_skip_mode" in model_cfg or "output_head_pre_norm" in model_cfg
    )

    attempts = [("saved config", cfg)]
    if not has_explicit_arch_keys:
        attempts.append(("legacy checkpoint compatibility mode", _legacy_model_cfg(cfg)))

    errors = []
    last_exc = None
    for label, cfg_variant in attempts:
        tf.keras.backend.clear_session()
        model = build_unet(cfg_variant)
        try:
            model.load_weights(str(weights_path))
            if label != "saved config":
                print(f"[sample_bank] Loaded weights using {label}.")
            return model
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            last_exc = exc

    msg = "Could not load checkpoint with any known architecture variant:\n"
    msg += "\n".join(f"  - {err}" for err in errors)
    raise ValueError(msg) from last_exc


def _resolve_weights_path(run_dir: Path, *, require_ema: bool) -> Path:
    if require_ema:
        candidates = [
            run_dir / "weights_ema_best_val.weights.h5",
            run_dir / "weights_ema_best.weights.h5",
        ]
    else:
        candidates = [
            run_dir / "weights_ema_best_val.weights.h5",
            run_dir / "weights_ema_best.weights.h5",
            run_dir / "weights_best_val.weights.h5",
            run_dir / "weights_best.weights.h5",
        ]

    for candidate in candidates:
        if candidate.exists():
            if not require_ema and "weights_best" in candidate.name and "ema" not in candidate.name:
                print(f"[sample_bank] EMA checkpoint not found; using non-EMA weights: {candidate.name}")
            return candidate

    raise FileNotFoundError(f"Could not find weights. Tried: {[str(p) for p in candidates]}")


def _canonical_generation_settings(
    *,
    guidance_scale: float,
    sampler: str,
    sampling_steps: int | None,
    timestep_schedule: str,
    ddim_eta: float,
) -> Dict[str, object]:
    return {
        "guidance_scale": float(guidance_scale),
        "sampler": str(sampler).strip().lower(),
        "sampling_steps": None if sampling_steps is None else int(sampling_steps),
        "timestep_schedule": str(timestep_schedule).strip().lower(),
        "ddim_eta": float(ddim_eta),
    }


def _build_new_manifest(
    *,
    bank_name: str,
    run_name: str,
    split: str,
    config_path: Path,
    weights_path: Path,
    generation_settings: Dict[str, object],
    use_wind_speed: bool,
    image_size: int,
    reference_counts_by_class: Dict[str, int],
    part_index: int,
    seed: int,
    real_seed: int,
    gen_batch_size: int,
    files_by_class: Dict[str, Dict[str, object]],
    generated_counts_by_class: Dict[str, int],
    conditioning_targets: Dict[int, float],
    repo: Path,
) -> Dict[str, object]:
    created_at = utc_now_iso()
    top_level_targets = (
        {"wind_kt_by_class": {str(k): float(v) for k, v in sorted(conditioning_targets.items())}}
        if conditioning_targets
        else {}
    )
    return {
        "schema": SAMPLE_BANK_SCHEMA,
        "schema_version": 2,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "run_name": run_name,
        "bank_name": bank_name,
        "split": split,
        "bt_dtype": "float32",
        "image_shape": [int(image_size), int(image_size)],
        "config_path": repo_relative_or_abs(config_path, repo),
        "weights_path": repo_relative_or_abs(weights_path, repo),
        "reference_counts_by_class": reference_counts_by_class,
        "generated_counts_by_class": generated_counts_by_class,
        "total_n_per_class": int(min(generated_counts_by_class.values())),
        "generation": generation_settings,
        "conditioning": {
            "use_wind_speed": bool(use_wind_speed),
        },
        "conditioning_targets": top_level_targets,
        "shards_by_class": {
            class_id: [dict(file_entry)]
            for class_id, file_entry in sorted(files_by_class.items())
        },
        "append_history": [
            {
                "part_index": int(part_index),
                "created_at_utc": created_at,
                "n_per_class_added": int(min(generated_counts_by_class.values())),
                "seed": int(seed),
                "real_seed": int(real_seed),
                "gen_batch_size": int(gen_batch_size),
                "files_by_class": {
                    class_id: dict(file_entry)
                    for class_id, file_entry in sorted(files_by_class.items())
                },
            }
        ],
        "next_part_index": int(part_index + 1),
        "suggested_next_seed": int(seed + 1),
        "suggested_next_real_seed": int(real_seed + 1),
    }


def _validate_append_settings(
    *,
    existing_bank: SampleBank,
    run_name: str,
    split: str,
    generation_settings: Dict[str, object],
    use_wind_speed: bool,
    image_size: int,
    weights_path: Path,
    repo: Path,
) -> None:
    if existing_bank.run_name != run_name:
        raise ValueError(
            f"Bank {existing_bank.root} belongs to run {existing_bank.run_name!r}, not {run_name!r}."
        )
    if existing_bank.split != split:
        raise ValueError(
            f"Bank {existing_bank.root} targets split {existing_bank.split!r}, not {split!r}."
        )
    if tuple(existing_bank.image_shape) != (int(image_size), int(image_size)):
        raise ValueError(
            f"Bank {existing_bank.root} uses image_shape={existing_bank.image_shape}, "
            f"but the current config uses {(int(image_size), int(image_size))}."
        )
    if bool(existing_bank.use_wind_speed) != bool(use_wind_speed):
        raise ValueError(
            f"Bank {existing_bank.root} uses use_wind_speed={existing_bank.use_wind_speed}, "
            f"but the current config uses {bool(use_wind_speed)}."
        )
    if existing_bank.generation != generation_settings:
        raise ValueError(
            f"Bank {existing_bank.root} was generated with different sampling settings.\n"
            f"Existing: {existing_bank.generation}\n"
            f"Current:  {generation_settings}"
        )
    existing_weights = str(existing_bank.manifest.get("weights_path", ""))
    current_weights = repo_relative_or_abs(weights_path, repo)
    if existing_weights and existing_weights != current_weights:
        raise ValueError(
            f"Bank {existing_bank.root} was generated from weights {existing_weights!r}, "
            f"but the current run resolves to {current_weights!r}."
        )


if __name__ == "__main__":
    args = parse_args()

    repo = repo_root()
    run_dir = repo / "runs" / args.name
    config_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Pass --config explicitly or ensure the run has a saved config.yaml."
        )

    cfg = load_config(str(config_path), overrides=args.override)
    configure_runtime(cfg)

    ev_cfg = dict(cfg.get("evaluation", {}))
    split = str(args.split).strip().lower()
    n_per_class = int(args.n_per_class)
    if n_per_class <= 0:
        raise ValueError(f"--n_per_class must be > 0, got {n_per_class}")

    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(ev_cfg.get("guidance_scale", 0.0))
    )
    sampler = str(args.sampler or ev_cfg.get("sampler", "dpmpp_2m"))
    sampling_steps = (
        int(args.sampling_steps)
        if args.sampling_steps is not None
        else ev_cfg.get("sampling_steps", ev_cfg.get("ddim_steps", None))
    )
    timestep_schedule = str(
        args.timestep_schedule or ev_cfg.get("timestep_schedule") or resolve_sampling_timestep_schedule(cfg)
    )
    ddim_eta = float(args.ddim_eta) if args.ddim_eta is not None else float(ev_cfg.get("ddim_eta", 0.0))
    generation_settings = _canonical_generation_settings(
        guidance_scale=guidance_scale,
        sampler=sampler,
        sampling_steps=sampling_steps,
        timestep_schedule=timestep_schedule,
        ddim_eta=ddim_eta,
    )

    gen_batch_size = args.gen_batch_size
    if gen_batch_size is None:
        gen_batch_size = ev_cfg.get("gen_batch_size")
    if gen_batch_size is None:
        gen_batch_size = cfg["data"].get("batch_size", n_per_class)
    gen_batch_size = int(gen_batch_size)
    if gen_batch_size <= 0:
        raise ValueError(f"--gen_batch_size must be > 0, got {gen_batch_size}")
    gen_batch_size = min(gen_batch_size, n_per_class)

    seed = int(args.seed) if args.seed is not None else None
    real_seed = int(args.real_seed) if args.real_seed is not None else None

    default_seed = int(ev_cfg.get("seed", 123))
    default_real_seed = int(ev_cfg.get("real_seed", 123))

    bank_name = args.bank_name or default_sample_bank_name(
        guidance_scale=guidance_scale,
        sampler=sampler,
        sampling_steps=sampling_steps,
        timestep_schedule=timestep_schedule,
        n_per_class=n_per_class,
        seed=seed if seed is not None else default_seed,
    )
    if args.out_dir is None:
        bank_root = sample_bank_dir(repo, args.name, split, bank_name)
    else:
        bank_root = Path(args.out_dir)
        if not bank_root.is_absolute():
            bank_root = repo / bank_root

    existing_bank = None
    if bank_root.exists() and (bank_root / "manifest.json").exists():
        if not args.append:
            raise FileExistsError(
                f"Sample bank already exists at {bank_root}. Use --append to add another shard."
            )
        existing_bank = SampleBank.from_dir(bank_root)
    elif bank_root.exists() and any(bank_root.iterdir()):
        raise FileExistsError(
            f"Sample-bank output directory already exists and is not empty: {bank_root}\n"
            "Choose a new --bank_name or clear the directory first."
        )
    elif args.append:
        raise FileNotFoundError(
            f"--append was requested, but no existing sample bank was found at {bank_root}."
        )

    weights_path = _resolve_weights_path(run_dir, require_ema=bool(args.use_ema))
    model = _load_model_with_arch_fallback(cfg, weights_path)
    diffusion = Diffusion(cfg)

    if existing_bank is not None:
        _validate_append_settings(
            existing_bank=existing_bank,
            run_name=args.name,
            split=split,
            generation_settings=generation_settings,
            use_wind_speed=bool(cfg.get("conditioning", {}).get("use_wind_speed", False)),
            image_size=int(cfg["data"]["image_size"]),
            weights_path=weights_path,
            repo=repo,
        )
        if seed is None:
            seed = int(existing_bank.suggested_next_seed)
        if real_seed is None:
            real_seed = int(existing_bank.suggested_next_real_seed)
        part_index = int(existing_bank.next_part_index)
    else:
        if seed is None:
            seed = default_seed
        if real_seed is None:
            real_seed = default_real_seed
        part_index = 0
        bank_root.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_cfg = cfg["data"]
    bt_min_k = float(data_cfg["bt_min_k"])
    bt_max_k = float(data_cfg["bt_max_k"])
    image_size = int(data_cfg["image_size"])
    num_classes = int(cfg["conditioning"]["num_ss_classes"])
    use_wind_speed = bool(cfg.get("conditioning", {}).get("use_wind_speed", False))

    reference_paths_by_class, reference_winds_by_class = _select_real_by_class(
        cfg,
        n_per_class=n_per_class,
        seed=real_seed,
        split=split,
        use_all=True,
    )

    sampling_timesteps = diffusion.get_sampling_timesteps(
        sampler=sampler,
        num_sampling_steps=sampling_steps,
        timestep_schedule=timestep_schedule,
    )
    batches_per_class = (n_per_class + gen_batch_size - 1) // gen_batch_size

    wind_targets_kt: Dict[int, float] = {}
    reference_counts_by_class = {
        str(class_id): int(len(paths))
        for class_id, paths in sorted(reference_paths_by_class.items())
    }
    shard_files_by_class: Dict[str, Dict[str, object]] = {}
    generated_counts_this_append: Dict[str, int] = {}

    sampling_pbar = None
    if bool(args.verbose):
        mode_label = "Appending shard" if existing_bank is not None else "Writing new bank"
        print(f"[sample_bank] {mode_label} for {args.name!r} at {bank_root}.")
        print(
            "[sample_bank] Progress plan: "
            f"{num_classes} classes x {batches_per_class} batches/class "
            f"x {len(sampling_timesteps)} reverse steps = "
            f"{num_classes * batches_per_class * len(sampling_timesteps)} updates."
        )
        print(
            f"[sample_bank] This invocation will add {n_per_class} samples/class as shard part {part_index:04d} "
            f"(seed={seed}, real_seed={real_seed})."
        )
        sampling_pbar = tqdm(
            total=num_classes * batches_per_class * len(sampling_timesteps),
            desc="Sample bank: generating",
            unit="step",
            leave=True,
        )

    for class_id in range(num_classes):
        if use_wind_speed:
            real_winds_c = reference_winds_by_class.get(class_id)
            if real_winds_c is not None and len(real_winds_c) > 0:
                if len(real_winds_c) >= n_per_class:
                    wind_schedule = np.asarray(real_winds_c[:n_per_class], dtype=np.float32)
                else:
                    rng_w = np.random.default_rng(seed + class_id)
                    idx = rng_w.choice(len(real_winds_c), size=n_per_class, replace=True)
                    wind_schedule = np.asarray(real_winds_c[idx], dtype=np.float32)
                wind_targets_kt[class_id] = float(wind_schedule.mean())
            else:
                wind_schedule = np.full(
                    n_per_class,
                    _resolve_eval_wind_target_kt(cfg, class_id),
                    dtype=np.float32,
                )
                wind_targets_kt[class_id] = float(wind_schedule[0])
        else:
            wind_schedule = None

        class_bt = np.empty((n_per_class, image_size, image_size), dtype=np.float32)
        offset = 0
        batch_idx = 0
        remaining = n_per_class
        while remaining > 0:
            bsz = min(gen_batch_size, remaining)
            wind_batch = wind_schedule[offset:offset + bsz] if wind_schedule is not None else None
            if sampling_pbar is not None:
                sampling_pbar.set_postfix_str(
                    f"class {class_id} batch {batch_idx + 1}/{batches_per_class}",
                    refresh=False,
                )
            raw_norm = diffusion.sample(
                model=model,
                batch_size=bsz,
                image_size=image_size,
                cond_value=class_id,
                wind_value_kt=wind_batch,
                guidance_scale=guidance_scale,
                sampler=sampler,
                num_sampling_steps=sampling_steps,
                timestep_schedule=timestep_schedule,
                ddim_eta=ddim_eta,
                show_progress=bool(args.verbose and sampling_pbar is None),
                return_both=False,
                progress_callback=sampling_pbar.update if sampling_pbar is not None else None,
            )
            class_bt[offset:offset + bsz] = denorm_bt(raw_norm.numpy(), bt_min_k, bt_max_k)[..., 0].astype(
                np.float32,
                copy=False,
            )
            offset += bsz
            remaining -= bsz
            batch_idx += 1

        part_token = f"part{part_index:04d}"
        class_bt_path = bank_root / f"class_{class_id}_bt_k_{part_token}.npy"
        np.save(class_bt_path, class_bt, allow_pickle=False)
        file_entry: Dict[str, object] = {
            "part_index": int(part_index),
            "count": int(class_bt.shape[0]),
            "bt_k": class_bt_path.name,
        }
        if wind_schedule is not None:
            wind_path = bank_root / f"class_{class_id}_wind_kt_{part_token}.npy"
            np.save(wind_path, np.asarray(wind_schedule, dtype=np.float32), allow_pickle=False)
            file_entry["wind_kt"] = wind_path.name
            file_entry["wind_kt_mean"] = float(np.mean(wind_schedule))
        shard_files_by_class[str(class_id)] = file_entry
        generated_counts_this_append[str(class_id)] = int(class_bt.shape[0])

    if sampling_pbar is not None:
        sampling_pbar.close()

    if existing_bank is None:
        manifest = _build_new_manifest(
            bank_name=bank_name,
            run_name=args.name,
            split=split,
            config_path=config_path,
            weights_path=weights_path,
            generation_settings=generation_settings,
            use_wind_speed=use_wind_speed,
            image_size=image_size,
            reference_counts_by_class=reference_counts_by_class,
            part_index=part_index,
            seed=seed,
            real_seed=real_seed,
            gen_batch_size=gen_batch_size,
            files_by_class=shard_files_by_class,
            generated_counts_by_class=generated_counts_this_append,
            conditioning_targets=wind_targets_kt,
            repo=repo,
        )
    else:
        manifest = deepcopy(existing_bank.manifest)
        manifest["updated_at_utc"] = utc_now_iso()
        manifest["reference_counts_by_class"] = reference_counts_by_class

        shards_by_class = manifest.get("shards_by_class", {})
        if not isinstance(shards_by_class, dict):
            raise ValueError(f"Invalid shards_by_class in existing bank manifest: {bank_root / 'manifest.json'}")
        for class_id, file_entry in sorted(shard_files_by_class.items()):
            shards_by_class.setdefault(class_id, [])
            shards_by_class[class_id].append(dict(file_entry))
        manifest["shards_by_class"] = shards_by_class

        generated_counts_by_class = manifest.get("generated_counts_by_class", {})
        if not isinstance(generated_counts_by_class, dict):
            generated_counts_by_class = {}
        for class_id, added_count in sorted(generated_counts_this_append.items()):
            generated_counts_by_class[class_id] = int(generated_counts_by_class.get(class_id, 0)) + int(added_count)
        manifest["generated_counts_by_class"] = generated_counts_by_class
        manifest["total_n_per_class"] = int(min(int(v) for v in generated_counts_by_class.values()))

        append_history = manifest.get("append_history", [])
        if not isinstance(append_history, list):
            append_history = []
        append_history.append(
            {
                "part_index": int(part_index),
                "created_at_utc": utc_now_iso(),
                "n_per_class_added": int(n_per_class),
                "seed": int(seed),
                "real_seed": int(real_seed),
                "gen_batch_size": int(gen_batch_size),
                "files_by_class": {
                    class_id: dict(file_entry)
                    for class_id, file_entry in sorted(shard_files_by_class.items())
                },
            }
        )
        manifest["append_history"] = append_history
        manifest["next_part_index"] = int(part_index + 1)
        manifest["suggested_next_seed"] = int(seed + 1)
        manifest["suggested_next_real_seed"] = int(real_seed + 1)

        if wind_targets_kt:
            cumulative_targets: Dict[str, float] = {}
            for class_id, shard_entries in sorted(shards_by_class.items()):
                total_weight = 0.0
                weighted_sum = 0.0
                for shard_entry in shard_entries:
                    if not isinstance(shard_entry, dict):
                        continue
                    mean_value = shard_entry.get("wind_kt_mean")
                    count = shard_entry.get("count")
                    if mean_value is None or count is None:
                        continue
                    weighted_sum += float(mean_value) * float(count)
                    total_weight += float(count)
                if total_weight > 0.0:
                    cumulative_targets[str(class_id)] = float(weighted_sum / total_weight)
            manifest["conditioning_targets"] = {"wind_kt_by_class": cumulative_targets}

    write_sample_bank_manifest(bank_root, manifest)

    if bool(args.verbose):
        total_after = manifest.get("total_n_per_class")
        print(
            f"[sample_bank] Bank ready at {bank_root}. "
            f"Added {n_per_class} samples/class in shard part {part_index:04d}; "
            f"bank now has at least {total_after} samples/class."
        )
