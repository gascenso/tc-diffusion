#!/usr/bin/env python3

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr
from numpy.lib.format import open_memmap
from tqdm import tqdm


def load_records(dataset_index_path: Path) -> Tuple[List[Tuple[str, int]], dict]:
    with dataset_index_path.open("r") as f:
        obj = json.load(f)

    records: List[Tuple[str, int]] = []
    seen = set()

    for class_str, rel_paths in sorted(obj["classes"].items(), key=lambda kv: int(kv[0])):
        label = int(class_str)
        for rel_path in rel_paths:
            rel_path = str(rel_path)
            if rel_path in seen:
                raise ValueError(f"Duplicate rel_path in dataset index: {rel_path}")
            seen.add(rel_path)
            records.append((rel_path, label))

    records.sort(key=lambda item: item[0])
    if not records:
        raise RuntimeError(f"No indexed samples found in {dataset_index_path}")

    return records, obj.get("meta", {})


def load_raw_bt(nc_path: Path, engine: str) -> np.ndarray:
    with xr.open_dataset(nc_path, engine=engine) as ds:
        bt = ds["bt"].values.astype(np.float32)

    if bt.ndim != 2:
        raise ValueError(f"Expected 2D 'bt' array in {nc_path}, found shape {tuple(bt.shape)}")
    return bt


def pack_dataset(
    *,
    data_root: Path,
    dataset_index_path: Path,
    output_dir: Path,
    samples_per_shard: int,
    engine: str,
    overwrite: bool,
):
    if samples_per_shard <= 0:
        raise ValueError(f"samples_per_shard must be > 0, got {samples_per_shard}")

    records, source_meta = load_records(dataset_index_path)
    num_samples = len(records)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    first_rel_path, _ = records[0]
    first_bt = load_raw_bt(data_root / first_rel_path, engine=engine)
    sample_shape = tuple(int(v) for v in first_bt.shape)

    rel_paths_path = output_dir / "rel_paths.txt"
    with rel_paths_path.open("w") as f:
        for rel_path, _ in records:
            f.write(rel_path + "\n")

    labels = np.asarray([label for _, label in records], dtype=np.int16)
    labels_path = output_dir / "labels.npy"
    np.save(labels_path, labels)

    shards = []
    shard_ranges = range(0, num_samples, samples_per_shard)
    for shard_idx, start in enumerate(shard_ranges):
        stop = min(start + samples_per_shard, num_samples)
        shard_rel_path = Path("shards") / f"bt_shard_{shard_idx:05d}.npy"
        shard_path = output_dir / shard_rel_path

        shard_mm = open_memmap(
            shard_path,
            mode="w+",
            dtype=np.float32,
            shape=(stop - start, *sample_shape),
        )

        shard_records = records[start:stop]
        for local_idx, (rel_path, _) in enumerate(
            tqdm(shard_records, desc=f"Writing shard {shard_idx:05d}", leave=True)
        ):
            bt = load_raw_bt(data_root / rel_path, engine=engine)
            if tuple(bt.shape) != sample_shape:
                raise ValueError(
                    f"Inconsistent bt shape for {rel_path}: expected {sample_shape}, found {tuple(bt.shape)}"
                )
            shard_mm[local_idx] = bt

        shard_mm.flush()
        del shard_mm

        shards.append(
            {
                "path": shard_rel_path.as_posix(),
                "start": int(start),
                "stop": int(stop),
            }
        )

    manifest = {
        "version": 1,
        "backend": "packed_memmap",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": num_samples,
        "sample_shape": list(sample_shape),
        "dtype": "float32",
        "samples_per_shard": int(samples_per_shard),
        "rel_paths_file": rel_paths_path.name,
        "labels_file": labels_path.name,
        "shards": shards,
        "source": {
            "data_root": str(data_root.resolve()),
            "dataset_index": str(dataset_index_path.resolve()),
            "dataset_index_meta": source_meta,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Packed dataset written to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Samples: {num_samples}")
    print(f"Sample shape: {sample_shape}")
    print(f"Shards: {len(shards)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pack indexed NetCDF samples into sharded float32 .npy memmaps for HPC training."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory containing the original NetCDF files.",
    )
    parser.add_argument(
        "--dataset_index",
        type=Path,
        default=Path("data/dataset_index.json"),
        help="Path to dataset_index.json. Only indexed samples are packed.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for the packed dataset.",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=4096,
        help="Number of samples per shard file.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        help="xarray engine to use when reading source NetCDF files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_dir if it already exists.",
    )

    args = parser.parse_args()
    pack_dataset(
        data_root=args.data_root,
        dataset_index_path=args.dataset_index,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        engine=args.engine,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
