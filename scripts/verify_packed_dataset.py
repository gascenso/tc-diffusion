#!/usr/bin/env python3

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import xarray as xr


class PackedDataset:
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r") as f:
            manifest = json.load(f)

        backend = str(manifest.get("backend", "")).lower()
        if backend != "packed_memmap":
            raise ValueError(
                f"Manifest backend mismatch: expected 'packed_memmap', found '{backend or 'missing'}'"
            )

        self.root = self.manifest_path.parent
        self.sample_shape = tuple(int(v) for v in manifest["sample_shape"])
        self.num_samples = int(manifest["num_samples"])

        with (self.root / manifest["rel_paths_file"]).open("r") as f:
            self.rel_paths = [line.strip() for line in f if line.strip()]
        self.labels = np.load(self.root / manifest["labels_file"], mmap_mode="r")
        self.shards = manifest["shards"]
        self._shard_starts = [int(shard["start"]) for shard in self.shards]
        self._memmaps = {}

        if len(self.rel_paths) != self.num_samples:
            raise ValueError("rel_paths count does not match manifest num_samples")
        if int(self.labels.shape[0]) != self.num_samples:
            raise ValueError("labels count does not match manifest num_samples")
        prev_stop = 0
        for shard in self.shards:
            start = int(shard["start"])
            stop = int(shard["stop"])
            if start != prev_stop:
                raise ValueError(
                    f"Shard layout must be contiguous; expected start={prev_stop}, found {start}"
                )
            if stop <= start:
                raise ValueError(f"Invalid shard bounds [{start}, {stop}) in manifest")
            prev_stop = stop
        if prev_stop != self.num_samples:
            raise ValueError(
                f"Shard coverage mismatch: last shard stops at {prev_stop}, expected {self.num_samples}"
            )

    def _resolve_shard(self, global_index: int):
        shard_idx = bisect_right(self._shard_starts, int(global_index)) - 1
        if shard_idx < 0:
            raise IndexError(f"Could not resolve shard for index {global_index}")

        shard = self.shards[shard_idx]
        start = int(shard["start"])
        stop = int(shard["stop"])
        if global_index < start or global_index >= stop:
            raise IndexError(f"Index {global_index} outside shard bounds [{start}, {stop})")
        return shard_idx, shard

    def _get_memmap(self, shard_idx: int):
        mm = self._memmaps.get(shard_idx)
        if mm is not None:
            return mm

        shard = self.shards[shard_idx]
        mm = np.load(self.root / shard["path"], mmap_mode="r")
        self._memmaps[shard_idx] = mm
        return mm

    def load_bt(self, global_index: int) -> np.ndarray:
        shard_idx, shard = self._resolve_shard(global_index)
        mm = self._get_memmap(shard_idx)
        local_idx = int(global_index) - int(shard["start"])
        return np.asarray(mm[local_idx], dtype=np.float32)


def load_dataset_index(dataset_index_path: Path) -> dict[str, int]:
    with dataset_index_path.open("r") as f:
        obj = json.load(f)

    file_to_label = {}
    for class_str, rel_paths in obj["classes"].items():
        label = int(class_str)
        for rel_path in rel_paths:
            rel_path = str(rel_path)
            if rel_path in file_to_label:
                raise ValueError(f"Duplicate rel_path in dataset index: {rel_path}")
            file_to_label[rel_path] = label
    return file_to_label


def load_raw_bt(data_root: Path, rel_path: str, engine: str) -> np.ndarray:
    with xr.open_dataset(data_root / rel_path, engine=engine) as ds:
        bt = ds["bt"].values.astype(np.float32)
    return bt


def main():
    parser = argparse.ArgumentParser(
        description="Verify a packed memmap dataset against the original NetCDF files."
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
        help="Path to dataset_index.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to packed dataset manifest.json.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of random samples to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        help="xarray engine to use for source NetCDF files.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance allowed between source and packed arrays.",
    )

    args = parser.parse_args()

    packed = PackedDataset(args.manifest)
    file_to_label = load_dataset_index(args.dataset_index)

    missing = [rel_path for rel_path in packed.rel_paths if rel_path not in file_to_label]
    if missing:
        raise RuntimeError(
            f"Packed dataset contains {len(missing)} rel_paths missing from dataset_index.json; "
            f"first example: {missing[0]}"
        )

    for idx, rel_path in enumerate(packed.rel_paths):
        expected_label = file_to_label[rel_path]
        actual_label = int(packed.labels[idx])
        if actual_label != expected_label:
            raise RuntimeError(
                f"Label mismatch for {rel_path}: expected {expected_label}, found {actual_label}"
            )

    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.num_samples), packed.num_samples)
    if sample_count <= 0:
        raise ValueError(f"num_samples must be > 0, got {args.num_samples}")

    picked = rng.choice(packed.num_samples, size=sample_count, replace=False)

    max_abs_diff = 0.0
    for global_index in picked:
        rel_path = packed.rel_paths[int(global_index)]
        raw_bt = load_raw_bt(args.data_root, rel_path, engine=args.engine)
        packed_bt = packed.load_bt(int(global_index))

        if raw_bt.shape != packed_bt.shape:
            raise RuntimeError(
                f"Shape mismatch for {rel_path}: source {raw_bt.shape}, packed {packed_bt.shape}"
            )

        diff = float(np.max(np.abs(raw_bt - packed_bt)))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > float(args.atol):
            raise RuntimeError(
                f"Value mismatch for {rel_path}: max_abs_diff={diff} exceeds atol={args.atol}"
            )

    print(f"Verified labels for {packed.num_samples} packed samples.")
    print(f"Compared {sample_count} random BT arrays.")
    print(f"Maximum absolute difference: {max_abs_diff}")


if __name__ == "__main__":
    main()
