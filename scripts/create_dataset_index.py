#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime

import xarray as xr
from tqdm import tqdm


# -----------------------------
# Saffir–Simpson classification
# -----------------------------
def wind_to_ss_category(wind_kt: float) -> int | None:
    """
    Convert WMO wind speed (kt) to Saffir–Simpson category.

    Returns:
        int in [0..5] or None if below TS threshold.
    """
    if wind_kt is None:
        return None

    if 35 <= wind_kt <= 63:
        return 0  # Tropical Storm
    elif 64 <= wind_kt <= 82:
        return 1
    elif 83 <= wind_kt <= 95:
        return 2
    elif 96 <= wind_kt <= 112:
        return 3
    elif 113 <= wind_kt <= 136:
        return 4
    elif wind_kt > 136:
        return 5
    else:
        return None  # < 35 kt (TD or weaker)


# -----------------------------
# Main indexing logic
# -----------------------------
def build_index(data_root: Path, output_path: Path):
    nc_files = sorted(data_root.rglob("*.nc"))

    if not nc_files:
        raise RuntimeError(f"No .nc files found under {data_root}")

    class_to_files = {str(i): [] for i in range(6)}

    skipped_no_wind = 0
    skipped_below_ts = 0

    for nc_path in tqdm(nc_files, desc="Indexing NetCDF files"):
        try:
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                if "wmo_wind" not in ds.attrs:
                    skipped_no_wind += 1
                    continue

                # Extract scalar wind value safely
                wind = ds.attrs["wmo_wind"]
                if hasattr(wind, "__len__"):
                    wind = float(wind.squeeze())
                else:
                    wind = float(wind)

                ss_cat = wind_to_ss_category(wind)

                if ss_cat is None:
                    skipped_below_ts += 1
                    continue

                # Store relative path
                rel_path = nc_path.relative_to(data_root).as_posix()
                class_to_files[str(ss_cat)].append(rel_path)

        except Exception as e:
            raise RuntimeError(f"Failed processing {nc_path}: {e}")

    total_indexed = sum(len(v) for v in class_to_files.values())

    index = {
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset_root": str(data_root.resolve()),
            "num_files_total": len(nc_files),
            "num_files_indexed": total_indexed,
            "num_files_skipped_no_wind": skipped_no_wind,
            "num_files_skipped_below_ts": skipped_below_ts,
            "ss_definition": {
                "0": "Tropical Storm (35–63 kt)",
                "1": "Category 1 (64–82 kt)",
                "2": "Category 2 (83–95 kt)",
                "3": "Category 3 (96–112 kt)",
                "4": "Category 4 (113–136 kt)",
                "5": "Category 5 (>136 kt)",
            },
        },
        "classes": class_to_files,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print("\nIndex written to:", output_path)
    print("Summary:")
    for k in range(6):
        print(f"  Class {k}: {len(class_to_files[str(k)])} samples")
    print(f"  Skipped (no wind): {skipped_no_wind}")
    print(f"  Skipped (<35 kt): {skipped_below_ts}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build dataset_index.json for TC diffusion")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory containing NetCDF TC snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_index.json"),
        help="Output JSON index file",
    )

    args = parser.parse_args()
    build_index(args.data_root, args.output)


if __name__ == "__main__":
    main()
