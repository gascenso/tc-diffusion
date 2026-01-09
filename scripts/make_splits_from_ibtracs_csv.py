#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def month_to_season_bin(month: int) -> str:
    # 4 coarse bins (works well and keeps strata populated)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def lat_to_bin(lat: float) -> str:
    # Signed latitude binning (simple + physically meaningful)
    a = abs(lat)
    if a < 10:
        return "lat0_10"
    if a < 20:
        return "lat10_20"
    if a < 30:
        return "lat20_30"
    return "lat30p"


def dur_to_bin(n: int) -> str:
    # Snapshot-count binning (robust across basins)
    if n <= 4:
        return "n1_4"
    if n <= 12:
        return "n5_12"
    if n <= 30:
        return "n13_30"
    return "n31p"


def build_idx_to_relpath(data_root: Path) -> dict[int, str]:
    """
    Scan all .nc files under data_root and map IDX_TRUE (prefix before first dash)
    to the file's relative path (POSIX).
    """
    idx_to_rel = {}
    nc_files = list(data_root.rglob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"No .nc files found under {data_root}")

    for p in tqdm(nc_files, desc="Scanning .nc files to map IDX_TRUE -> path"):
        name = p.name
        # Expect format: "<IDX_TRUE>-....nc"
        try:
            prefix = name.split("-", 1)[0]
            idx = int(prefix)
        except Exception:
            continue
        rel = p.relative_to(data_root).as_posix()
        idx_to_rel[idx] = rel

    return idx_to_rel


def make_storm_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make one-row-per-storm table with storm-level stratification features.
    """
    # Ensure ISO_TIME is datetime
    t = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df = df.copy()
    df["ISO_TIME_DT"] = t

    # We'll use WMO_WIND to define "peak". If WMO_WIND missing, treat as -inf.
    w = pd.to_numeric(df["WMO_WIND"], errors="coerce").fillna(-np.inf)
    df["WMO_WIND_NUM"] = w

    # Some storms may have SS_CAT; use numeric.
    ss = pd.to_numeric(df["SS_CAT"], errors="coerce").fillna(-1).astype(int)
    df["SS_CAT_NUM"] = ss

    rows = []
    for sid, g in df.groupby("SID", sort=False):
        n = len(g)

        # Basin: mode (most common)
        basin_mode = g["BASIN"].mode(dropna=True)
        basin = basin_mode.iloc[0] if len(basin_mode) else str(g["BASIN"].iloc[0])

        # Peak row: max WMO_WIND
        i_peak = g["WMO_WIND_NUM"].idxmax()
        g_peak = g.loc[i_peak]

        max_wind = float(g_peak["WMO_WIND_NUM"]) if np.isfinite(g_peak["WMO_WIND_NUM"]) else float(np.nan)
        max_ss = int(g["SS_CAT_NUM"].max())

        iso_peak = g_peak["ISO_TIME_DT"]
        month = int(iso_peak.month) if pd.notna(iso_peak) else 1
        season = month_to_season_bin(month)

        lat_peak = float(g_peak["LAT"])
        lat_bin = lat_to_bin(lat_peak)

        dur_bin = dur_to_bin(n)

        # Build stratum label (keeps train/val/test similar by construction)
        # You can refine later if needed.
        stratum = f"{basin}|ss{max_ss}|{season}|{lat_bin}|{dur_bin}"

        rows.append(
            {
                "SID": sid,
                "BASIN_MODE": basin,
                "MAX_WMO_WIND": max_wind,
                "MAX_SS_CAT": max_ss,
                "MONTH_AT_PEAK": month,
                "SEASON_BIN": season,
                "LAT_AT_PEAK": lat_peak,
                "LAT_BIN": lat_bin,
                "N_SNAPSHOTS": n,
                "DUR_BIN": dur_bin,
                "STRATUM": stratum,
            }
        )

    return pd.DataFrame(rows)


def stratified_group_split(
    storm_df: pd.DataFrame,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> dict[str, list[str]]:
    """
    Deterministic stratified split over storms, stratified by STRATUM.
    Allocates storms within each stratum to train/val/test according to fractions.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    rng = random.Random(seed)

    splits = {"train": [], "val": [], "test": []}

    # Group storms by stratum
    by_stratum = defaultdict(list)
    for _, row in storm_df.iterrows():
        by_stratum[row["STRATUM"]].append(row["SID"])

    for stratum, sids in by_stratum.items():
        sids = list(sids)
        rng.shuffle(sids)
        n = len(sids)

        n_train = int(round(frac_train * n))
        n_val = int(round(frac_val * n))
        n_test = n - n_train - n_val

        # Ensure non-negative
        n_train = max(0, min(n, n_train))
        n_val = max(0, min(n - n_train, n_val))
        n_test = n - n_train - n_val

        # Edge cases: tiny strata
        # If n is small, we prefer keeping at least 1 in train.
        if n == 1:
            splits["train"].extend(sids)
            continue
        if n == 2:
            splits["train"].append(sids[0])
            splits["val"].append(sids[1])
            continue

        splits["train"].extend(sids[:n_train])
        splits["val"].extend(sids[n_train:n_train + n_val])
        splits["test"].extend(sids[n_train + n_val:])

    # Final shuffle within split for convenience (deterministic)
    for k in splits:
        rng.shuffle(splits[k])

    return splits


def write_list(path: Path, items: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for x in items:
            f.write(str(x) + "\n")


def summarize_split(df: pd.DataFrame, files: list[str], name: str) -> dict:
    """
    Snapshot-level summary for auditing similarity.
    """
    rel_set = set(files)
    sub = df[df["REL_PATH"].isin(rel_set)].copy()

    out = {
        "name": name,
        "num_snapshots": int(len(sub)),
        "num_storms": int(sub["SID"].nunique()),
        "basin_counts": sub["BASIN"].value_counts().to_dict(),
        "ss_cat_counts": sub["SS_CAT"].value_counts().to_dict(),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=Path, required=True, help="IBTrACS postprocessed CSV")
    ap.add_argument("--data_root", type=Path, required=True, help="Root directory containing snapshot .nc files")
    ap.add_argument("--out_dir", type=Path, default=Path("data/splits"), help="Where to write split manifests")
    ap.add_argument("--frac_train", type=float, default=0.80)
    ap.add_argument("--frac_val", type=float, default=0.10)
    ap.add_argument("--frac_test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)

    required = ["SID", "IDX_TRUE", "BASIN", "ISO_TIME", "LAT", "LON", "WMO_WIND", "SS_CAT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    # Map IDX_TRUE -> rel path by scanning the filesystem once
    idx_to_rel = build_idx_to_relpath(args.data_root)

    # Attach REL_PATH to each row
    idx = pd.to_numeric(df["IDX_TRUE"], errors="coerce").astype("Int64")
    df = df.copy()
    df["IDX_TRUE_INT"] = idx
    df["REL_PATH"] = df["IDX_TRUE_INT"].map(lambda x: idx_to_rel.get(int(x)) if pd.notna(x) else None)

    n_missing = int(df["REL_PATH"].isna().sum())
    if n_missing > 0:
        # This should be very small. If large, something is inconsistent between CSV and file naming.
        print(f"[warn] {n_missing} CSV rows could not be mapped to a .nc file by IDX_TRUE prefix.")
        df = df.dropna(subset=["REL_PATH"])

    # Storm-level table for stratification
    storm_df = make_storm_table(df)

    # Perform stratified group split
    splits = stratified_group_split(
        storm_df=storm_df,
        frac_train=args.frac_train,
        frac_val=args.frac_val,
        frac_test=args.frac_test,
        seed=args.seed,
    )

    # Expand storms -> snapshot file lists
    files_by_split = {}
    for split_name, sid_list in splits.items():
        sid_set = set(sid_list)
        files = df[df["SID"].isin(sid_set)]["REL_PATH"].tolist()
        files_by_split[split_name] = files

    # Write outputs
    out_dir = args.out_dir
    write_list(out_dir / "storm_ids_train.txt", splits["train"])
    write_list(out_dir / "storm_ids_val.txt", splits["val"])
    write_list(out_dir / "storm_ids_test.txt", splits["test"])

    write_list(out_dir / "files_train.txt", files_by_split["train"])
    write_list(out_dir / "files_val.txt", files_by_split["val"])
    write_list(out_dir / "files_test.txt", files_by_split["test"])

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(args.csv_path),
        "data_root": str(args.data_root),
        "fractions": {"train": args.frac_train, "val": args.frac_val, "test": args.frac_test},
        "seed": args.seed,
        "storm_level": {
            "num_storms_total": int(storm_df.shape[0]),
            "strata_counts": storm_df["STRATUM"].value_counts().to_dict(),
        },
        "snapshot_level": {
            "total_rows_after_mapping": int(len(df)),
            "missing_rows_dropped": n_missing,
            "train": summarize_split(df, files_by_split["train"], "train"),
            "val": summarize_split(df, files_by_split["val"], "val"),
            "test": summarize_split(df, files_by_split["test"], "test"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "split_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("\nWrote split manifests to:", out_dir)
    print("Storm counts:", {k: len(v) for k, v in splits.items()})
    print("Snapshot counts:", {k: len(v) for k, v in files_by_split.items()})
    print("Report:", out_dir / "split_report.json")


if __name__ == "__main__":
    main()
