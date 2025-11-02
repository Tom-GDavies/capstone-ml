#!/usr/bin/env python3
"""
Flatten the OMTAD vessel CSV archive into a single cleaned file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional, Tuple
from zipfile import ZipFile, BadZipFile

import pandas as pd

from util.io_helpers import safe_write_csv

DEFAULT_SOURCE_DIR = Path("./data")
DEFAULT_STAGE_FLAT = Path("./data/stage/omtad_flat.csv")

MONTH_ALIASES = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}

EXPECTED_COLUMNS = {
    "CRAFT_ID": "craft_id",
    "LON": "lon",
    "LAT": "lat",
    "COURSE": "course_deg",
    "SPEED": "speed",
    "TIMESTAMP": "timestamp",
    "TRACK_ID": "track_id",
}


def canonical_month(name: str) -> str:
    key = name.strip().lower()
    return MONTH_ALIASES.get(key, name.strip().title())


def find_dataset_root(source: Path) -> Tuple[str, Path]:
    zip_path = source / "OMTAD-main.zip"
    folder_path = source / "source" / "OMTAD-main"

    if zip_path.exists():
        return "zip", zip_path
    if folder_path.exists():
        return "dir", folder_path
    raise FileNotFoundError(
        f"Could not find OMTAD-main.zip or source folder under {source}"
    )


def parse_path_meta(parts: Iterable[str]) -> Optional[Tuple[int, str, str]]:
    parts_list = list(parts)
    for idx, part in enumerate(parts_list):
        if part.lower() == "west grid":
            try:
                year = int(parts_list[idx + 1])
                vessel_class = parts_list[idx + 2].strip().lower()
                month = canonical_month(parts_list[idx + 3])
                return year, vessel_class, month
            except (IndexError, ValueError):
                return None
    return None


def iter_files_from_zip(zip_path: Path) -> Iterator[Tuple[str, int, str, str]]:
    try:
        with ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".csv"):
                    continue
                path_parts = PurePosixPath(info.filename).parts
                meta = parse_path_meta(path_parts)
                if not meta:
                    continue
                year, vessel_class, month = meta
                yield info.filename, year, vessel_class, month
    except BadZipFile as exc:
        raise RuntimeError(f"Failed to open zip archive: {zip_path}") from exc


def iter_files_from_dir(root: Path) -> Iterator[Tuple[Path, int, str, str]]:
    for csv_path in root.rglob("*.csv"):
        parts = csv_path.parts
        meta = parse_path_meta(parts)
        if not meta:
            continue
        year, vessel_class, month = meta
        yield csv_path, year, vessel_class, month


def load_csv_from_zip(zf: ZipFile, member: str) -> Optional[pd.DataFrame]:
    try:
        with zf.open(member, "r") as raw:
            # Wrap in TextIO to ensure pandas handles encoding consistently.
            text_file = TextIOWrapper(raw, encoding="utf-8", errors="ignore")
            df = pd.read_csv(text_file)
            return df
    except Exception as exc:  # pragma: no cover - log and continue
        print(f"Skipping {member}: {exc}", file=sys.stderr)
        return None


def load_csv_from_path(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
    except Exception as exc:  # pragma: no cover - log and continue
        print(f"Skipping {csv_path}: {exc}", file=sys.stderr)
        return None


def normalize_dataframe(
    df: pd.DataFrame,
    year: int,
    vessel_class: str,
    month: str,
) -> pd.DataFrame:
    rename_map = {col: EXPECTED_COLUMNS[col.upper()] for col in df.columns if col.upper() in EXPECTED_COLUMNS}
    df = df.rename(columns=rename_map)

    missing = [col for col in EXPECTED_COLUMNS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    numeric_cols = ["lon", "lat", "course_deg", "speed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["track_id"] = df["track_id"].astype(str).str.strip()
    df["craft_id"] = df["craft_id"].astype(str).str.strip()

    df = df.dropna(subset=["ts", "lat", "lon"])
    df = df[(df["lat"].between(-90.0, 90.0)) & (df["lon"].between(-180.0, 180.0))]

    df["vessel_class"] = vessel_class
    df["year"] = year
    df["month"] = month
    df["lat_bin"] = df["lat"].round(2)
    df["lon_bin"] = df["lon"].round(2)

    final_cols = [
        "craft_id",
        "ts",
        "lon",
        "lat",
        "course_deg",
        "speed",
        "track_id",
        "vessel_class",
        "year",
        "month",
        "lat_bin",
        "lon_bin",
    ]

    return df[final_cols]


def flatten(source_dir: Path, out_path: Path) -> None:
    mode, root = find_dataset_root(source_dir)
    frames = []
    summary_counts = defaultdict(int)

    if mode == "zip":
        with ZipFile(root) as zf:
            for member, year, vessel_class, month in iter_files_from_zip(root):
                raw = load_csv_from_zip(zf, member)
                if raw is None or raw.empty:
                    continue
                try:
                    normalized = normalize_dataframe(raw, year, vessel_class, month)
                except Exception as exc:
                    print(f"Skipping {member}: {exc}", file=sys.stderr)
                    continue
                frames.append(normalized)
                summary_counts[(year, vessel_class)] += len(normalized)
    else:
        for csv_path, year, vessel_class, month in iter_files_from_dir(root):
            raw = load_csv_from_path(csv_path)
            if raw is None or raw.empty:
                continue
            try:
                normalized = normalize_dataframe(raw, year, vessel_class, month)
            except Exception as exc:
                print(f"Skipping {csv_path}: {exc}", file=sys.stderr)
                continue
            frames.append(normalized)
            summary_counts[(year, vessel_class)] += len(normalized)

    if not frames:
        raise RuntimeError("No valid CSV data found. Ensure the OMTAD archive is present.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["track_id", "ts"], inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_csv(combined, out_path)

    print(f"Wrote {len(combined):,} rows to {out_path}")
    print("Counts by year and vessel_class:")
    records = [
        {"year": year, "vessel_class": vessel_class, "rows": count}
        for (year, vessel_class), count in sorted(summary_counts.items())
    ]
    if records:
        summary_df = pd.DataFrame(records)
        for _, row in summary_df.iterrows():
            print(f"  {row['year']}: {row['vessel_class']} -> {int(row['rows']):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten OMTAD archive into a single CSV.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing OMTAD-main.zip or source/OMTAD-main/ (default: ./data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_STAGE_FLAT,
        help="Output CSV path (default: ./data/stage/omtad_flat.csv).",
    )
    args = parser.parse_args()

    flatten(args.source, args.out)


if __name__ == "__main__":
    main()
