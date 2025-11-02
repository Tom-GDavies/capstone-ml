#!/usr/bin/env python3
"""
Construct per-track features from the flattened OMTAD dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from util.geo import haversine_nm, haversine_nm_vec, normalize_bearing_delta
from util.io_helpers import safe_write_csv, write_sample_head

DEFAULT_STAGE_FLAT = Path("./data/stage/omtad_flat.csv")
DEFAULT_STAGE_FEATURES = Path("./data/stage/track_features.csv")
SAMPLE_PATH = Path("./deliverables/omtad_xiv/samples/track_features_sample.csv")


def mode(series: pd.Series) -> str:
    counts = series.dropna().value_counts()
    if counts.empty:
        return ""
    return counts.idxmax()


def summarize_track(track_df: pd.DataFrame) -> dict:
    # Sort and coerce types
    tdf = track_df.sort_values("ts").copy()
    tdf["ts"] = pd.to_datetime(tdf["ts"], utc=True, errors="coerce")
    tdf = tdf.dropna(subset=["ts", "lat", "lon"])
    if len(tdf) < 2:
        # Minimal safe return for degenerate tracks
        return {
            "track_id": tdf["track_id"].iloc[0],
            "craft_id": tdf["craft_id"].iloc[0],
            "vessel_class": tdf["vessel_class"].mode(dropna=True).iloc[0],
            "year": int(pd.to_numeric(tdf["year"], errors="coerce").mode(dropna=True).iloc[0]),
            "month": tdf["month"].mode(dropna=True).iloc[0],
            "start_ts": tdf["ts"].iloc[0],
            "end_ts": tdf["ts"].iloc[-1],
            "n_points": len(tdf),
            "track_duration_min": 0.0,
            "mean_speed": float(tdf["speed"].mean()) if "speed" in tdf else 0.0,
            "max_speed": float(tdf["speed"].max()) if "speed" in tdf else 0.0,
            "mean_turn_rate": 0.0,
            "dist_nm": 0.0,
        }

    # To numpy
    ts_ns = tdf["ts"].values.astype("datetime64[ns]")
    lat = tdf["lat"].to_numpy(dtype="float64", copy=False)
    lon = tdf["lon"].to_numpy(dtype="float64", copy=False)
    speed = tdf["speed"].to_numpy(dtype="float64", copy=False)
    course = tdf["course_deg"].to_numpy(dtype="float64", copy=False)

    # Vectorised dt (seconds) between consecutive points
    dt_sec = np.diff(ts_ns).astype("timedelta64[s]").astype(np.float64)
    # Guard against non-positive/NaN dt
    valid = dt_sec > 0
    if not np.any(valid):
        dt_sec = np.ones_like(dt_sec, dtype=np.float64)  # avoid division by zero
        valid = np.ones_like(dt_sec, dtype=bool)

    # Vectorised bearing deltas in [-180, 180]
    dc = np.array(
        [normalize_bearing_delta(prev, curr) for prev, curr in zip(course[:-1], course[1:])],
        dtype=np.float64,
    )
    # Turn rate (deg/sec), absolute then mean over valid segments
    turn_rate = np.empty_like(dc, dtype=np.float64)
    turn_rate[:] = np.nan
    turn_rate[valid] = np.abs(dc[valid] / dt_sec[valid])
    mean_turn_rate = float(np.nanmean(turn_rate))

    # Vectorised great-circle distance (nm)
    if len(lat) == 2:
        dist_segments_nm = np.array([haversine_nm(lat[0], lon[0], lat[1], lon[1])], dtype=np.float64)
    else:
        dist_segments_nm = haversine_nm_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dist_nm = float(np.nansum(dist_segments_nm))

    start_ts = pd.to_datetime(ts_ns[0]).to_pydatetime()
    end_ts = pd.to_datetime(ts_ns[-1]).to_pydatetime()
    track_duration_min = float((ts_ns[-1] - ts_ns[0]).astype("timedelta64[s]").astype(np.int64) / 60.0)

    # Modes / representatives
    vessel_class = tdf["vessel_class"].mode(dropna=True).iloc[0]
    year = int(pd.to_numeric(tdf["year"], errors="coerce").mode(dropna=True).iloc[0])
    month = tdf["month"].mode(dropna=True).iloc[0]

    return {
        "track_id": tdf["track_id"].iloc[0],
        "craft_id": tdf["craft_id"].iloc[0],
        "vessel_class": vessel_class,
        "year": year,
        "month": month,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_points": int(len(tdf)),
        "track_duration_min": track_duration_min,
        "mean_speed": float(np.nanmean(speed)),
        "max_speed": float(np.nanmax(speed)),
        "mean_turn_rate": mean_turn_rate,
        "dist_nm": dist_nm,
    }


def build_features(in_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path, parse_dates=["ts"])
    if df.empty:
        raise RuntimeError("Flattened dataset is empty.")

    grouped = df.groupby("track_id", sort=False)
    records = [summarize_track(track_df) for _, track_df in grouped]
    features = pd.DataFrame.from_records(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_csv(features, out_path)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_sample_head(features, SAMPLE_PATH, n_rows=50)

    avg_duration = features["track_duration_min"].mean()
    print(f"Tracks processed: {len(features):,}")
    print(f"Average duration (minutes): {avg_duration:.2f}")

    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-track OMTAD features.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=DEFAULT_STAGE_FLAT,
        help="Input flattened CSV (default: ./data/stage/omtad_flat.csv).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=DEFAULT_STAGE_FEATURES,
        help="Output features CSV (default: ./data/stage/track_features.csv).",
    )
    args = parser.parse_args()

    build_features(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
