#!/usr/bin/env python3
"""
Create a curated demo track list for the XIV toolkit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_tracks(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if df.empty:
        return df
    take = min(n, len(df))
    return df.sample(n=take, random_state=random_state)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "stage" / "track_features.csv"
    default_output = script_dir / "deliverables" / "omtad_xiv" / "samples" / "demo_tracks.csv"

    parser = argparse.ArgumentParser(description="Generate a demo track list for the XIV walkthrough.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=default_input,
        help="Path to track_features.csv (default: ./data/stage/track_features.csv)",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=default_output,
        help="Output CSV path (default: ./deliverables/omtad_xiv/samples/demo_tracks.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    if df.empty:
        raise RuntimeError(f"No data found in {args.input_path}")

    selections = []
    for vessel_class, group in df.groupby("vessel_class"):
        selections.append(sample_tracks(group, 5))

    base_sample = pd.concat(selections, ignore_index=True).drop_duplicates(subset="track_id")

    fishing = df[df["vessel_class"] == "fishing"]
    extra_fishing = fishing[~fishing["track_id"].isin(base_sample["track_id"])]
    selections.append(sample_tracks(extra_fishing, 5))

    non_fishing = df[df["vessel_class"] != "fishing"]
    selections.append(sample_tracks(non_fishing[~non_fishing["track_id"].isin(base_sample["track_id"])], 5))

    combined = pd.concat(selections, ignore_index=True)
    combined = combined.drop_duplicates(subset="track_id")

    keep_cols = ["track_id", "vessel_class", "year", "month", "n_points", "mean_speed"]
    missing_cols = [col for col in keep_cols if col not in combined.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {missing_cols}")

    combined = combined[keep_cols]
    combined.sort_values(["vessel_class", "track_id"], inplace=True)

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Wrote {len(combined)} demo tracks to {output_path}")
    counts = combined.groupby("vessel_class")["track_id"].count()
    for vessel_class, count in counts.items():
        print(f"  {vessel_class}: {count} tracks")


if __name__ == "__main__":
    main()
