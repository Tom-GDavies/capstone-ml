#!/usr/bin/env python3
"""
Verify that all deliverable artefacts exist for the XIV demo bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path


CHECK_MARK = "\u2713"
CROSS_MARK = "\u2717"

DELIVERABLES_ROOT = Path("./deliverables/omtad_xiv")
EXPECTED = [
    (DELIVERABLES_ROOT / "models/omtad_vessel_type.pkl", True),
    (DELIVERABLES_ROOT / "models/omtad_fishing.pkl", True),
    (DELIVERABLES_ROOT / "models/omtad_vessel_type.meta.json", True),
    (DELIVERABLES_ROOT / "models/omtad_fishing.meta.json", True),
    (DELIVERABLES_ROOT / "charts/confusion_matrix_multiclass.png", True),
    (DELIVERABLES_ROOT / "charts/f1_per_class.png", True),
    (DELIVERABLES_ROOT / "charts/speed_hist_by_class.png", True),
    (DELIVERABLES_ROOT / "metrics/report.txt", True),
    (DELIVERABLES_ROOT / "samples/track_features_sample.csv", True),
]

OPTIONAL = [
    (DELIVERABLES_ROOT / "samples/demo_tracks.csv", False),
]


def describe(path: Path) -> str:
    try:
        return str(path.relative_to(Path(".")))
    except ValueError:
        return str(path)


def main() -> int:
    missing = False
    optional_set = {path for path, _ in OPTIONAL}

    for path, required in EXPECTED + OPTIONAL:
        exists = path.exists()
        symbol = CHECK_MARK if exists else CROSS_MARK
        note = " (optional)" if path in optional_set else ""
        print(f"{symbol} {describe(path)}{note}")
        if required and not exists:
            missing = True

    if missing:
        print("One or more required artefacts are missing.", file=sys.stderr)
        return 1

    print("All required artefacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
