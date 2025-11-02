"""
Lightweight geographic helpers used by the OMTAD analysis toolkit.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_NM = 1_852.0


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points in nautical miles.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    distance_m = EARTH_RADIUS_M * c
    return distance_m / METERS_PER_NM


def haversine_nm_vec(lat1, lon1, lat2, lon2):
    """
    Vectorised haversine distance in nautical miles.
    lat*, lon* are NumPy arrays of equal length.
    """
    R_m = 6_371_000.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = R_m * c
    return meters / METERS_PER_NM


def normalize_bearing_delta(prev_deg: float, curr_deg: float) -> float:
    """
    Normalize a course/bearing delta to the range [-180, 180] degrees.
    """
    raw = (curr_deg - prev_deg) % 360.0
    if raw > 180.0:
        raw -= 360.0
    return raw


def turn_rate(prev_course: Optional[float], course: Optional[float], dt_seconds: float) -> float:
    """
    Compute turn rate in degrees per second given two headings and the elapsed time.
    Returns 0.0 when input data is missing or dt_seconds <= 0.
    """
    if prev_course is None or course is None or not math.isfinite(prev_course) or not math.isfinite(course):
        return 0.0
    if dt_seconds <= 0.0:
        return 0.0
    delta = normalize_bearing_delta(prev_course, course)
    return delta / dt_seconds
