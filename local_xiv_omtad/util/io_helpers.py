"""
I/O utilities for resilient CSV handling and atomic writes.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import pandas as pd

DEFAULT_READ_KWARGS = {
    "dtype": None,
    "encoding": "utf-8",
    "encoding_errors": "ignore",
}

DEFAULT_WRITE_KWARGS = {
    "index": False,
}


def read_csv_in_chunks(path: Path, chunk_size: int = 100_000, **kwargs) -> Iterable[pd.DataFrame]:
    """
    Yield pandas DataFrame chunks from a CSV file while tolerating encoding issues.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    options = {**DEFAULT_READ_KWARGS, **kwargs}
    reader = pd.read_csv(csv_path, chunksize=chunk_size, **options)
    for chunk in reader:
        yield chunk


def safe_write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """
    Write a CSV using a temporary file, then atomically move into place.
    """
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    options = {**DEFAULT_WRITE_KWARGS, **kwargs}

    with NamedTemporaryFile("w", delete=False, dir=str(csv_path.parent), newline="") as tmp:
        temp_name = tmp.name
        df.to_csv(tmp.name, **options)

    os.replace(temp_name, csv_path)


def write_sample_head(df: pd.DataFrame, path: Path, n_rows: int = 50, **kwargs) -> None:
    """
    Persist the first n_rows of a DataFrame using safe_write_csv.
    """
    sample = df.head(n_rows)
    safe_write_csv(sample, path, **kwargs)
