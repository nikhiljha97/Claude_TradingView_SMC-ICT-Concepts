"""Shared helpers for the ML research sidecar.

The code intentionally sticks to the Python standard library so data download,
feature extraction, and inference payload inspection work before heavier ML
packages are installed.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Iterable


BAR_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, ROOT / "reports"):
      path.mkdir(parents=True, exist_ok=True)


def parse_time(value: str | int | float | None) -> dt.datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.timezone.utc)
    text = str(value).strip()
    if text.isdigit():
        return dt.datetime.fromtimestamp(int(text) / 1000, tz=dt.timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def to_millis(value: dt.datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return int(value.timestamp() * 1000)


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


def read_bars(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        bars = []
        for row in reader:
            try:
                bars.append({
                    "timestamp": row.get("timestamp") or row.get("open_time") or row.get("time"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return bars


def write_bars(path: str | os.PathLike, bars: Iterable[dict]) -> None:
    write_rows(path, bars, BAR_FIELDS)


def write_rows(path: str | os.PathLike, rows: Iterable[dict], fieldnames: list[str] | tuple[str, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: str | os.PathLike, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path: str | os.PathLike, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 or not math.isfinite(den) else num / den


def pct_change(prev: float, curr: float) -> float:
    return safe_div(curr - prev, prev)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
