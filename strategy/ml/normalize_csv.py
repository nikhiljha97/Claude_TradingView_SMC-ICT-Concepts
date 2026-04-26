#!/usr/bin/env python3
"""Normalize broker/export CSV files into the sidecar OHLCV schema."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs, write_bars


def parse_export_time(value: str) -> int:
    parsed = dt.datetime.strptime(value.strip(), "%m/%d/%Y %H:%M")
    parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def normalize(path: str, symbol: str) -> list[dict]:
    rows = []
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        for raw in reader:
            if not raw:
                continue
            if raw[0] == "Date":
                header = raw
                break
        if not header:
            raise ValueError(f"No Date header found in {path}")
        dict_reader = csv.DictReader(f, fieldnames=header)
        for row in dict_reader:
            try:
                rows.append({
                    "timestamp": parse_export_time(row["Date"]),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": 0.0,
                })
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    rows = normalize(args.input, args.symbol)
    out = Path(args.out) if args.out else RAW_DIR / "local" / f"{args.symbol.upper()}_15m_normalized.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_bars(out, rows)
    print(out)


if __name__ == "__main__":
    main()
