#!/usr/bin/env python3
"""Download and aggregate Dukascopy tick data into OHLCV bars.

Dukascopy serves hourly .bi5 tick files. This downloader keeps the scope
intentionally narrow: FX-style symbols such as EURUSD, USDJPY, XAUUSD are
downloaded as bid/ask ticks and aggregated into minute/hour bars.
"""

from __future__ import annotations

import argparse
import datetime as dt
import lzma
import struct
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs, parse_time, write_rows


BASE_URL = "https://datafeed.dukascopy.com/datafeed"
FIELDS = ("timestamp", "open", "high", "low", "close", "volume", "source", "symbol", "interval")


def symbol_path(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def price_scale(symbol: str) -> float:
    clean = symbol_path(symbol)
    if clean.endswith("JPY") or clean in {"XAUUSD", "XAGUSD"}:
        return 1000.0
    return 100000.0


def iter_hours(start: dt.datetime, end: dt.datetime):
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        yield cursor
        cursor += dt.timedelta(hours=1)


def fetch_hour(symbol: str, hour: dt.datetime) -> bytes | None:
    month_zero_based = hour.month - 1
    url = f"{BASE_URL}/{symbol_path(symbol)}/{hour.year}/{month_zero_based:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except Exception:
        return None


def parse_ticks(symbol: str, hour: dt.datetime, payload: bytes):
    scale = price_scale(symbol)
    raw = lzma.decompress(payload)
    record_size = 20
    for offset in range(0, len(raw) - record_size + 1, record_size):
        millis, ask, bid, ask_vol, bid_vol = struct.unpack(">IIIff", raw[offset:offset + record_size])
        ts = hour + dt.timedelta(milliseconds=millis)
        mid = ((ask / scale) + (bid / scale)) / 2
        volume = max(0.0, float(ask_vol) + float(bid_vol))
        yield ts, mid, volume


def bucket_start(ts: dt.datetime, interval: str) -> dt.datetime:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        minute = (ts.minute // minutes) * minutes
        return ts.replace(minute=minute, second=0, microsecond=0)
    if interval.endswith("h"):
        hours = int(interval[:-1])
        hour = (ts.hour // hours) * hours
        return ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported interval: {interval}")


def aggregate_ticks(ticks, interval: str):
    buckets = defaultdict(list)
    for ts, price, volume in ticks:
        buckets[bucket_start(ts, interval)].append((price, volume))
    rows = []
    for ts in sorted(buckets):
        values = buckets[ts]
        prices = [p for p, _ in values]
        rows.append({
            "timestamp": int(ts.timestamp() * 1000),
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(v for _, v in values),
        })
    return rows


def download_symbol(symbol: str, interval: str, start: str, end: str, out_dir: Path) -> Path:
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if not start_dt or not end_dt:
        raise ValueError("Both --start and --end are required.")

    ticks = []
    for hour in iter_hours(start_dt, end_dt):
        payload = fetch_hour(symbol, hour)
        if not payload:
            continue
        ticks.extend(parse_ticks(symbol, hour, payload))
        time.sleep(0.05)

    rows = aggregate_ticks(ticks, interval)
    for row in rows:
        row.update({"source": "dukascopy", "symbol": symbol_path(symbol), "interval": interval})

    path = out_dir / f"{symbol_path(symbol)}_{interval}_{start_dt.date()}_{end_dt.date()}.csv"
    write_rows(path, rows, FIELDS)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", default=str(RAW_DIR / "dukascopy"))
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        path = download_symbol(symbol, args.interval, args.start, args.end, out_dir)
        print(path)


if __name__ == "__main__":
    main()
