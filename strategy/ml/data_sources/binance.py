#!/usr/bin/env python3
"""Download Binance spot OHLCV klines.

Example:
  python strategy/ml/data_sources/binance.py \
    --symbols BTCUSDT ETHUSDT --interval 15m --start 2023-01-01 --end 2023-03-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs, interval_to_millis, parse_time, to_millis, write_rows


BASE_URL = "https://api.binance.com/api/v3/klines"
FIELDS = (
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "source",
    "symbol", "interval",
)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    })
    with urllib.request.urlopen(f"{BASE_URL}?{params}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_symbol(symbol: str, interval: str, start: str, end: str, out_dir: Path) -> Path:
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if not start_dt or not end_dt:
        raise ValueError("Both --start and --end are required.")

    start_ms = to_millis(start_dt)
    end_ms = to_millis(end_dt)
    step_ms = interval_to_millis(interval)
    rows = []
    cursor = start_ms

    while cursor < end_ms:
        batch = fetch_klines(symbol, interval, cursor, end_ms)
        if not batch:
            break
        for item in batch:
            rows.append({
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "close_time": item[6],
                "quote_volume": item[7],
                "trades": item[8],
                "taker_buy_base": item[9],
                "taker_buy_quote": item[10],
                "source": "binance",
                "symbol": symbol,
                "interval": interval,
            })
        next_cursor = int(batch[-1][0]) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)

    path = out_dir / f"{symbol}_{interval}_{start_dt.date()}_{end_dt.date()}.csv"
    write_rows(path, rows, FIELDS)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", default=str(RAW_DIR / "binance"))
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        path = download_symbol(symbol.upper(), args.interval, args.start, args.end, out_dir)
        print(path)


if __name__ == "__main__":
    main()
