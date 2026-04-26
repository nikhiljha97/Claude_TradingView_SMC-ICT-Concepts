#!/usr/bin/env python3
"""Retrain the PyTorch GRU from accumulated scanner OHLCV captures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import PROCESSED_DIR, ensure_dirs, read_bars, write_rows
from strategy.ml.features import load_gpr
from strategy.ml.labels import LABEL_FIELDS, label_rows
from strategy.ml.train_rnn import train_model


def count_rows(path: Path) -> int:
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpr", default=None)
    parser.add_argument("--seed-label", action="append", default=[])
    parser.add_argument("--min-bars", type=int, default=250)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lock", default=None)
    args = parser.parse_args()

    ensure_dirs()
    lock = Path(args.lock) if args.lock else None
    if lock:
        if lock.exists():
            print(f"lock exists: {lock}")
            return
        lock.write_text(str(Path(args.out)))

    try:
        live_dir = Path(args.live_dir)
        gpr = load_gpr(args.gpr)
        all_rows = []
        for csv_path in sorted(live_dir.glob("*_15.csv")):
            if count_rows(csv_path) < args.min_bars:
                continue
            bars = read_bars(csv_path)
            label_path = PROCESSED_DIR / f"live_{csv_path.stem}_labels.csv"
            rows = []
            rows.extend(label_rows(bars, "long", 2.5, 1.0, 32, gpr))
            rows.extend(label_rows(bars, "short", 2.5, 1.0, 32, gpr))
            write_rows(label_path, rows, LABEL_FIELDS)
            all_rows.extend(rows)

        for seed in args.seed_label:
            seed_path = Path(seed)
            if seed_path.exists():
                all_rows.extend(load_csv_rows(seed_path))

        if len(all_rows) < args.seq_len + 500:
            print({"trained": False, "reason": "not_enough_labeled_rows", "rows": len(all_rows)})
            return

        combined = PROCESSED_DIR / "live_retrain_labels.csv"
        write_rows(combined, all_rows, LABEL_FIELDS)
        report = train_model(
            str(combined),
            seq_len=args.seq_len,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_path=args.out,
        )
        print({"trained": True, "rows": len(all_rows), "report": report})
    finally:
        if lock and lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
