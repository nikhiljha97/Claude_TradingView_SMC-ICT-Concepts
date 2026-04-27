#!/usr/bin/env python3
"""Retrain the PyTorch GRU from accumulated scanner OHLCV captures."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import PROCESSED_DIR, ensure_dirs, read_bars, write_rows
from strategy.ml.features import build_features, load_gpr, load_news
from strategy.ml.labels import LABEL_FIELDS, label_rows
from strategy.ml.train_rnn import train_model


def count_rows(path: Path) -> int:
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trade_outcome_rows(trade_log: str | None, live_dir: Path, gpr=None, news=None) -> list[dict]:
    if not trade_log:
        return []
    path = Path(trade_log)
    if not path.exists():
        return []
    try:
        trades = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []

    bars_by_symbol = {}
    features_by_symbol = {}
    rows = []
    for trade in trades:
        outcome = trade.get("outcome")
        direction = trade.get("direction")
        symbol = trade.get("symbol")
        if outcome not in {"TP", "SL"} or direction not in {"BUY", "SELL"} or not symbol:
            continue
        if symbol not in bars_by_symbol:
            bars_path = live_dir / f"{symbol}_15.csv"
            if not bars_path.exists():
                continue
            bars_by_symbol[symbol] = read_bars(bars_path)
            features_by_symbol[symbol] = build_features(bars_by_symbol[symbol], gpr, news)
        features = features_by_symbol.get(symbol) or []
        if not features:
            continue
        try:
            trade_ts = int(__import__("datetime").datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            trade_ts = None
        feat = None
        if trade_ts is not None:
            candidates = [row for row in features if int(row["timestamp"]) <= trade_ts]
            feat = candidates[-1] if candidates else None
        feat = feat or features[-1]
        side = "long" if direction == "BUY" else "short"
        rows.append({
            **feat,
            "side_long": 1.0 if side == "long" else 0.0,
            "side_short": 1.0 if side == "short" else 0.0,
            "side": side,
            "tp": trade.get("tp1") or trade.get("tp2") or "",
            "sl": trade.get("sl") or "",
            "horizon": "manual",
            "label": 1 if outcome == "TP" else 0,
            "bars_to_event": "",
        })
    return rows


def refresh_dataset(script_name: str, out_path: str | None) -> None:
    if not out_path:
        return
    script = Path(__file__).resolve().parent / "data_sources" / script_name
    subprocess.run([sys.executable, str(script), "--out", out_path], check=False)


def read_model_summary(model_path: Path) -> dict:
    summary_path = model_path.with_suffix(".json")
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}


def append_history(history_path: str | None, event: dict) -> None:
    if not history_path:
        return
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def metric_value(summary: dict, metric: str) -> float | None:
    if metric == "test_accuracy":
        value = summary.get("test_accuracy")
    else:
        value = (summary.get("metrics") or {}).get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def promote_candidate(candidate_path: Path, active_path: Path) -> None:
    active_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate_path), str(active_path))
    candidate_summary = candidate_path.with_suffix(".json")
    if candidate_summary.exists():
        shutil.move(str(candidate_summary), str(active_path.with_suffix(".json")))


def remove_candidate(candidate_path: Path) -> None:
    for path in (candidate_path, candidate_path.with_suffix(".json")):
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpr", default=None)
    parser.add_argument("--news", default=None)
    parser.add_argument("--refresh-gpr", action="store_true")
    parser.add_argument("--refresh-news", action="store_true")
    parser.add_argument("--seed-label", action="append", default=[])
    parser.add_argument("--trade-log", default=None)
    parser.add_argument("--min-bars", type=int, default=250)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lock", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--promotion-metric", default="test_accuracy")
    parser.add_argument("--min-promotion-score", type=float, default=0.5)
    parser.add_argument("--min-promotion-delta", type=float, default=0.0)
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
        if args.refresh_gpr:
            refresh_dataset("gpr.py", args.gpr)
        if args.refresh_news:
            refresh_dataset("news.py", args.news)
        gpr = load_gpr(args.gpr)
        news = load_news(args.news)
        all_rows = []
        for csv_path in sorted(live_dir.glob("*_15.csv")):
            if count_rows(csv_path) < args.min_bars:
                continue
            bars = read_bars(csv_path)
            label_path = PROCESSED_DIR / f"live_{csv_path.stem}_labels.csv"
            rows = []
            rows.extend(label_rows(bars, "long", 2.5, 1.0, 32, gpr, news))
            rows.extend(label_rows(bars, "short", 2.5, 1.0, 32, gpr, news))
            write_rows(label_path, rows, LABEL_FIELDS)
            all_rows.extend(rows)

        for seed in args.seed_label:
            seed_path = Path(seed)
            if seed_path.exists():
                all_rows.extend(load_csv_rows(seed_path))

        manual_rows = trade_outcome_rows(args.trade_log, live_dir, gpr, news)
        if manual_rows:
            manual_path = PROCESSED_DIR / "manual_trade_outcome_labels.csv"
            write_rows(manual_path, manual_rows, LABEL_FIELDS)
            all_rows.extend(manual_rows)

        if len(all_rows) < args.seq_len + 500:
            print({"trained": False, "reason": "not_enough_labeled_rows", "rows": len(all_rows)})
            return

        combined = PROCESSED_DIR / "live_retrain_labels.csv"
        write_rows(combined, all_rows, LABEL_FIELDS)
        active_path = Path(args.out)
        candidate_path = active_path.with_name(f"{active_path.stem}.candidate.{os.getpid()}{active_path.suffix}")
        current_summary = read_model_summary(active_path)
        report = train_model(
            str(combined),
            seq_len=args.seq_len,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_path=str(candidate_path),
        )
        current_score = metric_value(current_summary, args.promotion_metric)
        candidate_score = metric_value(report, args.promotion_metric)
        min_required = max(
            args.min_promotion_score,
            (current_score + args.min_promotion_delta) if current_score is not None else args.min_promotion_score,
        )
        promoted = candidate_score is not None and candidate_score >= min_required
        reason = "promoted" if promoted else "candidate_below_threshold"
        if promoted:
            promote_candidate(candidate_path, active_path)
        else:
            remove_candidate(candidate_path)

        event = {
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "rows": len(all_rows),
            "manualRows": len(manual_rows),
            "metric": args.promotion_metric,
            "currentScore": current_score,
            "candidateScore": candidate_score,
            "minRequired": min_required,
            "promoted": promoted,
            "reason": reason,
            "report": report,
        }
        append_history(args.history, event)
        print({"trained": True, "rows": len(all_rows), "promoted": promoted, "reason": reason, "report": report})
    finally:
        if lock and lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
