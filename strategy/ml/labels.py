#!/usr/bin/env python3
"""Create TP-before-SL labels using a simple triple-barrier style rule."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import PROCESSED_DIR, ensure_dirs, read_bars, safe_div, write_rows
from strategy.ml.features import FEATURE_FIELDS, build_features, load_gpr, load_news


LABEL_FIELDS = (*FEATURE_FIELDS, "side_long", "side_short", "side", "tp", "sl", "horizon", "label", "bars_to_event")


def label_rows(bars: list[dict], side: str, tp_atr: float, sl_atr: float, horizon: int, gpr=None, news=None) -> list[dict]:
    features = build_features(bars, gpr, news)
    by_ts = {row["timestamp"]: row for row in features}
    rows = []

    for i, bar in enumerate(bars):
        feat = by_ts.get(bar["timestamp"])
        if not feat or i + 1 >= len(bars):
            continue
        atr_pct = float(feat["atr_14_pct"])
        entry = bar["close"]
        risk = entry * atr_pct * sl_atr
        reward = entry * atr_pct * tp_atr
        if risk <= 0 or reward <= 0:
            continue

        if side == "long":
            tp = entry + reward
            sl = entry - risk
        else:
            tp = entry - reward
            sl = entry + risk

        label = 0
        bars_to_event = horizon
        for offset, future in enumerate(bars[i + 1:i + 1 + horizon], start=1):
            hit_tp = future["high"] >= tp if side == "long" else future["low"] <= tp
            hit_sl = future["low"] <= sl if side == "long" else future["high"] >= sl
            if hit_tp and hit_sl:
                label = 0
                bars_to_event = offset
                break
            if hit_tp:
                label = 1
                bars_to_event = offset
                break
            if hit_sl:
                label = 0
                bars_to_event = offset
                break

        rows.append({
            **feat,
            "side_long": 1.0 if side == "long" else 0.0,
            "side_short": 1.0 if side == "short" else 0.0,
            "side": side,
            "tp": tp,
            "sl": sl,
            "horizon": horizon,
            "label": label,
            "bars_to_event": bars_to_event,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gpr", default=None)
    parser.add_argument("--news", default=None)
    parser.add_argument("--side", choices=("long", "short", "both"), default="both")
    parser.add_argument("--tp-atr", type=float, default=2.5)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    bars = read_bars(args.input)
    gpr = load_gpr(args.gpr)
    news = load_news(args.news)
    sides = ["long", "short"] if args.side == "both" else [args.side]
    rows = []
    for side in sides:
        rows.extend(label_rows(bars, side, args.tp_atr, args.sl_atr, args.horizon, gpr, news))

    out = Path(args.out) if args.out else PROCESSED_DIR / (Path(args.input).stem + "_labels.csv")
    write_rows(out, rows, LABEL_FIELDS)
    print(out)


if __name__ == "__main__":
    main()
