#!/usr/bin/env python3
"""Summarize the active RNN checkpoint and recent promotion history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy.ml.common import MODEL_DIR, ROOT


FAMILIES = {
    "price": ("ret_", "range_", "body_", "upper_", "lower_", "atr_", "vol_", "trend_", "compression_"),
    "smc_ict": ("smc_", "ict_"),
    "auction": ("auction_",),
    "confluence": ("choch_", "mss_", "pivot_", "swing_", "fib_", "ms_"),
    "geopolitical": ("gpr_", "news_"),
    "side": ("side_",),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_history(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def feature_family_counts(features: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in FAMILIES}
    counts["other"] = 0
    for feature in features:
        matched = False
        for name, prefixes in FAMILIES.items():
            if feature.startswith(prefixes):
                counts[name] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    return counts


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_report(model: dict, history: list[dict]) -> None:
    metrics = model.get("metrics") or {}
    features = model.get("features") or []
    print("Active RNN model")
    print("================")
    print(f"type: {model.get('type', 'unknown')}")
    print(f"features: {len(features)}")
    print(f"seq_len: {model.get('seq_len', 'n/a')}")
    print(f"hidden: {model.get('hidden', 'n/a')}")
    print(f"train_samples: {model.get('train_samples', 'n/a')}")
    print(f"test_samples: {model.get('test_samples', 'n/a')}")
    print("")
    print("Metrics")
    print("-------")
    for key in ("utility_score", "accuracy", "balanced_accuracy", "auc", "precision", "recall", "f1", "log_loss", "brier", "positive_rate", "label_positive_rate"):
        print(f"{key}: {fmt(metrics.get(key))}")
    print(f"confusion: TP={metrics.get('tp', 'n/a')} FP={metrics.get('fp', 'n/a')} TN={metrics.get('tn', 'n/a')} FN={metrics.get('fn', 'n/a')}")
    print("")
    print("Feature families")
    print("----------------")
    for name, count in feature_family_counts(features).items():
        print(f"{name}: {count}")
    if history:
        print("")
        print("Recent retrains")
        print("---------------")
        for event in history:
            print(
                f"{event.get('timestamp', 'unknown')} | promoted={event.get('promoted')} "
                f"| metric={event.get('metric')} | candidate={fmt(event.get('candidateScore'))} "
                f"| current={fmt(event.get('currentScore'))} | rows={event.get('rows', 'n/a')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_DIR / "rnn.json"))
    parser.add_argument("--history", default=str(ROOT / "reports" / "retrain_history.jsonl"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model = load_json(Path(args.model))
    history = load_history(Path(args.history), args.limit)
    if args.json:
        print(json.dumps({"model": model, "history": history, "featureFamilies": feature_family_counts(model.get("features") or [])}, indent=2, sort_keys=True))
    else:
        print_report(model, history)


if __name__ == "__main__":
    main()
