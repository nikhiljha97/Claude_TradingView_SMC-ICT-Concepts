#!/usr/bin/env python3
"""Train a tiny auditable logistic baseline.

This is a sanity check before the RNN. If a simple model cannot beat noise, a
larger sequence model is likely just memorizing.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import MODEL_DIR, ensure_dirs, save_json
from strategy.ml.features import FEATURE_FIELDS


FEATURES = [f for f in FEATURE_FIELDS if f not in {"timestamp", "close"}] + ["side_long", "side_short"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def load_dataset(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(([float(row[k]) for k in FEATURES], int(row["label"])))
            except (KeyError, ValueError):
                continue
    return rows


def normalize(rows):
    cols = list(zip(*(x for x, _ in rows)))
    means = [sum(col) / len(col) for col in cols]
    stds = []
    for col, m in zip(cols, means):
        var = sum((v - m) ** 2 for v in col) / max(1, len(col) - 1)
        stds.append(math.sqrt(var) or 1.0)
    norm = [([(v - means[i]) / stds[i] for i, v in enumerate(x)], y) for x, y in rows]
    return norm, means, stds


def train(rows, epochs: int, lr: float):
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    for _ in range(epochs):
        random.shuffle(rows)
        for x, y in rows:
            p = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            err = p - y
            for i, value in enumerate(x):
                weights[i] -= lr * err * value
            bias -= lr * err
    return weights, bias


def evaluate(rows, weights, bias):
    if not rows:
        return {"accuracy": None, "samples": 0}
    correct = 0
    loss = 0.0
    for x, y in rows:
        p = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
        correct += int((p >= 0.5) == bool(y))
        loss += -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
    return {"accuracy": correct / len(rows), "log_loss": loss / len(rows), "samples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out", default=str(MODEL_DIR / "baseline.json"))
    args = parser.parse_args()

    ensure_dirs()
    rows = load_dataset(args.input)
    if len(rows) < 200:
        raise SystemExit("Need at least 200 labeled rows for a useful baseline.")
    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]
    train_rows, means, stds = normalize(train_rows)
    test_rows = [([(v - means[i]) / stds[i] for i, v in enumerate(x)], y) for x, y in test_rows]
    weights, bias = train(train_rows, args.epochs, args.lr)
    report = {
        "type": "logistic_baseline",
        "features": FEATURES,
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
        "train": evaluate(train_rows, weights, bias),
        "test": evaluate(test_rows, weights, bias),
    }
    save_json(args.out, report)
    print(args.out)
    print(report["test"])


if __name__ == "__main__":
    main()
