#!/usr/bin/env python3
"""Train the mandatory neural core with only the Python standard library.

The model is a compact Elman-style RNN over recent feature sequences. It exists
so the scanner can have a neural gate without requiring PyTorch at runtime.
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


FEATURES = [f for f in FEATURE_FIELDS if f not in {"timestamp", "close"}]


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
    return [([(v - means[i]) / stds[i] for i, v in enumerate(x)], y) for x, y in rows], means, stds


def make_sequences(rows, seq_len):
    seqs = []
    for i in range(seq_len, len(rows)):
        seqs.append(([rows[j][0] for j in range(i - seq_len, i)], rows[i][1]))
    return seqs


def init_matrix(rows, cols, scale=0.08):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def forward(model, seq):
    hidden = [0.0] * model["hidden"]
    for x in seq:
        nxt = []
        for j in range(model["hidden"]):
            activation = model["bh"][j] + dot(model["wxh"][j], x) + dot(model["whh"][j], hidden)
            nxt.append(math.tanh(activation))
        hidden = nxt
    logit = model["bo"] + dot(model["why"], hidden)
    return sigmoid(logit), hidden


def train(seqs, feature_count, hidden, epochs, lr):
    model = {
        "hidden": hidden,
        "wxh": init_matrix(hidden, feature_count),
        "whh": init_matrix(hidden, hidden),
        "bh": [0.0] * hidden,
        "why": [random.uniform(-0.08, 0.08) for _ in range(hidden)],
        "bo": 0.0,
    }
    for _ in range(epochs):
        random.shuffle(seqs)
        for seq, y in seqs:
            p, h = forward(model, seq)
            err = p - y
            for j in range(hidden):
                model["why"][j] -= lr * err * h[j]
            model["bo"] -= lr * err

            # Truncated one-step recurrent update keeps training simple and stable.
            last_x = seq[-1]
            dh = [err * model["why"][j] * (1 - h[j] * h[j]) for j in range(hidden)]
            prev_h = [0.0] * hidden
            if len(seq) > 1:
                _, prev_h = forward(model, seq[:-1])
            for j in range(hidden):
                for k, value in enumerate(last_x):
                    model["wxh"][j][k] -= lr * dh[j] * value
                for k, value in enumerate(prev_h):
                    model["whh"][j][k] -= lr * dh[j] * value
                model["bh"][j] -= lr * dh[j]
    return model


def evaluate(model, seqs):
    if not seqs:
        return {"samples": 0, "accuracy": None, "log_loss": None}
    correct = 0
    loss = 0.0
    for seq, y in seqs:
        p, _ = forward(model, seq)
        correct += int((p >= 0.5) == bool(y))
        loss += -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
    return {"samples": len(seqs), "accuracy": correct / len(seqs), "log_loss": loss / len(seqs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--out", default=str(MODEL_DIR / "neural_core.json"))
    args = parser.parse_args()

    ensure_dirs()
    rows = load_dataset(args.input)
    if len(rows) < args.seq_len + 200:
        raise SystemExit("Need at least seq_len + 200 labeled rows for neural training.")
    random.seed(42)
    random.shuffle(rows)
    rows, means, stds = normalize(rows)
    seqs = make_sequences(rows, args.seq_len)
    split = int(len(seqs) * 0.8)
    model = train(seqs[:split], len(FEATURES), args.hidden, args.epochs, args.lr)
    report = {
        "type": "standard_library_rnn",
        "features": FEATURES,
        "means": means,
        "stds": stds,
        "seq_len": args.seq_len,
        **model,
        "train": evaluate(model, seqs[:split]),
        "test": evaluate(model, seqs[split:]),
    }
    save_json(args.out, report)
    print(args.out)
    print(report["test"])


if __name__ == "__main__":
    main()
