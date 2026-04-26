#!/usr/bin/env python3
"""Train a small GRU model on labeled feature sequences.

Requires PyTorch:
  python -m pip install torch
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import MODEL_DIR, ensure_dirs, save_json
from strategy.ml.train_baseline import FEATURES


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for train_rnn.py. Install with: python -m pip install torch") from exc
    return torch, nn


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(([float(row[k]) for k in FEATURES], int(row["label"])))
            except (KeyError, ValueError):
                continue
    return rows


def make_sequences(rows, seq_len: int):
    xs, ys = [], []
    for i in range(seq_len, len(rows)):
        xs.append([rows[j][0] for j in range(i - seq_len, i)])
        ys.append(rows[i][1])
    return xs, ys


def normalize_sequences(xs):
    flat = [v for seq in xs for row in seq for v in row]
    # Feature-wise stats are easier for inference, so compute by column.
    cols = list(zip(*(row for seq in xs for row in seq)))
    means = [sum(col) / len(col) for col in cols]
    stds = []
    for col, mean in zip(cols, means):
        var = sum((v - mean) ** 2 for v in col) / max(1, len(col) - 1)
        stds.append(var ** 0.5 or 1.0)
    norm = [[[(v - means[i]) / stds[i] for i, v in enumerate(row)] for row in seq] for seq in xs]
    return norm, means, stds


def train_model(input_path: str, seq_len: int, hidden: int, epochs: int, batch_size: int, out_path: str):
    torch, nn = require_torch()
    ensure_dirs()
    rows = load_rows(input_path)
    if len(rows) < seq_len + 500:
        raise SystemExit("Need more labeled rows before training an RNN.")

    xs, ys = make_sequences(rows, seq_len)
    xs, means, stds = normalize_sequences(xs)
    split = int(len(xs) * 0.8)
    x_train = torch.tensor(xs[:split], dtype=torch.float32)
    y_train = torch.tensor(ys[:split], dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(xs[split:], dtype=torch.float32)
    y_test = torch.tensor(ys[split:], dtype=torch.float32).view(-1, 1)
    print(f"samples train={len(x_train)} test={len(x_test)} seq_len={seq_len} features={len(FEATURES)}", flush=True)

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=len(FEATURES), hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    model = GRUClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0])
        total_loss = 0.0
        batches = 0
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
        print(f"epoch={epoch + 1} loss={total_loss / max(1, batches):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test))
        preds = probs >= 0.5
        accuracy = (preds.float() == y_test).float().mean().item()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "features": FEATURES,
        "means": means,
        "stds": stds,
        "seq_len": seq_len,
        "hidden": hidden,
        "test_accuracy": accuracy,
    }
    torch.save(checkpoint, out)
    summary = {
        "type": "gru",
        "features": FEATURES,
        "seq_len": seq_len,
        "hidden": hidden,
        "test_accuracy": accuracy,
    }
    save_json(out.with_suffix(".json"), summary)
    print(out)
    print({"test_accuracy": accuracy})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default=str(MODEL_DIR / "rnn.pt"))
    args = parser.parse_args()

    train_model(args.input, args.seq_len, args.hidden, args.epochs, args.batch_size, args.out)


if __name__ == "__main__":
    main()
