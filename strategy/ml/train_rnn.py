#!/usr/bin/env python3
"""Train a small GRU model on labeled feature sequences.

Requires PyTorch:
  python -m pip install torch
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
from strategy.ml.train_baseline import FEATURES


def feature_value(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


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
                rows.append(([feature_value(row, k) for k in FEATURES], int(row["label"])))
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


def binary_auc(labels: list[int], probs: list[float]) -> float | None:
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(probs, labels), key=lambda item: item[0])
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        pos_rank_sum += sum(avg_rank for _, label in ordered[i:j] if label == 1)
        rank += j - i
        i = j

    return (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(labels: list[int], probs: list[float]) -> dict:
    preds = [1 if prob >= 0.5 else 0 for prob in probs]
    total = max(1, len(labels))
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = sensitivity
    f1 = (2 * precision * recall / max(1e-12, precision + recall)) if precision or recall else 0.0
    auc = binary_auc(labels, probs)
    eps = 1e-7
    log_loss = -sum(
        label * math.log(min(1 - eps, max(eps, prob))) +
        (1 - label) * math.log(min(1 - eps, max(eps, 1 - prob)))
        for label, prob in zip(labels, probs)
    ) / total
    brier = sum((prob - label) ** 2 for label, prob in zip(labels, probs)) / total
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "log_loss": log_loss,
        "brier": brier,
        "positive_rate": sum(preds) / total,
        "label_positive_rate": sum(labels) / total,
        "utility_score": ((auc if auc is not None else 0.5) * 0.45) + (((sensitivity + specificity) / 2) * 0.35) + (f1 * 0.20),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_model(input_path: str, seq_len: int, hidden: int, epochs: int, batch_size: int, out_path: str):
    torch, nn = require_torch()
    random.seed(42)
    torch.manual_seed(42)
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
    positives = float(y_train.sum().item())
    negatives = float(y_train.numel() - positives)
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        probs_list = [float(value) for value in probs.view(-1).tolist()]
        y_test_list = [int(value) for value in y_test.view(-1).tolist()]
        metrics = classification_metrics(y_test_list, probs_list)
        accuracy = metrics["accuracy"]

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
        "metrics": metrics,
    }
    torch.save(checkpoint, out)
    summary = {
        "type": "gru",
        "features": FEATURES,
        "seq_len": seq_len,
        "hidden": hidden,
        "test_accuracy": accuracy,
        "metrics": metrics,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
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
