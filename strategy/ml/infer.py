#!/usr/bin/env python3
"""Score a current setup with the trained ML sidecar.

Input is JSON on stdin:
  {"signal": {...}, "bars15M": [...], "modelPath": "strategy/ml/models/rnn.pt"}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import load_json
from strategy.ml.features import build_features, load_gpr, load_news


def normalize_timestamp(value) -> int | str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return int(numeric * 1000) if numeric < 10_000_000_000 else int(numeric)


def normalize_bars(bars: list[dict]) -> list[dict]:
    normalized = []
    for bar in bars:
        try:
            timestamp = normalize_timestamp(bar.get("timestamp") or bar.get("time") or bar.get("t") or bar.get("date"))
            normalized.append({
                "timestamp": timestamp,
                "open": float(bar.get("open", bar.get("o"))),
                "high": float(bar.get("high", bar.get("h"))),
                "low": float(bar.get("low", bar.get("l"))),
                "close": float(bar.get("close", bar.get("c"))),
                "volume": float(bar.get("volume", bar.get("v", 0)) or 0),
            })
        except (TypeError, ValueError):
            continue
    return normalized


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def signal_side(signal: dict) -> tuple[float, float]:
    direction = str(signal.get("direction", "")).upper()
    return (1.0 if direction == "BUY" else 0.0, 1.0 if direction == "SELL" else 0.0)


def attach_side(rows: list[dict], signal: dict) -> list[dict]:
    side_long, side_short = signal_side(signal)
    for row in rows:
        row["side_long"] = side_long
        row["side_short"] = side_short
    return rows


def score_baseline(model: dict, bars: list[dict], signal: dict, gpr=None, news=None) -> dict:
    rows = attach_side(build_features(bars, gpr, news), signal)
    if not rows:
        return {"success": False, "reason": "not_enough_bars"}
    latest = rows[-1]
    values = []
    for i, name in enumerate(model["features"]):
        value = float(latest.get(name, 0.0))
        values.append((value - model["means"][i]) / model["stds"][i])
    logit = sum(w * v for w, v in zip(model["weights"], values)) + model["bias"]
    probability = sigmoid(logit)
    return {
        "success": True,
        "modelType": "logistic_baseline",
        "probability": probability,
        "thresholdPassed": probability >= 0.5,
    }


def score_rnn(model_path: Path, bars: list[dict], signal: dict, gpr=None, news=None) -> dict:
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"success": False, "reason": "torch_not_installed"}

    checkpoint = torch.load(model_path, map_location="cpu")
    rows = attach_side(build_features(bars, gpr, news), signal)
    seq_len = checkpoint["seq_len"]
    if len(rows) < seq_len:
        return {"success": False, "reason": "not_enough_bars"}

    features = checkpoint["features"]
    means = checkpoint["means"]
    stds = checkpoint["stds"]
    seq = []
    for row in rows[-seq_len:]:
        seq.append([(float(row.get(name, 0.0)) - means[i]) / stds[i] for i, name in enumerate(features)])

    cell = checkpoint.get("cell") or checkpoint.get("type") or "gru"

    class RecurrentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            recurrent = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = recurrent(input_size=len(features), hidden_size=checkpoint["hidden"], batch_first=True)
            self.head = nn.Linear(checkpoint["hidden"], 1)

        def forward(self, x):
            _, state = self.rnn(x)
            h = state[0] if cell == "lstm" else state
            return self.head(h[-1])

    state_dict = checkpoint["state_dict"]
    if any(key.startswith("gru.") or key.startswith("lstm.") for key in state_dict):
        state_dict = {
            key.replace("gru.", "rnn.", 1).replace("lstm.", "rnn.", 1): value
            for key, value in state_dict.items()
        }
    model = RecurrentClassifier()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.tensor([seq], dtype=torch.float32))).item()
    return {
        "success": True,
        "modelType": cell,
        "probability": probability,
        "thresholdPassed": probability >= 0.5,
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpr", default=None)
    parser.add_argument("--news", default=None)
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    model_path = Path(args.model or payload.get("modelPath", "strategy/ml/models/rnn.pt"))
    bars = normalize_bars(payload.get("bars15M") or payload.get("bars") or [])
    signal = payload.get("signal") or {}
    gpr = load_gpr(args.gpr or payload.get("gprPath"))
    news = load_news(args.news or payload.get("newsPath"))

    if not model_path.exists():
        print(json.dumps({"success": False, "reason": "model_missing", "modelPath": str(model_path)}))
        return

    if model_path.suffix == ".json":
        model = load_json(model_path)
        result = score_baseline(model, bars, signal, gpr, news)
    elif model_path.suffix == ".pt":
        result = score_rnn(model_path, bars, signal, gpr, news)
    else:
        result = {"success": False, "reason": "unsupported_model_type"}

    result["modelPath"] = str(model_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
