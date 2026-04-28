# ML / RNN Research Sidecar

This folder adds an optional machine-learning layer to the existing scanner. It is not a replacement for the current SMC/ICT/Pivot signal engine. The intended role is narrower:

> Estimate whether a rule-based setup has historically reached TP before SL under similar recent bar conditions.

## Workflow

1. Download open historical data.
2. Build auditable OHLCV, SMC/ICT, auction, geopolitical, and headline-pressure features.
3. Create TP-before-SL labels.
4. Train a simple logistic baseline.
5. Train a GRU/RNN only after the baseline is useful.
6. Let the Node scanner use the model as an optional filter.

## Download Crypto From Binance

```bash
python -m strategy.ml.data_sources.binance \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
  --interval 15m \
  --start 2023-01-01 \
  --end 2024-01-01
```

## Download FX / Metals From Dukascopy

```bash
python -m strategy.ml.data_sources.dukascopy \
  --symbols EURUSD GBPUSD USDJPY XAUUSD \
  --interval 15m \
  --start 2024-01-01 \
  --end 2024-01-08
```

Tick-level Dukascopy downloads can be large. Start with a few days, then scale.

## Build Labels

```bash
python -m strategy.ml.labels \
  --input strategy/ml/data/raw/binance/BTCUSDT_15m_2023-01-01_2024-01-01.csv \
  --side both \
  --tp-atr 2.5 \
  --sl-atr 1.0 \
  --horizon 32
```

The default label is `1` when TP is hit before SL within the next 32 bars.

## Train Baseline

```bash
python -m strategy.ml.train_baseline \
  --input strategy/ml/data/processed/BTCUSDT_15m_2023-01-01_2024-01-01_labels.csv
```

This writes `strategy/ml/models/baseline.json`. Use this before training a neural net.

## Train GRU / RNN

```bash
python -m pip install torch
python -m strategy.ml.train_rnn \
  --input strategy/ml/data/processed/BTCUSDT_15m_2023-01-01_2024-01-01_labels.csv \
  --seq-len 64
```

This writes `strategy/ml/models/rnn.pt`.

## Geopolitical / News Features

The model uses two geopolitical context layers:

- Official daily AI-GPR data from Matteo Iacoviello and Jonathan Tong:
  `strategy/ml/data/raw/gpr/ai_gpr_data_daily.csv`
- Near-real-time headline pressure from GDELT plus broad RSS feeds:
  `strategy/ml/data/raw/news/geopolitical_news_daily.csv`

Refresh them manually:

```bash
python -m strategy.ml.data_sources.gpr
python -m strategy.ml.data_sources.news
```

Hourly live retraining refreshes both datasets when `refreshGpr` and
`refreshNews` are enabled in `strategy/config.json`.

## Confluence Feature Families

The RNN feature schema includes the rule vocabulary plus three additional
research layers:

- CHOCH/MSS quality: `choch_*`, `mss_*`, pivot regularity, and swing-sequence
  quality.
- Fibonacci/OTE location quality: retracement zone flags, distance to
  0.618/0.705/0.79, leg size versus ATR, RR proxy, and `fib_ote_quality`.
- Multi-scale structure: short/intermediate/major trend values, alignment
  score, pivot prominence, distance to major high/low, and structure
  confidence.

Existing checkpoints remain loadable because inference reads the feature list
from the active checkpoint. These new columns become part of the neural gate
after a retrain creates and promotes a checkpoint trained on the expanded
schema.

Each retrain writes a promotion event to
`strategy/ml/reports/retrain_history.jsonl`. The hourly job trains a
candidate model first, compares it with the active model using
`promotionMetric` (default `utility_score`), and promotes the candidate only
when it reaches `minPromotionScore` and beats the current model by at least
`minPromotionDelta`. `utility_score` combines AUC, balanced accuracy, and F1
so a majority-class model cannot win promotion only by predicting "no TP" for
everything.

## Scanner Integration

The scanner integration is optional and controlled by `config.json`:

```json
"ml": {
  "enabled": false,
  "python": "python3",
  "modelPath": "strategy/ml/models/baseline.json",
  "minProbability": 0.55,
  "failOpen": true
}
```

With `failOpen: true`, missing models or Python dependency issues do not block alerts. The scanner records the ML status in `signal.details.ml`.
