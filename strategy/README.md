# SMC + Pivot Boss Multi-TF Trading Scanner

Automated 15-minute scanner for **BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, XAUUSD** that monitors a live TradingView Desktop chart, runs a multi-timeframe SMC + Floor Pivot strategy, and pushes high-conviction alerts to your Telegram. Trade outcomes you reply with become training data — component weights adjust via online EWMA so accuracy improves over time.

## Strategy synthesis

Distilled from your reading list:

- **Smart Money Concepts** (VasilyTrader) — BOS / CHoCH structure, Order Blocks, Fair Value Gaps, Liquidity Sweeps, premium/discount zones
- **ICT (Inner Circle Trader)** — Killzones (Asia/London/NY), Power of 3 (PO3), OTE (Optimal Trade Entry 62–79% Fib), Market Structure Shift (MSS), Silver Bullet, New Day Opening Gap (NDOG), Buy/Sell-side Liquidity (BSL/SSL), Breaker Blocks, Displacement candle confirmation
- **Secrets of a Pivot Boss** (Ochoa) — Standard + Expanded Floor Pivots, Camarilla equation, Pivot Trend Analysis, Hot Zones
- **Inside the Black Box** (Narang) + **Quantitative Trading** (Chan) — systematic alpha model, risk filtering, sharpe-aware sizing
- **Max Dama on Automated Trading** — EWMA online updates, regression-style signal calibration
- **Building Winning Algo Trading Systems** (Davey) — robustness checks, no curve-fitting

### ICT Killzones
| Session | UTC Time | EST Time | Notes |
|---------|----------|----------|-------|
| Asia | 00:00–04:00 | 8PM–12AM | Accumulation phase — low volatility |
| London | 07:00–10:00 | 2AM–5AM | Manipulation sweep of Asia range |
| NY AM | 12:00–15:00 | 7AM–10AM | Distribution — real directional move |
| Silver Bullet | 15:00–16:00 | 10AM–11AM | Highest quality FVG entries on 15M |

### ICT Signal Chain (ideal setup)
1. **Accumulation** (Asia): price consolidates, builds BSL/SSL above/below range
2. **Manipulation** (London): sweeps Asia high or low — triggers retail stops (Power of 3 phase 2)
3. **MSS**: after the sweep, 15M close beyond the opposing swing confirms direction change
4. **OTE**: price retraces 62–79% of the MSS impulse → entry zone
5. **Distribution** (NY): price delivers to TP1 (Daily R1/S1) and TP2 (R2/S2)

### Multi-TF hierarchy
| Timeframe | Role |
|-----------|------|
| Weekly  | macro trend |
| Daily   | structure + Floor Pivots (P, R1-R4, S1-S4, TC, BC) + Camarilla |
| 4H      | Order Blocks, FVGs, premium/discount zone |
| 1H      | BOS / CHoCH confirmation |
| **15M** | **entry trigger** — liquidity sweep + imbalance candle |

### Signal scoring (max ≈ 13.25)
- **HTF alignment** (Weekly + Daily + Pivot trend agree) → `2.5`
- **4H Order Block / FVG / Breaker Block at price** → `2.0` (ICT OB with displacement = full; generic = 75%)
- **15M liquidity sweep** (SSL/BSL sweep = full `2.0`; generic swing sweep = `1.6`)
- **1H/15M structure confirmation** (ICT MSS = `1.5`; plain BOS = `1.1`)
- **Floor Pivot or Camarilla H3/L3 confluence** (within 0.3%) → `1.0`
- **ICT Killzone active** (Asia/London/NY) → `1.0`; Silver Bullet window → `1.0` (full)
- **ICT OTE** (62–79% Fibonacci retracement of impulse) → `0.75`
- **ICT NDOG** (New Day Opening Gap magnetism) → `0.5`
- **Power of 3 distribution phase** → `0.25` bonus
- **RR ≥ 1:3 bonus** → `1.0`
- **Additional research confluence** (CHOCH/MSS quality + Fib/OTE location + multi-scale structure alignment) → capped by `weights.additionalConfluence`

A signal fires when:
- score ≥ **6.5/10**
- RR ≥ **1:2.5**
- 3+ directional conditions agree
- no duplicate alert on same symbol+direction in last 4h

### SL / TP
- **SL** anchored to OB/FVG low/high minus 0.5×ATR(14) buffer (sweep-aware)
- **TP1** at 1:2.5 RR or nearest Daily R1/S1, whichever comes first
- **TP2** at next major pivot level (R2/S2)

## File layout
```
strategy/
├── config.json          # Telegram token, pairs, params, weights
├── cdp_client.js        # Standalone CDP connection to TradingView (port 9222)
├── pivots.js            # Floor Pivots + Camarilla
├── analyzer.js          # SMC structure, OB/FVG, sweep detection, scoring
├── market_store.js      # Appends every scan's OHLCV bars + signal snapshots
├── ml_bridge.js         # Optional Python ML/RNN probability filter
├── ml_retrainer.js      # Starts background PyTorch retraining from live data
├── ml/                  # Open-data downloaders, features, labels, baseline, GRU/RNN
├── telegram.js          # Bot API wrapper (send + poll updates)
├── learning.js          # EWMA component-edge tracking + trade log
├── scanner.js           # Main entry point (called every 15 min)
├── setup.js             # First-run: discovers your chat_id
├── com.tradingview.scanner.plist  # launchd schedule
├── trades.json          # All fired alerts + outcomes (auto-created)
├── weights.json         # EWMA accuracy per signal component (auto-created)
└── scanner.log          # Last run output (auto-created)
```

Local-only runtime and legacy files are intentionally ignored by Git:

- `strategy/config.json` contains private Telegram settings; commit `strategy/config.example.json` instead.
- `strategy/scanner.run.lock` is created while a scan is running so overlapping 15-minute triggers are skipped.
- `strategy/trades.json`, `strategy/weights.json`, `strategy/ml/data/**`, and `strategy/ml/reports/**` are local learning/runtime state.
- `alerts_log/` and root `rules.json` are legacy local files and are not used by the current scanner.

## Core ML / Neural Layer

The scanner calls a Python/PyTorch recurrent sidecar after the rule-based setup is built. The active checkpoint can be GRU or LSTM; existing GRU checkpoints remain valid, while new retrains can be configured to train LSTM candidates. The model is a required neural gate:

```
OHLCV history → SMC/ICT/Pivot setup → ML probability → Telegram alert
```

If the model is missing or below the configured probability threshold, the alert is blocked. The model receives price/volume features plus explicit SMC/ICT, auction-market, side, official AI-GPR geopolitical-risk features, and near-real-time headline-pressure features from GDELT/RSS feeds.

The ML feature layer also includes research-grade confluence columns:

- CHOCH/MSS quality: direction, break strength, recency, pivot regularity, swing-sequence quality.
- Fibonacci/OTE location: retracement zone, distance to 0.618/0.705/0.79, leg size versus ATR, RR proxy, OTE quality.
- Multi-scale structure: short/intermediate/major trend alignment, pivot prominence, distance to major high/low, structure confidence.

The JavaScript analyzer exposes the same idea as `details.additionalConfluence` and a capped `additionalConfluence` score contribution. It is deliberately soft: it can add conviction to an already directional setup, but it does not bypass RR, continuity, or the RNN gate.

Every scanner cycle can also append live OHLCV bars and signal snapshots:

```
strategy/ml/data/live/<SYMBOL>_<TF>.csv
strategy/ml/data/live/scan_signals.jsonl
```

Background retraining uses the accumulated live 15-minute bars, creates TP-before-SL labels, combines them with seed labels and manual TP/SL outcomes, then trains a candidate recurrent model. Manual TP/SL feedback carries extra outcome metadata such as duration, exit price, pips/points captured, and realized R. The candidate is logged to `strategy/ml/reports/retrain_history.jsonl` and only replaces `strategy/ml/models/rnn.pt` when it meets the configured promotion metric.

Runtime data and reports are ignored by Git:

```
strategy/ml/data/raw/
strategy/ml/data/processed/
strategy/ml/data/live/
strategy/ml/reports/
```

The active RNN is the tracked pair `strategy/ml/models/rnn.pt` and
`strategy/ml/models/rnn.json`. `rnn.pt` is the PyTorch checkpoint used for
inference; `rnn.json` is the human-readable summary for the same model.

To inspect the active neural gate and recent promotion decisions:

```bash
npm run model:report
```

To enable the model in `config.json`:

```json
"ml": {
  "enabled": true,
  "python": ".venv/bin/python",
  "modelPath": "strategy/ml/models/rnn.pt",
  "minProbability": 0.55,
  "failOpen": false,
  "timeoutMs": 8000,
  "retrain": {
    "enabled": true,
    "minBars": 250,
    "seqLen": 32,
    "hidden": 32,
    "cell": "lstm",
    "epochs": 3,
    "historyPath": "strategy/ml/reports/retrain_history.jsonl",
    "promotionMetric": "utility_score",
    "minPromotionScore": 0.4,
    "minPromotionDelta": 0
  }
}
```

See `ml/README.md` for open-data download, labeling, baseline training, and RNN training commands.

## Setup (3 steps)

### 1. Capture your Telegram chat_id (one-time)

Open Telegram, find your bot (you'll know its name from BotFather), and start a chat. Then run:

```bash
cd ~/tradingview-mcp/strategy
node setup.js
```

Send `/start` to the bot from Telegram. The script captures your `chat_id` into `config.json` and sends a confirmation message back.

### 2. Verify TradingView Desktop is running with CDP

```bash
node ~/tradingview-mcp/src/cli/index.js health   # if a CLI exists
# OR just run the scanner once manually:
node ~/tradingview-mcp/strategy/scanner.js
```

The first manual run is the smoke test — it'll switch through every pair/timeframe and log results to console. If it errors with "TradingView not found on CDP port 9222", launch the desktop app and ensure it was started with remote debugging (the MCP's `tv_launch` tool handles this).

### 3. Schedule it (every 15 minutes)

```bash
cp ~/tradingview-mcp/strategy/com.tradingview.scanner.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradingview.scanner.plist
launchctl start com.tradingview.scanner    # run once now to verify
```

The job fires at minute :00, :15, :30, :45 of every hour. Logs stream to `scanner.log` and `scanner.error.log`.

To pause:
```bash
launchctl unload ~/Library/LaunchAgents/com.tradingview.scanner.plist
```

## How feedback / learning works

Every alert includes a short trade ID like `a3f9c1d2`. When the trade closes:

```
TP HIT a3f9c1d2     ← reply in Telegram
SL HIT a3f9c1d2
TP HIT a3f9c1d2 216.25  ← optional exact exit price
```

The next scan cycle picks up the reply, marks the trade in `trades.json`, stores outcome duration, approximate or exact exit price, pips/points captured, and realized R, then updates EWMA edge for **every component that contributed to that signal**. Components with consistently high edge get scaled up; weak ones decay. Adjustments only kick in after 5+ samples per component.

Trade duration is not used as a live feature for the same signal because that would leak future information. It is stored as outcome metadata and used in retraining/evaluation so the model can learn which historical setup types resolved efficiently versus slowly or poorly.

### Telegram commands
- `/stats` — current win-rate + per-component edge
- `/trades` — last 10 alerts and their outcomes

## Risk & sanity caveats

- The scanner reads from your **live chart**. It rapidly switches symbols/timeframes during each cycle — don't be surprised if the chart jumps around at :00/:15/:30/:45.
- No order is ever placed. Alerts are *signals*; you decide whether to take them.
- The minimum 1:2.5 RR is enforced *before* alerts fire — sub-RR setups are silently skipped.
- XAUUSD is auto-skipped Friday 17:00 → Sunday 18:00 local time.
- The Telegram token in `config.json` is a secret. Don't commit it. Rotate via @BotFather if it leaks.
- The "reinforcement learning" here is intentionally simple (EWMA on per-component win rate) — not deep RL. It nudges weights immediately, while the neural model retrains separately from accumulated bars and closed-trade outcomes.
