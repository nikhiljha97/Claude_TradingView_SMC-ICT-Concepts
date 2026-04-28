# TradingView MCP + Local Neural Trading Scanner

This repository has two connected parts:

1. **TradingView MCP Bridge** - a local bridge that lets an AI assistant or CLI interact with your running TradingView Desktop app through Chrome DevTools Protocol.
2. **Local Trading Scanner** - a 15-minute signal scanner that reads your own TradingView chart, evaluates configured markets, applies rule-based confluence, gates signals through a PyTorch GRU/LSTM recurrent model, sends Telegram alerts, records outcomes, and retrains locally.

Plain English version: this system watches your charts on your laptop, checks whether a trade setup is good enough, asks a neural model for confirmation, and messages you on Telegram. It does **not** place trades for you.

> [!WARNING]
> This project is not affiliated with, endorsed by, or associated with TradingView Inc. It interacts with your locally running TradingView Desktop application through Chrome DevTools Protocol. Review the [Disclaimer](#disclaimer) before use.

> [!IMPORTANT]
> This is research and decision-support software, not financial advice and not an execution system. It never places broker orders. You are responsible for every trading decision and for ensuring your usage complies with TradingView, exchange, broker, and data-provider terms.

> [!CAUTION]
> TradingView Desktop internals can change without notice. The bridge depends on the local Electron debug interface and may break after TradingView updates.

## Current Production State

The live local scanner is designed to run on macOS through `launchd`.

- Schedule: every 15 minutes at `:00`, `:15`, `:30`, and `:45`
- Watchlists: weekday and weekend markets from `strategy/config.json`
- Alerts: Telegram messages only when all gates pass
- Risk filter: minimum actual risk/reward of `1:2.5`
- ML gate: required PyTorch recurrent model using 76 engineered features; GRU checkpoints remain supported and LSTM retrains can be enabled
- Learning loop: live OHLCV storage, Telegram feedback, hourly retraining, model promotion ledger
- Safety: no broker connection, no order placement, no automatic execution

The scanner runs locally. Codex, Claude, or any other AI agent does **not** need to stay connected after the LaunchAgent is started.

## Who This Is For

For a layperson:

- The system checks charts every 15 minutes.
- It only sends a Telegram alert when the setup passes multiple filters.
- It asks you whether you took the trade.
- If you later reply `TP HIT <tradeId>` or `SL HIT <tradeId>`, that outcome becomes learning data.

For a software engineer:

- `strategy/scanner.js` orchestrates chart control, data capture, signal generation, ML inference, Telegram messaging, and background retraining.
- `launchd` schedules the scanner as a local recurring job.
- Runtime state is intentionally local and mostly ignored by Git.
- Model checkpoints are versioned only when explicitly committed.

For a data scientist:

- The active neural gate is a PyTorch recurrent model, currently GRU unless an LSTM candidate has been promoted.
- The model receives price/volume, SMC/ICT, auction, geopolitical/news, side, and confluence features.
- Retraining writes candidate models first.
- Candidate promotion uses a utility score, not plain accuracy, to avoid majority-class models that simply predict "no TP".

For an algo trader:

- The rule engine provides candidate setups.
- The neural model acts as a probability gate, not a standalone alpha engine.
- RR, continuity, higher-timeframe bias, and model probability must all cooperate before an alert fires.
- This is signal generation and research infrastructure, not a production execution stack.

## Signal Flow

```text
TradingView Desktop chart
        |
        v
OHLCV capture across W / D / 4H / 1H / 15M
        |
        v
Rule engine: SMC + ICT + pivots + auction logic
        |
        v
Risk, continuity, and duplicate-suppression checks
        |
        v
PyTorch GRU/LSTM probability gate
        |
        v
Telegram alert, only if the setup remains valid
        |
        v
User feedback: YES / NO / TP HIT / SL HIT
        |
        v
Local labels, retraining, candidate promotion ledger
```

## Core Trading Logic

The scanner combines interpretable market logic with a neural gate.

Rule and confluence layers include:

- Weekly, daily, 4H, 1H, and 15M structure
- SMC concepts: BOS, CHoCH, order blocks, fair value gaps, liquidity sweeps
- ICT concepts: MSS, OTE, killzones, PO3, NDOG, BSL/SSL
- Floor pivots and Camarilla levels
- Auction behavior: failed auctions, initiative moves, balance/compression
- Risk/reward validation after target adjustment
- Continuity logic to reduce flip-flopping between buy and sell signals

Neural feature families include:

| Family | Examples |
|---|---|
| Price/volume | returns, ATR, range, wick, compression, volume z-score |
| SMC/ICT | BOS, sweeps, equal highs/lows, FVG, premium/discount, killzones |
| Auction | failed up/down, initiative buy/sell, balance |
| Confluence | CHOCH/MSS quality, Fib/OTE quality, multi-scale structure alignment |
| Geopolitical/news | AI-GPR, oil/non-oil risk, RSS/GDELT headline pressure |
| Side | long/short encoding |

The current active neural model has `76` features. Check your exact live model with:

```bash
npm run model:report
```

## Model Training And Promotion

The system retrains from local data and user feedback.

What gets stored locally:

- Captured OHLCV bars in `strategy/ml/data/live/`
- Generated labels in `strategy/ml/data/processed/`
- Telegram trade history in `strategy/trades.json`
- Model reports in `strategy/ml/reports/`

Retraining process:

1. Build labels from live 15-minute bars.
2. Add manual TP/SL outcomes from Telegram replies.
3. Add seed labels when configured.
4. Train a candidate GRU or LSTM checkpoint.
5. Compare candidate metrics against the active checkpoint.
6. Promote only if the candidate passes the configured promotion metric.

Promotion currently uses `utility_score`, which combines AUC, balanced accuracy, and F1. This is intentional. Plain accuracy can look good when a model mostly predicts the majority class, which is not useful for a trading gate.

## Telegram Workflow

When an alert fires, the message includes:

- Symbol and direction
- Entry, stop, TP1, TP2
- Risk/reward
- Confidence and score breakdown
- Setup reasoning
- ML probability
- Research confluence diagnostics when present
- Trade ID

Reply examples:

```text
YES abc123
NO abc123
TP HIT abc123
SL HIT abc123
TP HIT abc123 216.25
```

The optional number is the exact exit price. If you omit it, the system uses
the planned TP1 or SL as the approximate exit. Closed trades store duration,
pips/points captured, and realized R for retraining/evaluation.

Commands:

```text
/stats
/trades
```

## Important Files

| Path | Purpose |
|---|---|
| `README.md` | Repo-level overview and production guide |
| `strategy/README.md` | Scanner-specific trading and operations documentation |
| `strategy/ml/README.md` | ML, data, feature, and retraining documentation |
| `strategy/config.example.json` | Safe config template |
| `strategy/config.json` | Private local config with Telegram secrets, ignored by Git |
| `strategy/scanner.js` | Main 15-minute scanner |
| `strategy/analyzer.js` | Rule engine and interpretable confluence |
| `strategy/ml/features.py` | ML feature engineering |
| `strategy/ml/train_rnn.py` | PyTorch GRU/LSTM training |
| `strategy/ml/retrain_live.py` | Live retraining and model promotion |
| `strategy/ml/model_report.py` | Active model and retrain-history report |
| `strategy/ml/models/rnn.pt` | Active PyTorch checkpoint |
| `strategy/ml/models/rnn.json` | Active model metadata |
| `scripts/trading_local.sh` | Local scanner start/stop/status/log helper |
| `tests/ml_features.test.js` | Guardrail test for confluence feature generation |

## Local Runtime State

These files are local runtime state and should not contain Git-tracked secrets:

- `strategy/config.json`
- `strategy/trades.json`
- `strategy/weights.json`
- `strategy/scanner.run.lock`
- `strategy/ml/data/raw/`
- `strategy/ml/data/processed/`
- `strategy/ml/data/live/`
- `strategy/ml/reports/`
- `alerts_log/`
- root `rules.json`

The model pair `strategy/ml/models/rnn.pt` and `strategy/ml/models/rnn.json` is tracked only when deliberately committed.

## Quick Start For This Local Fork

From the repo root:

```bash
cd /Users/nikhiljha/tradingview-mcp
npm install
```

Create your private config from the example if it does not already exist:

```bash
cp strategy/config.example.json strategy/config.json
```

Then add your Telegram token and chat settings to `strategy/config.json`. Do not commit this file.

Start TradingView with the debug port and load the scanner:

```bash
./scripts/trading_local.sh start
```

Useful local commands:

```bash
./scripts/trading_local.sh restart       # reload scanner code and run one scan now
./scripts/trading_local.sh stop          # stop scheduled scanner
./scripts/trading_local.sh status        # show scanner, keep-awake, and TradingView status
./scripts/trading_local.sh logs          # follow scanner logs
./scripts/trading_local.sh keep-awake-on # prevent idle sleep while logged in
npm run model:report                     # summarize active RNN and retrain history
npm run test:unit                        # run local unit tests
```

## Operating Model

The scanner is a macOS LaunchAgent.

- You do not need Codex or Claude connected after startup.
- The process PID can change every scan cycle.
- The terminal can be closed if the LaunchAgent is loaded.
- If the laptop sleeps fully, scheduled scans will pause until the machine wakes.
- `keep-awake-on` uses `caffeinate -i` to reduce idle sleep while logged in, but closing the lid or power settings may still suspend work.

Check status:

```bash
./scripts/trading_local.sh status
```

Watch logs:

```bash
./scripts/trading_local.sh logs
```

## Testing And Validation

Run the main local checks:

```bash
npm run test:unit
node --check strategy/scanner.js
node --check strategy/analyzer.js
python3 -m py_compile strategy/ml/features.py strategy/ml/train_rnn.py strategy/ml/retrain_live.py
npm run model:report
```

Current unit coverage includes:

- CLI routing
- Pine analysis
- ML bridge defaults
- market-store persistence
- signal continuity
- risk/reward validation
- confluence feature generation

Recommended validation after meaningful strategy or ML changes:

1. Run unit tests.
2. Run `npm run model:report`.
3. Run one scanner cycle.
4. Inspect `strategy/scanner.log`.
5. Inspect `strategy/ml/data/live/scan_signals.jsonl`.
6. Confirm Telegram alert formatting if a signal fires.
7. Compare new model metrics with previous retrain history.

## MCP Bridge Setup

TradingView Desktop must be running with Chrome DevTools Protocol enabled on port `9222`.

Mac:

```bash
./scripts/launch_tv_debug_mac.sh
```

Windows:

```bash
scripts\launch_tv_debug.bat
```

Linux:

```bash
./scripts/launch_tv_debug_linux.sh
```

Manual launch:

```bash
/path/to/TradingView --remote-debugging-port=9222
```

Claude Code MCP config example:

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "node",
      "args": ["/path/to/tradingview-mcp/src/server.js"]
    }
  }
}
```

## CLI Examples

Every MCP tool is also available through the `tv` CLI.

```bash
node src/cli/index.js status
node src/cli/index.js quote
node src/cli/index.js symbol EURUSD
node src/cli/index.js timeframe 15
node src/cli/index.js ohlcv --summary
node src/cli/index.js pine analyze --file script.pine
```

Optional global link:

```bash
npm link
tv status
```

## High-Level Architecture

```text
Claude Code / CLI
      |
      v
MCP server and tv CLI
      |
      v
Chrome DevTools Protocol on localhost:9222
      |
      v
TradingView Desktop

Local scanner
      |
      v
TradingView OHLCV capture -> analyzer -> ML bridge -> Telegram
      |
      v
local data store -> retraining -> candidate promotion -> active RNN
```

Key modules:

- `src/server.js` exposes MCP tools.
- `src/cli/index.js` exposes terminal commands.
- `strategy/cdp_client.js` controls TradingView chart state.
- `strategy/scanner.js` runs the 15-minute cycle.
- `strategy/ml_bridge.js` calls Python inference.
- `strategy/ml/retrain_live.py` builds labels and trains candidates.

## Data And Compliance Notes

The MCP bridge communicates with your locally running TradingView Desktop app through localhost CDP. It does not connect directly to TradingView servers.

The scanner does store local OHLCV captures and signal snapshots on your machine when enabled. That data remains local unless you copy, publish, or commit it.

Do not redistribute, sell, or externally publish TradingView or exchange market data captured through this tool. You are responsible for complying with TradingView, exchange, data-provider, and broker terms.

## Known Limitations

- TradingView Desktop updates can break CDP selectors or internal APIs.
- The scanner depends on the chart being available and responsive.
- The ML model is only as good as its labels and data coverage.
- A model can improve utility metrics without guaranteeing live profitability.
- GDELT and RSS feeds can timeout or rate-limit; the system falls back to available news layers.
- This is not low-latency infrastructure and not an order-execution engine.
- Laptop sleep interrupts scheduled work.

## Roadmap

Practical next improvements:

- Add walk-forward backtesting for promoted checkpoints.
- Add model calibration plots and threshold tuning.
- Add feature-importance/permutation reports.
- Add Telegram summary reports after each retrain.
- Add separate paper-trading outcome tracking.
- Add CI checks for Python feature extraction and model-report output.

## Attributions

This project is not affiliated with, endorsed by, or associated with:

- TradingView Inc. TradingView is a trademark of TradingView Inc.
- Anthropic. Claude and Claude Code are trademarks of Anthropic, PBC.

The source originated as a TradingView MCP bridge and has been extended locally with scanner, neural-gate, retraining, and Telegram workflow layers.

## Disclaimer

This project is provided for personal, educational, and research purposes only.

How this tool works: it uses Chrome DevTools Protocol, a standard debugging interface built into Chromium-based applications. It does not reverse engineer a proprietary TradingView network protocol, connect directly to TradingView servers, or bypass access controls. The debug port must be explicitly enabled by the user through `--remote-debugging-port=9222`.

By using this software, you acknowledge and agree that:

1. You are solely responsible for ensuring your use of this tool complies with TradingView's Terms of Use and all applicable laws.
2. TradingView's Terms of Use may restrict automated data collection, scraping, and non-display usage of their platform and data.
3. You assume all risk associated with using this tool.
4. This tool must not be used to redistribute, resell, or commercially exploit TradingView market data.
5. This tool must not be used to circumvent TradingView access controls or subscription restrictions.
6. This tool does not place trades and should not be treated as financial advice.
7. Market data remains subject to exchange and data-provider licensing terms.
8. Internal TradingView application interfaces may change or break at any time.

Use at your own risk. If you are unsure whether your intended use complies with TradingView's terms, do not use this tool.

## License

MIT - see [LICENSE](LICENSE) for details.

The MIT license applies to this project's source code only. It does not grant rights to TradingView software, data, trademarks, or intellectual property.
