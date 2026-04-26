import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { appendBars, recordFetchedBars, recordScanSnapshot } from '../strategy/market_store.js';
import { maybeStartRetraining } from '../strategy/ml_retrainer.js';

function tempConfig() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-market-store-'));
  return { dataCapture: { enabled: true, path: dir }, dir };
}

describe('market store', () => {
  it('appends deduped bars by timestamp', () => {
    const config = tempConfig();
    const bars = [
      { timestamp: 1000, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 },
      { timestamp: 1000, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 },
      { time: 2, o: 2, h: 3, l: 1, c: 2.5, v: 12 },
    ];

    const first = appendBars(config, 'EUR/USD', '15', bars);
    const second = appendBars(config, 'EUR/USD', '15', bars);

    assert.equal(first.written, 2);
    assert.equal(second.written, 0);
    const file = path.join(config.dir, 'EUR_USD_15.csv');
    const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
    assert.equal(lines.length, 3);
  });

  it('records scan snapshots as jsonl', () => {
    const config = tempConfig();
    recordScanSnapshot(config, 'XAUUSD', {
      direction: 'BUY',
      score: 7,
      rr: 3,
      breakdown: { htfAlignment: 2.5 },
      details: { ml: { probability: 0.61 } },
    });
    const file = path.join(config.dir, 'scan_signals.jsonl');
    const row = JSON.parse(fs.readFileSync(file, 'utf8').trim());
    assert.equal(row.symbol, 'XAUUSD');
    assert.equal(row.ml.probability, 0.61);
  });

  it('does not start retraining when disabled', () => {
    const result = maybeStartRetraining({ ml: { retrain: { enabled: false } } });
    assert.equal(result.started, false);
    assert.equal(result.reason, 'disabled');
  });

  it('skips retraining when interval has not elapsed', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-retrain-state-'));
    const statePath = path.join(dir, 'state.json');
    fs.writeFileSync(statePath, JSON.stringify({ lastStartedAt: new Date().toISOString() }));
    const result = maybeStartRetraining({
      dataCapture: { path: dir },
      ml: {
        python: 'python3',
        modelPath: path.join(dir, 'rnn.pt'),
        retrain: {
          enabled: true,
          minHoursBetweenRuns: 24,
          statePath,
          lockPath: path.join(dir, 'lock'),
          logPath: path.join(dir, 'log'),
        },
      },
    });
    assert.equal(result.started, false);
    assert.equal(result.reason, 'interval_not_elapsed');
  });
});
