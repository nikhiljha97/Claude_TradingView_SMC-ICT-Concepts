import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { formatAlert, parseFeedback } from '../strategy/telegram.js';
import { computeOutcomeMetrics } from '../strategy/learning.js';

describe('trade feedback outcome accounting', () => {
  it('parses TP/SL feedback with optional exact exit price', () => {
    assert.deepEqual(parseFeedback('TP HIT a1b2c3d4'), {
      outcome: 'TP',
      tradeId: 'a1b2c3d4',
    });
    assert.deepEqual(parseFeedback('SL HIT a1b2c3d4 @ 215.016'), {
      outcome: 'SL',
      tradeId: 'a1b2c3d4',
      exitPrice: 215.016,
    });
  });

  it('computes duration, realized R, and pips for a closed trade', () => {
    const trade = {
      tradeId: 'abc12345',
      timestamp: '2026-01-01T00:00:00.000Z',
      symbol: 'EURUSD',
      direction: 'BUY',
      entry: 1.1,
      sl: 1.095,
      tp1: 1.1125,
    };
    const metrics = computeOutcomeMetrics(trade, 'TP', null, new Date('2026-01-01T01:30:00.000Z'));
    assert.equal(metrics.outcomePriceSource, 'planned_level');
    assert.equal(metrics.durationMinutes, 90);
    assert.equal(metrics.durationBars15m, 6);
    assert.ok(Math.abs(metrics.realizedR - 2.5) < 1e-9);
    assert.ok(Math.abs(metrics.pipsCaptured - 125) < 1e-6);
  });

  it('makes Telegram alert headers visually loud', () => {
    const text = formatAlert({
      symbol: 'EURUSD',
      direction: 'BUY',
      entry: 1.1,
      sl: 1.095,
      tp1: 1.1125,
      tp2: 1.12,
      rr: 2.5,
      score: 8,
      maxScore: 13.2,
      tradeId: 'abc12345',
      breakdown: { htfAlignment: 3 },
      details: {},
    });
    assert.match(text.split('\n')[0], /^🚨 ❗❗❗ 🟢 BUY \*EURUSD\*/);
  });
});
