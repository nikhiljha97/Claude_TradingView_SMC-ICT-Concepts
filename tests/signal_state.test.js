import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { shouldAlert, setupFingerprint, htfAlignment } from '../strategy/signal_state.js';

function signal(overrides = {}) {
  return {
    symbol: 'BTCUSDT',
    direction: 'SELL',
    entry: 100,
    atr: 10,
    score: 7,
    details: {
      weeklyTrend: 'bearish',
      dailyTrend: 'bearish',
      fourHTrend: 'bearish',
      pivotTrend: 'bearish',
      dailyBias: { bias: 'bearish' },
      weeklyTemplate: { bias: 'bearish' },
      sweep15M: { direction: 'bearish', sweptLevel: 102 },
      mss15M: { direction: 'bearish', mssLevel: 99 },
      bos1H: { direction: 'bearish' },
      pd4H: { zone: 'premium' },
      killzone: { name: 'NY_AM' },
    },
    ...overrides,
  };
}

describe('signal continuity', () => {
  it('fingerprints the higher timeframe and setup context', () => {
    const a = setupFingerprint(signal());
    const b = setupFingerprint(signal({ details: { ...signal().details, fourHTrend: 'bullish' } }));
    assert.notEqual(a, b);
  });

  it('suppresses unchanged skipped setup before resend window', () => {
    const current = signal();
    current.setupFingerprint = setupFingerprint(current);
    const previous = {
      tradeId: 'abc',
      symbol: 'BTCUSDT',
      direction: 'SELL',
      timestamp: new Date().toISOString(),
      entry: 100,
      score: 7,
      confirmed: false,
      setupFingerprint: current.setupFingerprint,
    };
    const result = shouldAlert(current, [previous], {
      alertContinuity: { resendSkippedAfterMinutes: 45, minAtrRefresh: 0.35 },
    });
    assert.equal(result.allow, false);
    assert.equal(result.reason, 'same_setup_not_refreshed');
  });

  it('requires stronger HTF conviction before opposite-direction flip', () => {
    const current = signal({
      direction: 'BUY',
      score: 7.2,
      details: {
        ...signal().details,
        weeklyTrend: 'bearish',
        dailyTrend: 'bearish',
        fourHTrend: 'bullish',
        dailyBias: { bias: 'bearish' },
      },
    });
    const previous = {
      tradeId: 'old',
      symbol: 'BTCUSDT',
      direction: 'SELL',
      timestamp: new Date().toISOString(),
      entry: 100,
      score: 7,
      confirmed: null,
    };
    const result = shouldAlert(current, [previous], {
      alertContinuity: { oppositeDirectionCooldownHours: 2, flipScoreMargin: 1.25, flipMinHtfAligned: 3 },
    });
    assert.equal(result.allow, false);
    assert.equal(result.reason, 'opposite_direction_needs_more_conviction');
  });

  it('counts HTF alignment for current direction', () => {
    const result = htfAlignment(signal());
    assert.equal(result.aligned, true);
    assert.ok(result.count >= 3);
  });
});
