import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

function actualRR(direction, entry, sl, target) {
  const risk = Math.abs(Number(entry) - Number(sl));
  const reward = direction === 'BUY'
    ? Number(target) - Number(entry)
    : Number(entry) - Number(target);
  return risk > 0 && reward > 0 ? reward / risk : 0;
}

describe('risk/reward validation', () => {
  it('rejects apparent 1:2.5 labels when the adjusted target is actually closer', () => {
    const rr = actualRR('BUY', 215.652, 215.016, 215.932);
    assert.equal(Number(rr.toFixed(2)), 0.44);
    assert.ok(rr < 2.5);
  });

  it('accepts only actual target distance at 2.5R or better', () => {
    const rr = actualRR('SELL', 100, 104, 90);
    assert.equal(rr, 2.5);
    assert.ok(rr >= 2.5);
  });
});
