import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { mlConfig, scoreWithMl } from '../strategy/ml_bridge.js';

describe('ml bridge', () => {
  it('can be disabled explicitly', async () => {
    const result = await scoreWithMl({ ml: { enabled: false } }, {});
    assert.equal(result.enabled, false);
    assert.equal(result.passed, false);
  });

  it('is enabled by default as the core gate', () => {
    const config = mlConfig({});
    assert.equal(config.enabled, true);
    assert.ok(config.modelPath.endsWith('strategy/ml/models/rnn.pt'));
  });

  it('normalizes config defaults', () => {
    const config = mlConfig({ ml: { enabled: true, modelPath: 'strategy/ml/models/rnn.pt' } });
    assert.equal(config.enabled, true);
    assert.equal(config.python, 'python3');
    assert.equal(config.minProbability, 0.50);
    assert.equal(config.failOpen, false);
    assert.ok(config.modelPath.endsWith('strategy/ml/models/rnn.pt'));
  });
});
