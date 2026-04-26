/**
 * Reinforcement learning log.
 *
 * Tracks every fired alert with all signal components, then updates per-component
 * accuracy weights (EWMA) when the user replies TP HIT / SL HIT.
 *
 * Inspired by:
 *   - Max Dama: EMA-smoothed online stat updates (memoryless, fast)
 *   - Inside the Black Box: continual model recalibration
 *
 * Component weights start at config.json defaults. Each component's "edge" is
 * tracked as an EWMA of (TP=1, SL=0). After enough samples, the weight is scaled
 * by 2*edge so high-accuracy components count more, low-accuracy ones decay.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const TRADES_LOG = path.resolve(import.meta.dirname || './', 'trades.json');
const WEIGHTS_FILE = path.resolve(import.meta.dirname || './', 'weights.json');

const EWMA_ALPHA = 0.15; // smoothing factor — newer trades weighted more
const MIN_SAMPLES_TO_ADJUST = 5; // don't adjust weights until we have data

export function loadTrades() {
  if (!fs.existsSync(TRADES_LOG)) return [];
  try { return JSON.parse(fs.readFileSync(TRADES_LOG, 'utf8')); } catch { return []; }
}

export function saveTrades(trades) {
  fs.writeFileSync(TRADES_LOG, JSON.stringify(trades, null, 2));
}

export function loadWeights(defaults) {
  const init = () => ({
    base: { ...defaults },
    ewma: Object.fromEntries(Object.keys(defaults).map(k => [k, 0.5])),
    counts: Object.fromEntries(Object.keys(defaults).map(k => [k, 0])),
    totalTrades: 0, wins: 0, losses: 0,
  });

  if (!fs.existsSync(WEIGHTS_FILE)) {
    const w = init();
    fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(w, null, 2));
    return w;
  }

  try {
    const stored = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
    // Merge any new weight keys added to config.json that aren't in weights.json yet
    let changed = false;
    for (const k of Object.keys(defaults)) {
      if (!(k in stored.base)) {
        stored.base[k]   = defaults[k];
        stored.ewma[k]   = 0.5;
        stored.counts[k] = 0;
        changed = true;
      }
    }
    if (changed) fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(stored, null, 2));
    return stored;
  } catch {
    const w = init();
    fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(w, null, 2));
    return w;
  }
}

export function saveWeights(weights) {
  fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(weights, null, 2));
}

/**
 * Get the current effective weights (base * 2 * ewma) for the analyzer.
 * Falls back to base weights if not enough samples for a component yet.
 */
export function effectiveWeights(weights) {
  const eff = {};
  for (const [k, baseVal] of Object.entries(weights.base)) {
    const ewma = weights.ewma[k] ?? 0.5;
    const count = weights.counts[k] ?? 0;
    if (count < MIN_SAMPLES_TO_ADJUST) {
      eff[k] = baseVal;
    } else {
      // Scale by 2*edge: edge=0.5 → weight unchanged; edge=1 → 2x; edge=0 → 0
      eff[k] = baseVal * Math.max(0.1, Math.min(2.0, 2 * ewma));
    }
  }
  return eff;
}

/**
 * Generate short trade ID (8 hex chars).
 */
export function genTradeId() {
  return crypto.randomBytes(4).toString('hex');
}

/**
 * Log a fired alert. Returns the trade record.
 */
export function logAlert(signal) {
  const trades = loadTrades();
  const trade = {
    tradeId: signal.tradeId,
    timestamp: new Date().toISOString(),
    symbol: signal.symbol,
    direction: signal.direction,
    entry: signal.entry,
    sl: signal.sl,
    tp1: signal.tp1,
    tp2: signal.tp2,
    rr: signal.rr,
    score: signal.score,
    maxScore: signal.maxScore,
    breakdown: signal.breakdown,
    setupFingerprint: signal.setupFingerprint,
    htfAlignment: signal.htfAlignment,
    activeComponents: Object.keys(signal.breakdown).filter(k => signal.breakdown[k] > 0),
    confirmed: null, // null=awaiting, true=took trade, false=skipped
    outcome: null,   // 'TP' | 'SL' | null
    outcomeAt: null,
  };
  trades.push(trade);
  saveTrades(trades);
  return trade;
}

/**
 * Apply user feedback: update outcome and EWMA weights.
 */
export function applyFeedback(tradeId, outcome, weights) {
  const trades = loadTrades();
  const trade = trades.find(t => t.tradeId === tradeId);
  if (!trade) return { ok: false, error: `Trade ${tradeId} not found.` };
  if (trade.outcome) return { ok: false, error: `Trade ${tradeId} already logged as ${trade.outcome}.` };

  trade.outcome = outcome;
  trade.outcomeAt = new Date().toISOString();
  saveTrades(trades);

  trade.confirmed = true; // TP/SL reply implies they took the trade
  const win = outcome === 'TP' ? 1 : 0;
  weights.totalTrades = (weights.totalTrades || 0) + 1;
  weights.wins   = (weights.wins   || 0) + win;
  weights.losses = (weights.losses || 0) + (1 - win);

  for (const comp of trade.activeComponents) {
    const prev = weights.ewma[comp] ?? 0.5;
    weights.ewma[comp] = prev + EWMA_ALPHA * (win - prev);
    weights.counts[comp] = (weights.counts[comp] ?? 0) + 1;
  }
  saveWeights(weights);

  return {
    ok: true,
    trade,
    stats: {
      totalTrades: weights.totalTrades,
      winRate: weights.totalTrades > 0 ? (weights.wins / weights.totalTrades * 100).toFixed(1) + '%' : 'n/a',
    }
  };
}

/**
 * Parse a trade confirmation: "YES abc123" or "NO abc123".
 * Returns { took: true|false, tradeId } or null.
 */
export function parseConfirmation(text) {
  if (!text) return null;
  const m = text.trim().match(/^(YES|NO)\s+([a-z0-9]+)/i);
  if (!m) return null;
  return { took: m[1].toUpperCase() === 'YES', tradeId: m[2].toLowerCase() };
}

/**
 * Record whether the user took the trade. Controls cooldown behaviour:
 * confirmed=true → cooldown applies; confirmed=false → cooldown skipped.
 */
export function applyConfirmation(tradeId, took) {
  const trades = loadTrades();
  const trade = trades.find(t => t.tradeId === tradeId);
  if (!trade) return { ok: false, error: `Trade ${tradeId} not found.` };
  if (trade.confirmed !== null && trade.confirmed !== undefined)
    return { ok: false, error: `Trade ${tradeId} already ${trade.confirmed ? 'confirmed' : 'skipped'}.` };
  trade.confirmed = took;
  saveTrades(trades);
  return { ok: true, trade };
}

/**
 * Format current performance stats for telegram reply.
 */
export function formatStats(weights) {
  const total = weights.totalTrades || 0;
  const wr = total > 0 ? (weights.wins / total * 100).toFixed(1) : '0';
  const componentLines = Object.entries(weights.ewma).map(([k, v]) => {
    const c = weights.counts[k] || 0;
    const eff = effectiveWeights(weights)[k];
    return `  • ${k}: ${(v * 100).toFixed(0)}% (${c}n, w=${eff.toFixed(2)})`;
  }).join('\n');
  return [
    `📊 *Performance Stats*`,
    `Trades: ${total} | Wins: ${weights.wins} | Losses: ${weights.losses}`,
    `Win Rate: ${wr}%`,
    ``,
    `*Component edge / weight:*`,
    componentLines,
  ].join('\n');
}
