#!/usr/bin/env node
/**
 * Trading Scanner — runs every 15 minutes via launchd/cron.
 *
 * For each pair (skipping XAUUSD on weekends):
 *   1. Switch TradingView chart to that symbol
 *   2. Pull OHLCV at Weekly, Daily, 4H, 1H, 15M
 *   3. Run SMC + Pivot Boss multi-TF analysis
 *   4. Score signal; if score ≥ threshold and RR ≥ 2.5, fire Telegram alert
 *   5. Log trade to trades.json for reinforcement learning
 *
 * Also processes incoming Telegram replies (TP HIT / SL HIT) at the start of each scan.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { setSymbol, setTimeframe, getOhlcv, closeClient } from './cdp_client.js';
import { computeSignal } from './analyzer.js';
import { scoreWithMl } from './ml_bridge.js';
import { recordFetchedBars, recordScanSnapshot } from './market_store.js';
import { maybeStartRetraining } from './ml_retrainer.js';
import { refreshNewsContext } from './context_refresh.js';
import { shouldAlert } from './signal_state.js';
import { sendMessage, getUpdates, formatAlert, parseFeedback } from './telegram.js';
import {
  loadTrades, loadWeights, saveWeights, effectiveWeights,
  logAlert, applyFeedback, applyConfirmation, parseConfirmation, genTradeId, formatStats
} from './learning.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.join(__dirname, 'config.json');
const RUN_LOCK_PATH = path.join(__dirname, 'scanner.run.lock');
const SCAN_SEPARATOR = '============================================================';

function loadConfig() { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); }
function saveConfig(c) { fs.writeFileSync(CONFIG_PATH, JSON.stringify(c, null, 2)); }

function acquireRunLock() {
  try {
    fs.writeFileSync(RUN_LOCK_PATH, JSON.stringify({
      pid: process.pid,
      startedAt: new Date().toISOString(),
    }), { flag: 'wx' });
    return true;
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    try {
      const lock = JSON.parse(fs.readFileSync(RUN_LOCK_PATH, 'utf8'));
      const ageMs = Date.now() - new Date(lock.startedAt).getTime();
      if (Number.isFinite(ageMs) && ageMs > 14 * 60 * 1000) {
        fs.unlinkSync(RUN_LOCK_PATH);
        return acquireRunLock();
      }
      console.log(`\n\n\n${SCAN_SEPARATOR}\n${SCAN_SEPARATOR}`);
      console.log(`[scanner] Previous scan still running pid=${lock.pid || 'unknown'} started=${lock.startedAt || 'unknown'} — skipping this trigger.`);
      return false;
    } catch {
      fs.unlinkSync(RUN_LOCK_PATH);
      return acquireRunLock();
    }
  }
}

function releaseRunLock() {
  try { fs.unlinkSync(RUN_LOCK_PATH); } catch {}
}

function isWeekendSession(now = new Date()) {
  // Weekend session: Fri 17:00 EST → Sun 18:00 EST (crypto-only watchlist)
  const est = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const day  = est.getDay();   // 0=Sun, 5=Fri, 6=Sat
  const hour = est.getHours();
  if (day === 6) return true;                  // all Saturday
  if (day === 5 && hour >= 17) return true;    // Fri after 5pm EST
  if (day === 0 && hour < 18) return true;     // Sun before 6pm EST
  return false;
}

async function processFeedback(config, weights) {
  const updates = await getUpdates(config.telegram.token, (config.telegram.last_update_id || 0) + 1);
  if (!updates.length) return [];

  const replies = [];
  let maxId = config.telegram.last_update_id || 0;

  for (const u of updates) {
    if (u.update_id > maxId) maxId = u.update_id;
    const msg = u.message;
    if (!msg) continue;

    // Auto-detect chat_id on first contact
    if (!config.telegram.chat_id) {
      config.telegram.chat_id = msg.chat.id;
      console.log(`[scanner] Captured Telegram chat_id: ${msg.chat.id}`);
    }

    // /stats command
    if (msg.text === '/stats') {
      await sendMessage(config.telegram.token, msg.chat.id, formatStats(weights));
      continue;
    }
    // /trades command — show last 10
    if (msg.text === '/trades') {
      const trades = loadTrades().slice(-10).reverse();
      const txt = trades.length === 0 ? '_No trades logged yet._' :
        '*Last 10 alerts:*\n\n' + trades.map(t =>
          `\`${t.tradeId}\` ${t.symbol} ${t.direction} @ ${t.entry?.toFixed(2)} → ${t.outcome || 'OPEN'}`
        ).join('\n');
      await sendMessage(config.telegram.token, msg.chat.id, txt);
      continue;
    }

    // YES/NO trade confirmation
    const conf = parseConfirmation(msg.text);
    if (conf) {
      const result = applyConfirmation(conf.tradeId, conf.took);
      if (result.ok) {
        const t = result.trade;
        const pair = `${t.symbol} ${t.direction}`;
        const reply = conf.took
          ? `✅ *${pair}* \`${conf.tradeId}\` marked as *taken* — cooldown active, won't re-alert this setup.\nReply \`TP HIT ${conf.tradeId}\` or \`SL HIT ${conf.tradeId}\` when done.`
          : `⏭ *${pair}* \`${conf.tradeId}\` marked as *skipped* — will re-alert if setup holds next cycle.`;
        await sendMessage(config.telegram.token, msg.chat.id, reply);
      } else {
        await sendMessage(config.telegram.token, msg.chat.id, `⚠️ ${result.error}`);
      }
      continue;
    }

    // TP/SL feedback
    const fb = parseFeedback(msg.text);
    if (fb) {
      const result = applyFeedback(fb.tradeId, fb.outcome, weights);
      if (result.ok) {
        await sendMessage(config.telegram.token, msg.chat.id,
          `✅ Logged ${fb.outcome} HIT for \`${fb.tradeId}\` (${result.trade.symbol})\nWin rate: ${result.stats.winRate} (${result.stats.totalTrades} trades)`);
        replies.push(result);
      } else {
        await sendMessage(config.telegram.token, msg.chat.id, `⚠️ ${result.error}`);
      }
    }
  }

  config.telegram.last_update_id = maxId;
  saveConfig(config);
  return replies;
}

async function fetchTimeframeBars(symbol, tf, count) {
  await setTimeframe(tf);
  return await getOhlcv(count);
}

function mlSummary(result) {
  if (!result?.enabled) return 'disabled';
  if (result.success) return `p=${Number(result.probability).toFixed(3)} min=${result.minProbability}`;
  return `unavailable (${result.reason || 'unknown'})`;
}

async function scoreDirectionalProbe(config, payload, direction) {
  return scoreWithMl(config, {
    ...payload,
    signal: {
      ...payload.signal,
      direction,
    },
  });
}

async function scanSymbol(pair, config, weights) {
  console.log(`[scanner] Scanning ${pair.symbol}...`);
  await setSymbol(pair.symbol);

  // Fetch OHLCV at all timeframes (chart switches between calls)
  const barsWeekly = await fetchTimeframeBars(pair.symbol, 'W', 30);
  const barsDaily  = await fetchTimeframeBars(pair.symbol, 'D', 60);
  const bars4H     = await fetchTimeframeBars(pair.symbol, '240', 100);
  const bars1H     = await fetchTimeframeBars(pair.symbol, '60', 100);
  const bars15M    = await fetchTimeframeBars(pair.symbol, '15', config.strategy.entryBarsToFetch || 180);
  const stored = recordFetchedBars(config, pair.symbol, {
    W: barsWeekly,
    D: barsDaily,
    240: bars4H,
    60: bars1H,
    15: bars15M,
  }) || [];
  const written = stored.reduce((sum, item) => sum + (item.written || 0), 0);
  if (written > 0) console.log(`[scanner] Stored ${written} new OHLCV bars for ${pair.symbol}`);

  const eff = effectiveWeights(weights);
  const signal = computeSignal({
    symbol: pair.symbol,
    bars4H, bars1H, bars15M, barsDaily, barsWeekly,
    weights: eff,
  });

  const mlPayload = {
    signal,
    bars15M,
    bars1H,
    bars4H,
    barsDaily,
  };

  let mlResult = { enabled: config?.ml?.enabled !== false, passed: false, success: false, reason: 'no_rule_direction' };
  if (signal.direction !== 'NONE') {
    mlResult = await scoreWithMl(config, mlPayload);
    signal.details.ml = mlResult;
    if (mlResult.enabled) {
      console.log(`[scanner] ${pair.symbol}: ML ${mlSummary(mlResult)} | passed=${mlResult.passed}`);
    }
    if (mlResult.enabled && !mlResult.passed) {
      signal.direction = 'NONE';
    }
  } else {
    signal.details.ml = mlResult;
    if (config?.ml?.probeWhenNoSignal !== false) {
      const [buyProbe, sellProbe] = await Promise.all([
        scoreDirectionalProbe(config, mlPayload, 'BUY'),
        scoreDirectionalProbe(config, mlPayload, 'SELL'),
      ]);
      signal.details.mlCandidates = { BUY: buyProbe, SELL: sellProbe };
      if (buyProbe.enabled || sellProbe.enabled) {
        console.log(`[scanner] ${pair.symbol}: ML side probes BUY ${mlSummary(buyProbe)} | SELL ${mlSummary(sellProbe)} (research only)`);
      }
    }
  }
  recordScanSnapshot(config, pair.symbol, signal);

  console.log(`[scanner] ${pair.symbol}: ${signal.direction} | score ${signal.score}/${signal.maxScore.toFixed(1)} | RR ${signal.rr || 'n/a'}`);

  if (signal.direction === 'NONE') {
    if (signal.details.mlCandidates) {
      console.log(`[scanner] No alert for ${pair.symbol}: rule engine returned NONE; ML side probes are research-only until structure confirms a direction.`);
    }
    return null;
  }

  const continuity = signal.direction !== 'NONE'
    ? shouldAlert(signal, loadTrades(), config)
    : { allow: false, reason: 'no_signal' };
  signal.details.continuity = continuity;
  if (signal.direction !== 'NONE' && !continuity.allow) {
    console.log(`[scanner] Suppressed ${pair.symbol} ${signal.direction}: ${continuity.reason}`);
  }

  if (signal.score >= config.strategy.alertScoreThreshold &&
      signal.rr >= config.strategy.minRR &&
      continuity.allow) {
    signal.tradeId = genTradeId();
    logAlert(signal);
    if (config.telegram.chat_id) {
      await sendMessage(config.telegram.token, config.telegram.chat_id, formatAlert(signal));
      console.log(`[scanner] 🚨 Alert sent for ${pair.symbol} ${signal.direction}`);
    } else {
      console.log(`[scanner] Alert ready but no chat_id yet — message the bot with /start first.`);
    }
    return signal;
  }

  const reasons = [];
  if (signal.score < config.strategy.alertScoreThreshold) {
    reasons.push(`score ${signal.score} < ${config.strategy.alertScoreThreshold}`);
  }
  if (signal.rr < config.strategy.minRR) {
    reasons.push(`RR ${signal.rr || 'n/a'} < ${config.strategy.minRR}`);
  }
  if (!continuity.allow) {
    reasons.push(`continuity ${continuity.reason || 'blocked'}`);
  }
  console.log(`[scanner] No alert for ${pair.symbol} ${signal.direction}: ${reasons.join(', ')}`);
  return null;
}

async function main() {
  if (!acquireRunLock()) return;
  const config = loadConfig();
  const weights = loadWeights(config.weights);
  const startTime = Date.now();
  const estTime = new Date().toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  console.log(`\n\n\n${SCAN_SEPARATOR}\n${SCAN_SEPARATOR}`);
  console.log(`[scanner] === NEW SCAN started at ${estTime} EST ===`);

  // 1. Process any pending Telegram feedback first
  try {
    await processFeedback(config, weights);
  } catch (e) {
    console.error(`[scanner] Telegram feedback error: ${e.message}`);
  }

  const newsRefresh = await refreshNewsContext(config);
  if (newsRefresh.refreshed) {
    console.log(`[scanner] News context refreshed in ${(newsRefresh.elapsedMs / 1000).toFixed(1)}s`);
  } else if (newsRefresh.reason && !['fresh', 'disabled'].includes(newsRefresh.reason)) {
    console.log(`[scanner] News context refresh skipped: ${newsRefresh.reason}`);
  }

  // 2. Pick watchlist based on session (weekday forex/indices vs weekend crypto)
  const weekend = isWeekendSession();
  const pairs = weekend ? config.weekendPairs : config.weekdayPairs;
  console.log(`[scanner] Session: ${weekend ? 'WEEKEND (crypto)' : 'WEEKDAY (forex/indices/crypto)'} — ${pairs.length} pairs`);

  const alerts = [];
  for (const pair of pairs) {
    try {
      const sig = await scanSymbol(pair, config, weights);
      if (sig) alerts.push(sig);
    } catch (e) {
      console.error(`[scanner] Error scanning ${pair.symbol}: ${e.message}`);
    }
  }

  await closeClient();
  const retrain = maybeStartRetraining(config);
  if (retrain.started) {
    console.log(`[scanner] Started background RNN retraining pid=${retrain.pid} log=${retrain.logPath}`);
  } else if (retrain.reason && retrain.reason !== 'disabled') {
    console.log(`[scanner] RNN retraining skipped: ${retrain.reason}`);
  }
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[scanner] === Cycle complete in ${elapsed}s, ${alerts.length} alert(s) fired ===`);
  releaseRunLock();
}

main().catch(err => {
  console.error('[scanner] Fatal:', err);
  releaseRunLock();
  process.exit(1);
});
