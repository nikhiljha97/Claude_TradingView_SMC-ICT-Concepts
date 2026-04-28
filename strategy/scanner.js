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
import { setSymbol, setTimeframe, getOhlcv, getChartSymbol, closeClient } from './cdp_client.js';
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
const ANSI_RED = '\x1b[31m';
const ANSI_BOLD = '\x1b[1m';
const ANSI_RESET = '\x1b[0m';

function loadConfig() { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); }
function saveConfig(c) { fs.writeFileSync(CONFIG_PATH, JSON.stringify(c, null, 2)); }

function terminalAlert(message) {
  console.log(`${ANSI_BOLD}${ANSI_RED}${message}${ANSI_RESET}`);
}

function configuredSymbol(pair) {
  if (pair.tradingViewSymbol) return pair.tradingViewSymbol;
  if (pair.exchange && pair.symbol && !String(pair.symbol).includes(':')) {
    return `${pair.exchange}:${pair.symbol}`;
  }
  return pair.symbol;
}

function instrumentIdentity(configured, resolved) {
  const value = String(resolved || configured || '');
  const [exchange, ...rest] = value.includes(':') ? value.split(':') : [null, value];
  return {
    exchange: exchange || null,
    displaySymbol: rest.join(':') || value,
    tradingViewSymbol: value,
  };
}

function pidIsRunning(pid) {
  const numeric = Number(pid);
  if (!Number.isInteger(numeric) || numeric <= 0) return false;
  try {
    process.kill(numeric, 0);
    return true;
  } catch {
    return false;
  }
}

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
      if (!pidIsRunning(lock.pid)) {
        fs.unlinkSync(RUN_LOCK_PATH);
        return acquireRunLock();
      }
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

function minutesFromClock(value) {
  const [hh, mm = '0'] = String(value || '').split(':');
  const hour = Number(hh);
  const minute = Number(mm);
  if (!Number.isInteger(hour) || !Number.isInteger(minute)) return null;
  if (hour < 0 || hour > 23 || minute < 0 || minute > 59) return null;
  return hour * 60 + minute;
}

function isWithinClockWindow(currentMinutes, start, end) {
  const startMinutes = minutesFromClock(start);
  const endMinutes = minutesFromClock(end);
  if (startMinutes == null || endMinutes == null) return false;
  if (startMinutes === endMinutes) return true;
  if (startMinutes < endMinutes) {
    return currentMinutes >= startMinutes && currentMinutes < endMinutes;
  }
  return currentMinutes >= startMinutes || currentMinutes < endMinutes;
}

function getNoTradeStatus(config, now = new Date()) {
  const windows = config.strategy?.noTradeWindows || [
    {
      name: 'weekday_3pm_7pm_ny',
      timezone: 'America/New_York',
      days: [1, 2, 3, 4, 5],
      start: '15:00',
      end: '19:00',
    },
  ];

  for (const window of windows) {
    if (window.enabled === false) continue;
    const timezone = window.timezone || 'America/New_York';
    const local = new Date(now.toLocaleString('en-US', { timeZone: timezone }));
    const day = local.getDay();
    const allowedDays = Array.isArray(window.days) ? window.days : [1, 2, 3, 4, 5];
    if (!allowedDays.includes(day)) continue;
    const currentMinutes = local.getHours() * 60 + local.getMinutes();
    if (!isWithinClockWindow(currentMinutes, window.start, window.end)) continue;
    return {
      active: true,
      name: window.name || 'no_trade_window',
      timezone,
      start: window.start,
      end: window.end,
    };
  }

  return { active: false };
}

function defaultAlertSessions() {
  return [
    {
      name: 'asia_tokyo',
      timezone: 'America/New_York',
      days: [0, 1, 2, 3, 4],
      start: '20:00',
      end: '00:00',
      allowedSymbolContains: ['JPY', 'AUD', 'NZD', 'BTC', 'ETH', 'XRP', 'SOL', 'ETC'],
      extraScoreMargin: 0.75,
      extraMlProbability: 0.05,
    },
    {
      name: 'london_killzone',
      timezone: 'America/New_York',
      days: [1, 2, 3, 4, 5],
      start: '02:00',
      end: '05:00',
    },
    {
      name: 'new_york_am_killzone',
      timezone: 'America/New_York',
      days: [1, 2, 3, 4, 5],
      start: '08:30',
      end: '11:00',
    },
    {
      name: 'london_close_killzone',
      timezone: 'America/New_York',
      days: [1, 2, 3, 4, 5],
      start: '10:00',
      end: '12:00',
    },
  ];
}

function sessionIsActive(session, now = new Date()) {
  if (session.enabled === false) return false;
  const timezone = session.timezone || 'America/New_York';
  const local = new Date(now.toLocaleString('en-US', { timeZone: timezone }));
  const day = local.getDay();
  const allowedDays = Array.isArray(session.days) ? session.days : [1, 2, 3, 4, 5];
  if (!allowedDays.includes(day)) return false;
  const currentMinutes = local.getHours() * 60 + local.getMinutes();
  return isWithinClockWindow(currentMinutes, session.start, session.end);
}

function symbolAllowedInSession(symbol, session) {
  const exact = Array.isArray(session.allowedSymbols)
    ? session.allowedSymbols.map(s => String(s).toUpperCase())
    : null;
  const contains = Array.isArray(session.allowedSymbolContains)
    ? session.allowedSymbolContains.map(s => String(s).toUpperCase())
    : null;
  const normalized = String(symbol || '').toUpperCase();
  if (exact?.length && !exact.includes(normalized)) return false;
  if (contains?.length && !contains.some(token => normalized.includes(token))) return false;
  return true;
}

function getAlertSessionStatus(config, symbol, signal, mlResult, now = new Date()) {
  const sessions = config.strategy?.alertSessions || defaultAlertSessions();
  const active = sessions.filter(session => sessionIsActive(session, now));
  if (!active.length) {
    return { allow: false, reason: 'outside_ict_sessions' };
  }

  const baseScore = Number(config.strategy?.alertScoreThreshold || 0);
  const baseProbability = Number(mlResult?.minProbability || config.ml?.minProbability || 0);
  const evaluated = active.map(session => {
    const requiredScore = baseScore + Number(session.extraScoreMargin || 0);
    const requiredProbability = baseProbability + Number(session.extraMlProbability || 0);
    const probability = Number(mlResult?.probability);
    const checks = [];
    if (!symbolAllowedInSession(symbol, session)) checks.push('symbol_not_allowed');
    if (Number(signal?.score || 0) < requiredScore) checks.push(`score_below_${requiredScore.toFixed(2)}`);
    if (Number(session.extraMlProbability || 0) > 0 && Number.isFinite(probability) && probability < requiredProbability) {
      checks.push(`ml_below_${requiredProbability.toFixed(2)}`);
    }
    return {
      name: session.name || 'ict_session',
      timezone: session.timezone || 'America/New_York',
      start: session.start,
      end: session.end,
      requiredScore,
      requiredProbability,
      allow: checks.length === 0,
      reason: checks.join(',') || 'allowed',
    };
  });

  return evaluated.find(session => session.allow) || {
    ...evaluated[0],
    allow: false,
  };
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
          ? `✅ *${pair}* \`${conf.tradeId}\` marked as *taken* — cooldown active, won't re-alert this setup.\nReply \`TP1 HIT ${conf.tradeId}\`, \`TP2 HIT ${conf.tradeId}\`, \`TP3 HIT ${conf.tradeId}\`, or \`SL HIT ${conf.tradeId}\`.`
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
      const result = applyFeedback(fb.tradeId, fb.outcome, weights, fb.exitPrice);
      if (result.ok) {
        const metrics = result.metrics || result.trade;
        const r = Number.isFinite(metrics.realizedR) ? metrics.realizedR.toFixed(2) : 'n/a';
        const mins = Number.isFinite(metrics.durationMinutes) ? metrics.durationMinutes.toFixed(0) : 'n/a';
        const pips = Number.isFinite(metrics.pipsCaptured) ? metrics.pipsCaptured.toFixed(1) : 'n/a';
        const prefix = result.partial ? 'Logged partial' : 'Logged final';
        await sendMessage(config.telegram.token, msg.chat.id,
          `✅ ${prefix} ${fb.outcome} HIT for \`${fb.tradeId}\` (${result.trade.symbol})\nR: ${r} | pips/points: ${pips} | duration: ${mins} min\nWin rate: ${result.stats.winRate} (${result.stats.totalTrades} closed trades)`);
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
  const noTrade = getNoTradeStatus(config);
  const requestedSymbol = configuredSymbol(pair);
  await setSymbol(requestedSymbol);
  const resolvedSymbol = await getChartSymbol().catch(() => null);
  const identity = instrumentIdentity(requestedSymbol, resolvedSymbol);
  if (identity.exchange) {
    console.log(`[scanner] Feed: ${identity.exchange}:${identity.displaySymbol}`);
  }

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
  const updated = stored.reduce((sum, item) => sum + (item.updated || 0), 0);
  if (written > 0 || updated > 0) {
    console.log(`[scanner] Stored ${written} new / ${updated} updated OHLCV bars for ${pair.symbol}`);
  }

  const eff = effectiveWeights(weights);
  const signal = computeSignal({
    symbol: pair.symbol,
    bars4H, bars1H, bars15M, barsDaily, barsWeekly,
    weights: eff,
  });
  signal.exchange = identity.exchange;
  signal.displaySymbol = identity.displaySymbol;
  signal.tradingViewSymbol = identity.tradingViewSymbol;

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
  const alertSession = getAlertSessionStatus(config, pair.symbol, signal, mlResult);
  signal.details.continuity = continuity;
  signal.details.noTradeWindow = noTrade;
  signal.details.alertSession = alertSession;
  if (signal.direction !== 'NONE' && !continuity.allow) {
    console.log(`[scanner] Suppressed ${pair.symbol} ${signal.direction}: ${continuity.reason}`);
  }
  if (signal.direction !== 'NONE' && noTrade.active) {
    console.log(`[scanner] Suppressed ${pair.symbol} ${signal.direction}: no_trade_window ${noTrade.start}-${noTrade.end} ${noTrade.timezone}`);
  }
  if (signal.direction !== 'NONE' && !alertSession.allow) {
    console.log(`[scanner] Suppressed ${pair.symbol} ${signal.direction}: ${alertSession.reason}`);
  }

  if (signal.score >= config.strategy.alertScoreThreshold &&
      signal.rr >= config.strategy.minRR &&
      !noTrade.active &&
      alertSession.allow &&
      continuity.allow) {
    signal.tradeId = genTradeId();
    logAlert(signal);
    if (config.telegram.chat_id) {
      await sendMessage(config.telegram.token, config.telegram.chat_id, formatAlert(signal));
      terminalAlert(`\n❗❗❗🚨🚨🚨 RED ALERT 🚨🚨🚨❗❗❗`);
      terminalAlert(`[scanner] ALERT SENT for ${pair.symbol} ${signal.direction}`);
      terminalAlert(`❗❗❗🚨🚨🚨 RED ALERT 🚨🚨🚨❗❗❗\n`);
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
  if (noTrade.active) {
    reasons.push(`no-trade window ${noTrade.start}-${noTrade.end} ${noTrade.timezone}`);
  }
  if (!alertSession.allow) {
    reasons.push(`ICT session ${alertSession.reason}`);
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
  const noTrade = getNoTradeStatus(config);
  if (noTrade.active) {
    console.log(`[scanner] No-trade window active: ${noTrade.start}-${noTrade.end} ${noTrade.timezone}. Scans/data capture continue; new trade alerts are suppressed.`);
  }
  const activeSessions = (config.strategy?.alertSessions || defaultAlertSessions()).filter(session => sessionIsActive(session));
  if (activeSessions.length) {
    console.log(`[scanner] Active ICT alert session(s): ${activeSessions.map(s => `${s.name || 'ict_session'} ${s.start}-${s.end} ${s.timezone || 'America/New_York'}`).join(', ')}`);
  } else {
    console.log('[scanner] No active ICT alert session. Scans/data capture continue; new trade alerts are suppressed.');
  }

  const alerts = [];
  for (const [index, pair] of pairs.entries()) {
    try {
      await processFeedback(config, weights);
    } catch (e) {
      console.error(`[scanner] Telegram feedback error before ${pair.symbol}: ${e.message}`);
    }
    try {
      const sig = await scanSymbol(pair, config, weights);
      if (sig) alerts.push(sig);
    } catch (e) {
      console.error(`[scanner] Error scanning ${pair.symbol}: ${e.message}`);
    }
    if (index < pairs.length - 1) process.stdout.write('\n\n');
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
