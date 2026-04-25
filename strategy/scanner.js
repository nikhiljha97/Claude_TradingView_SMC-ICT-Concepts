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
import { sendMessage, getUpdates, formatAlert, parseFeedback } from './telegram.js';
import {
  loadTrades, loadWeights, saveWeights, effectiveWeights,
  logAlert, applyFeedback, applyConfirmation, parseConfirmation, genTradeId, formatStats
} from './learning.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.join(__dirname, 'config.json');

function loadConfig() { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); }
function saveConfig(c) { fs.writeFileSync(CONFIG_PATH, JSON.stringify(c, null, 2)); }

function isMarketOpen(pair, now = new Date()) {
  if (pair.alwaysOpen) return true;
  // XAUUSD: closed Friday 17:00 ET → Sunday 18:00 ET
  // Approximate using local time (user said "Sunday 6PM to Friday 5PM")
  const day = now.getDay();   // 0=Sun, 5=Fri, 6=Sat
  const hour = now.getHours();
  if (day === 6) return false;                         // all of Saturday closed
  if (day === 5 && hour >= 17) return false;           // Fri after 5pm
  if (day === 0 && hour < 18) return false;            // Sun before 6pm
  return true;
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
        const reply = conf.took
          ? `✅ \`${conf.tradeId}\` marked as *taken* — cooldown active, won't re-alert this setup.\nReply \`TP HIT ${conf.tradeId}\` or \`SL HIT ${conf.tradeId}\` when done.`
          : `⏭ \`${conf.tradeId}\` marked as *skipped* — will re-alert if setup holds next cycle.`;
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

async function scanSymbol(pair, config, weights) {
  console.log(`[scanner] Scanning ${pair.symbol}...`);
  await setSymbol(pair.symbol);

  // Fetch OHLCV at all timeframes (chart switches between calls)
  const barsWeekly = await fetchTimeframeBars(pair.symbol, 'W', 30);
  const barsDaily  = await fetchTimeframeBars(pair.symbol, 'D', 60);
  const bars4H     = await fetchTimeframeBars(pair.symbol, '240', 100);
  const bars1H     = await fetchTimeframeBars(pair.symbol, '60', 100);
  const bars15M    = await fetchTimeframeBars(pair.symbol, '15', 60);

  const eff = effectiveWeights(weights);
  const signal = computeSignal({
    symbol: pair.symbol,
    bars4H, bars1H, bars15M, barsDaily, barsWeekly,
    weights: eff,
  });

  console.log(`[scanner] ${pair.symbol}: ${signal.direction} | score ${signal.score}/${signal.maxScore.toFixed(1)} | RR ${signal.rr || 'n/a'}`);

  // Only apply cooldown if user explicitly confirmed they took the trade (YES or TP/SL reply).
  // Unconfirmed (null) or skipped (false) trades do not trigger cooldown.
  const recentTrades = loadTrades().filter(t => {
    const age = (Date.now() - new Date(t.timestamp).getTime()) / 3600000;
    return t.symbol === pair.symbol && age < 4 && t.direction === signal.direction && t.confirmed === true;
  });

  if (signal.direction !== 'NONE' &&
      signal.score >= config.strategy.alertScoreThreshold &&
      signal.rr >= config.strategy.minRR &&
      recentTrades.length === 0) {
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

  return null;
}

async function main() {
  const config = loadConfig();
  const weights = loadWeights(config.weights);
  const startTime = Date.now();
  const estTime = new Date().toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  console.log(`[scanner] === Cycle started at ${estTime} EST ===`);

  // 1. Process any pending Telegram feedback first
  try {
    await processFeedback(config, weights);
  } catch (e) {
    console.error(`[scanner] Telegram feedback error: ${e.message}`);
  }

  // 2. Scan each pair
  const now = new Date();
  const alerts = [];
  for (const pair of config.pairs) {
    if (!isMarketOpen(pair, now)) {
      console.log(`[scanner] ${pair.symbol} market closed — skipping.`);
      continue;
    }
    try {
      const sig = await scanSymbol(pair, config, weights);
      if (sig) alerts.push(sig);
    } catch (e) {
      console.error(`[scanner] Error scanning ${pair.symbol}: ${e.message}`);
    }
  }

  await closeClient();
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[scanner] === Cycle complete in ${elapsed}s, ${alerts.length} alert(s) fired ===`);
}

main().catch(err => {
  console.error('[scanner] Fatal:', err);
  process.exit(1);
});
