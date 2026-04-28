import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

const HEADER = 'timestamp,open,high,low,close,volume\n';

function dataRoot(config) {
  const configured = config?.dataCapture?.path || 'strategy/ml/data/live';
  return path.isAbsolute(configured) ? configured : path.join(REPO_ROOT, configured);
}

function safeName(value) {
  return String(value).replace(/[^a-zA-Z0-9_.-]/g, '_');
}

function barTimestamp(bar) {
  const value = bar.timestamp ?? bar.time ?? bar.t ?? bar.date;
  if (value == null) return null;
  if (typeof value === 'number') return value > 10_000_000_000 ? value : value * 1000;
  const asNumber = Number(value);
  if (Number.isFinite(asNumber)) return asNumber > 10_000_000_000 ? asNumber : asNumber * 1000;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeBar(bar) {
  const timestamp = barTimestamp(bar);
  const open = Number(bar.open ?? bar.o);
  const high = Number(bar.high ?? bar.h);
  const low = Number(bar.low ?? bar.l);
  const close = Number(bar.close ?? bar.c);
  const volume = Number(bar.volume ?? bar.v ?? 0);
  if (![timestamp, open, high, low, close, volume].every(Number.isFinite)) return null;
  return { timestamp, open, high, low, close, volume };
}

function rowForBar(bar) {
  return `${bar.timestamp},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`;
}

function loadRows(filePath) {
  const rows = new Map();
  if (!fs.existsSync(filePath)) return rows;
  const text = fs.readFileSync(filePath, 'utf8').trim();
  if (!text) return rows;
  const lines = text.split('\n').slice(1);
  for (const line of lines) {
    if (!line.trim()) continue;
    const timestamp = line.split(',')[0];
    if (timestamp) rows.set(timestamp, line);
  }
  return rows;
}

export function appendBars(config, symbol, timeframe, bars) {
  if (config?.dataCapture?.enabled === false) return { written: 0 };
  const root = dataRoot(config);
  fs.mkdirSync(root, { recursive: true });
  const filePath = path.join(root, `${safeName(symbol)}_${safeName(timeframe)}.csv`);
  if (!fs.existsSync(filePath)) fs.writeFileSync(filePath, HEADER);

  const existing = loadRows(filePath);
  const incoming = new Map();
  for (const raw of bars || []) {
    const bar = normalizeBar(raw);
    if (!bar) continue;
    const key = String(bar.timestamp);
    incoming.set(key, rowForBar(bar));
  }

  let written = 0;
  let updated = 0;
  for (const [timestamp, row] of incoming.entries()) {
    if (!existing.has(timestamp)) {
      written += 1;
    } else if (existing.get(timestamp) !== row) {
      updated += 1;
    } else {
      continue;
    }
    existing.set(timestamp, row);
  }

  if (written || updated) {
    const rows = [...existing.values()].sort((a, b) => Number(a.split(',')[0]) - Number(b.split(',')[0]));
    fs.writeFileSync(filePath, HEADER + rows.join('\n') + '\n');
  }
  return { filePath, written, updated };
}

export function recordScanSnapshot(config, symbol, signal) {
  if (config?.dataCapture?.enabled === false) return;
  const root = dataRoot(config);
  fs.mkdirSync(root, { recursive: true });
  const filePath = path.join(root, 'scan_signals.jsonl');
  const payload = {
    timestamp: new Date().toISOString(),
    symbol,
    exchange: signal.exchange,
    displaySymbol: signal.displaySymbol,
    tradingViewSymbol: signal.tradingViewSymbol,
    direction: signal.direction,
    score: signal.score,
    rr: signal.rr,
    entry: signal.entry,
    sl: signal.sl,
    tp1: signal.tp1,
    tp2: signal.tp2,
    tp3: signal.tp3,
    breakdown: signal.breakdown,
    additionalConfluence: signal.details?.additionalConfluence,
    ml: signal.details?.ml,
    mlCandidates: signal.details?.mlCandidates,
  };
  fs.appendFileSync(filePath, JSON.stringify(payload) + '\n');
}

export function recordFetchedBars(config, symbol, frames) {
  if (config?.dataCapture?.enabled === false) return;
  const results = [];
  for (const [timeframe, bars] of Object.entries(frames)) {
    results.push({ timeframe, ...appendBars(config, symbol, timeframe, bars) });
  }
  return results;
}
