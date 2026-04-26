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

function existingTimestamps(filePath) {
  if (!fs.existsSync(filePath)) return new Set();
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n').slice(1);
  return new Set(lines.map(line => line.split(',')[0]));
}

export function appendBars(config, symbol, timeframe, bars) {
  if (config?.dataCapture?.enabled === false) return { written: 0 };
  const root = dataRoot(config);
  fs.mkdirSync(root, { recursive: true });
  const filePath = path.join(root, `${safeName(symbol)}_${safeName(timeframe)}.csv`);
  if (!fs.existsSync(filePath)) fs.writeFileSync(filePath, HEADER);

  const seen = existingTimestamps(filePath);
  const rows = [];
  for (const raw of bars || []) {
    const bar = normalizeBar(raw);
    if (!bar) continue;
    const key = String(bar.timestamp);
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push(`${bar.timestamp},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`);
  }
  rows.sort((a, b) => Number(a.split(',')[0]) - Number(b.split(',')[0]));
  if (rows.length) fs.appendFileSync(filePath, rows.join('\n') + '\n');
  return { filePath, written: rows.length };
}

export function recordScanSnapshot(config, symbol, signal) {
  if (config?.dataCapture?.enabled === false) return;
  const root = dataRoot(config);
  fs.mkdirSync(root, { recursive: true });
  const filePath = path.join(root, 'scan_signals.jsonl');
  const payload = {
    timestamp: new Date().toISOString(),
    symbol,
    direction: signal.direction,
    score: signal.score,
    rr: signal.rr,
    entry: signal.entry,
    sl: signal.sl,
    tp1: signal.tp1,
    tp2: signal.tp2,
    breakdown: signal.breakdown,
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
