import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

function resolveRepoPath(value) {
  return path.isAbsolute(value) ? value : path.join(REPO_ROOT, value);
}

function ageMinutes(filePath) {
  if (!fs.existsSync(filePath)) return Infinity;
  return (Date.now() - fs.statSync(filePath).mtimeMs) / 60000;
}

function readStateAgeMinutes(statePath) {
  if (!fs.existsSync(statePath)) return Infinity;
  try {
    const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    const last = new Date(state.lastStartedAt).getTime();
    return Number.isFinite(last) ? (Date.now() - last) / 60000 : Infinity;
  } catch {
    return Infinity;
  }
}

export function refreshNewsContext(config) {
  const refresh = config?.newsRefresh || {};
  if (refresh.enabled === false) return Promise.resolve({ refreshed: false, reason: 'disabled' });

  const retrain = config?.ml?.retrain || {};
  const newsPath = resolveRepoPath(refresh.path || retrain.newsPath || 'strategy/ml/data/raw/news/geopolitical_news_daily.csv');
  const logPath = resolveRepoPath(refresh.logPath || 'strategy/ml/reports/news_refresh.log');
  const statePath = resolveRepoPath(refresh.statePath || 'strategy/ml/reports/news_refresh_state.json');
  const lockPath = resolveRepoPath(refresh.lockPath || 'strategy/ml/reports/news_refresh.lock');
  const minMinutes = Number(refresh.minMinutesBetweenRuns ?? 15);
  const timeoutMs = Number(refresh.timeoutMs ?? 20000);
  const sourceTimeoutSec = Math.max(3, Math.floor(Number(refresh.sourceTimeoutMs ?? 8000) / 1000));
  const maxGdeltRecords = Number(refresh.maxGdeltRecords ?? 40);
  const gdeltRetries = Number(refresh.gdeltRetries ?? 0);
  const gdeltTimespan = refresh.gdeltTimespan || '1d';
  const python = config?.ml?.python || '.venv/bin/python';

  if (ageMinutes(newsPath) < minMinutes && readStateAgeMinutes(statePath) < minMinutes) {
    return Promise.resolve({ refreshed: false, reason: 'fresh' });
  }
  if (fs.existsSync(lockPath)) return Promise.resolve({ refreshed: false, reason: 'locked' });

  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  fs.writeFileSync(lockPath, new Date().toISOString());
  fs.writeFileSync(statePath, JSON.stringify({ lastStartedAt: new Date().toISOString() }, null, 2));

  const args = [
    path.join(__dirname, 'ml', 'data_sources', 'news.py'),
    '--out', newsPath,
    '--timeout', String(sourceTimeoutSec),
    '--max-gdelt', String(maxGdeltRecords),
    '--gdelt-retries', String(gdeltRetries),
    '--gdelt-timespan', gdeltTimespan,
  ];
  const log = fs.openSync(logPath, 'a');
  const startedAt = Date.now();

  return new Promise((resolve) => {
    const child = spawn(python, args, {
      cwd: REPO_ROOT,
      stdio: ['ignore', log, log],
    });
    const done = (payload) => {
      try { fs.closeSync(log); } catch {}
      try { fs.unlinkSync(lockPath); } catch {}
      resolve(payload);
    };
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      done({ refreshed: false, reason: 'timeout', elapsedMs: Date.now() - startedAt });
    }, timeoutMs);
    child.on('error', err => {
      clearTimeout(timer);
      done({ refreshed: false, reason: err.message, elapsedMs: Date.now() - startedAt });
    });
    child.on('close', code => {
      clearTimeout(timer);
      done({ refreshed: code === 0, reason: code === 0 ? 'updated' : `exit_${code}`, elapsedMs: Date.now() - startedAt });
    });
  });
}
