import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

function resolveRepoPath(value) {
  return path.isAbsolute(value) ? value : path.join(REPO_ROOT, value);
}

export function maybeStartRetraining(config) {
  const retrain = config?.ml?.retrain;
  if (!retrain?.enabled) return { started: false, reason: 'disabled' };

  const modelPath = resolveRepoPath(config.ml.modelPath || 'strategy/ml/models/rnn.pt');
  const liveDir = resolveRepoPath(config.dataCapture?.path || 'strategy/ml/data/live');
  const logPath = resolveRepoPath(retrain.logPath || 'strategy/ml/reports/retrain.log');
  const lockPath = resolveRepoPath(retrain.lockPath || 'strategy/ml/models/retrain.lock');
  const statePath = resolveRepoPath(retrain.statePath || 'strategy/ml/reports/retrain_state.json');
  const python = config.ml.python || '.venv/bin/python';
  const minHours = Number(retrain.minHoursBetweenRuns ?? 24);

  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  if (fs.existsSync(lockPath)) return { started: false, reason: 'locked' };
  if (fs.existsSync(statePath)) {
    try {
      const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
      const last = new Date(state.lastStartedAt).getTime();
      const ageHours = (Date.now() - last) / 3600000;
      if (Number.isFinite(ageHours) && ageHours < minHours) {
        return { started: false, reason: 'interval_not_elapsed', nextInHours: minHours - ageHours };
      }
    } catch {
      // Corrupt state should not permanently block retraining.
    }
  }

  const args = [
    path.join(__dirname, 'ml', 'retrain_live.py'),
    '--live-dir', liveDir,
    '--out', modelPath,
    '--min-bars', String(retrain.minBars || 250),
    '--seq-len', String(retrain.seqLen || 32),
    '--hidden', String(retrain.hidden || 32),
    '--epochs', String(retrain.epochs || 3),
    '--lock', lockPath,
  ];
  if (retrain.gprPath) args.push('--gpr', resolveRepoPath(retrain.gprPath));
  for (const seed of retrain.seedLabels || []) args.push('--seed-label', resolveRepoPath(seed));

  const log = fs.openSync(logPath, 'a');
  fs.mkdirSync(path.dirname(statePath), { recursive: true });
  fs.writeFileSync(statePath, JSON.stringify({ lastStartedAt: new Date().toISOString() }, null, 2));
  const child = spawn(python, args, {
    cwd: REPO_ROOT,
    detached: true,
    stdio: ['ignore', log, log],
  });
  child.unref();
  return { started: true, pid: child.pid, logPath };
}
