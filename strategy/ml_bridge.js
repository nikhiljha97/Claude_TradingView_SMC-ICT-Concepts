import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

function resolveModelPath(modelPath) {
  if (!modelPath) return path.join(__dirname, 'ml', 'models', 'rnn.pt');
  return path.isAbsolute(modelPath) ? modelPath : path.join(REPO_ROOT, modelPath);
}

export function mlConfig(config) {
  return {
    enabled: config?.ml?.enabled !== false,
    python: config?.ml?.python || 'python3',
    modelPath: resolveModelPath(config?.ml?.modelPath),
    minProbability: Number(config?.ml?.minProbability ?? 0.55),
    failOpen: config?.ml?.failOpen === true,
    timeoutMs: Number(config?.ml?.timeoutMs ?? 8000),
  };
}

export function scoreWithMl(config, payload) {
  const ml = mlConfig(config);
  if (!ml.enabled) {
    return Promise.resolve({ enabled: false, passed: false, reason: 'disabled' });
  }

  const script = path.join(__dirname, 'ml', 'infer.py');
  const input = JSON.stringify({ ...payload, modelPath: ml.modelPath });

  return new Promise((resolve) => {
    const child = spawn(ml.python, [script, '--model', ml.modelPath], {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      resolve({ enabled: true, passed: false, success: false, reason: 'timeout' });
    }, ml.timeoutMs);

    child.stdout.on('data', chunk => stdout += chunk);
    child.stderr.on('data', chunk => stderr += chunk);
    child.on('error', err => {
      clearTimeout(timer);
      resolve({ enabled: true, passed: false, success: false, reason: err.message });
    });
    child.on('close', () => {
      clearTimeout(timer);
      try {
        const result = JSON.parse(stdout || '{}');
        const probability = Number(result.probability);
        const passed = result.success
          ? probability >= ml.minProbability
          : ml.failOpen;
        resolve({
          enabled: true,
          ...result,
          probability: Number.isFinite(probability) ? probability : null,
          minProbability: ml.minProbability,
          passed,
          stderr: stderr.trim() || undefined,
        });
      } catch (err) {
        resolve({
          enabled: true,
          passed: false,
          success: false,
          reason: 'invalid_ml_response',
          stderr: stderr.trim() || undefined,
        });
      }
    });

    child.stdin.end(input);
  });
}
