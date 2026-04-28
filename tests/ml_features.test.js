import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';

const CONFLUENCE_FIELDS = [
  'choch_bull',
  'choch_bear',
  'choch_strength',
  'mss_bull',
  'mss_bear',
  'fib_retrace_pct',
  'fib_ote_quality',
  'ms_alignment_score',
  'ms_structure_confidence',
];

function writeSyntheticBars(filePath) {
  const rows = ['timestamp,open,high,low,close,volume'];
  const start = Date.parse('2026-01-01T00:00:00Z');
  for (let i = 0; i < 180; i++) {
    const wave = Math.sin(i / 5) * 2.5;
    const trend = i * 0.025;
    const close = 100 + trend + wave;
    const open = close - Math.sin(i / 3) * 0.35;
    const high = Math.max(open, close) + 0.4 + (i % 7) * 0.03;
    const low = Math.min(open, close) - 0.4 - (i % 5) * 0.025;
    const volume = 1000 + (i % 23) * 17;
    rows.push(`${start + i * 900000},${open},${high},${low},${close},${volume}`);
  }
  fs.writeFileSync(filePath, rows.join('\n'));
}

function readCsv(filePath) {
  const [headerLine, ...lines] = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  const header = headerLine.split(',');
  return {
    header,
    rows: lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [header[index], value]))),
  };
}

describe('ML confluence features', () => {
  it('emits populated CHOCH, Fibonacci, and multi-scale columns', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-ml-features-'));
    const input = path.join(dir, 'synthetic.csv');
    const output = path.join(dir, 'features.csv');
    writeSyntheticBars(input);

    const python = process.env.PYTHON || 'python3';
    const result = spawnSync(python, ['-m', 'strategy.ml.features', '--input', input, '--out', output], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });

    assert.equal(result.status, 0, result.stderr || result.stdout);
    const { header, rows } = readCsv(output);
    assert.ok(rows.length > 80);
    for (const field of CONFLUENCE_FIELDS) {
      assert.ok(header.includes(field), `${field} missing from feature header`);
      const values = rows.map(row => Number(row[field]));
      assert.ok(values.every(Number.isFinite), `${field} contains non-numeric values`);
    }
    const variedFields = CONFLUENCE_FIELDS.filter(field => new Set(rows.map(row => row[field])).size > 1);
    assert.ok(variedFields.length >= 3, `expected varied confluence fields, got ${variedFields.join(', ')}`);
  });
});
