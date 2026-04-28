/**
 * Standalone CDP client for the trading scanner.
 * Connects independently to TradingView Desktop on port 9222.
 */
import CDP from 'chrome-remote-interface';

const CDP_HOST = 'localhost';
const CDP_PORT = 9222;

const CHART_API = "window.TradingViewApi._activeChartWidgetWV.value()";
const BARS_PATH = "window.TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries().bars()";
const SYMBOL_LOAD_TIMEOUT_MS = 15000;

let _client = null;

export async function getClient() {
  if (_client) return _client;
  const targets = await CDP.List({ host: CDP_HOST, port: CDP_PORT });
  const target = targets.find(t => t.type === 'page' && t.url.includes('tradingview'));
  if (!target) throw new Error('TradingView Desktop not found on CDP port 9222. Launch it first.');
  _client = await CDP({ host: CDP_HOST, port: CDP_PORT, target: target.id });
  await _client.Runtime.enable();
  return _client;
}

export async function closeClient() {
  if (_client) { try { await _client.close(); } catch {} _client = null; }
}

export async function evaluate(expression) {
  const client = await getClient();
  const result = await client.Runtime.evaluate({
    expression,
    returnByValue: true,
    awaitPromise: true,
    timeout: 15000,
  });
  if (result.exceptionDetails) {
    throw new Error(`CDP eval error: ${result.exceptionDetails.text || JSON.stringify(result.exceptionDetails)}`);
  }
  return result.result?.value;
}

function normalizeSymbol(value) {
  return String(value || '')
    .split(':')
    .pop()
    .replace(/[^a-z0-9]/gi, '')
    .toUpperCase();
}

function symbolsMatch(requested, resolved) {
  const req = normalizeSymbol(requested);
  const got = normalizeSymbol(resolved);
  return Boolean(req && got && req === got);
}

async function chartState() {
  return await evaluate(`
    (function() {
      var chart = ${CHART_API};
      var symbol = chart && typeof chart.symbol === 'function' ? chart.symbol() : null;
      var resolution = chart && typeof chart.resolution === 'function' ? chart.resolution() : null;
      var bars = ${BARS_PATH};
      if (!bars || typeof bars.lastIndex !== 'function') {
        return { symbol: symbol, resolution: resolution, barSignature: null, barCount: 0 };
      }
      var end = bars.lastIndex();
      var start = bars.firstIndex();
      var v = bars.valueAt(end);
      return {
        symbol: symbol,
        resolution: resolution,
        barCount: Math.max(0, end - start + 1),
        barSignature: v ? [v[0], v[1], v[2], v[3], v[4]].join('|') : null,
      };
    })()
  `);
}

async function waitForSymbolData(symbol, previousState) {
  const started = Date.now();
  const previousSymbol = previousState?.symbol;
  const requireBarChange = previousSymbol && !symbolsMatch(symbol, previousSymbol);
  let lastState = null;

  while (Date.now() - started < SYMBOL_LOAD_TIMEOUT_MS) {
    lastState = await chartState();
    const symbolReady = symbolsMatch(symbol, lastState?.symbol);
    const barsReady = lastState?.barCount > 0 && lastState?.barSignature;
    const barsChanged = !requireBarChange || lastState.barSignature !== previousState?.barSignature;
    if (symbolReady && barsReady && barsChanged) return lastState;
    await waitForBars(350);
  }

  throw new Error(
    `Symbol load timeout for ${symbol}; chart=${lastState?.symbol || 'unknown'} ` +
    `barsChanged=${lastState?.barSignature !== previousState?.barSignature}`
  );
}

export async function setSymbol(symbol) {
  const before = await chartState().catch(() => null);
  await evaluate(`
    (function() {
      var chart = ${CHART_API};
      return new Promise(function(resolve) {
        chart.setSymbol(${JSON.stringify(symbol)}, {});
        setTimeout(resolve, 500);
      });
    })()
  `);
  await waitForSymbolData(symbol, before);
}

export async function getChartSymbol() {
  return await evaluate(`
    (function() {
      var chart = ${CHART_API};
      return chart && typeof chart.symbol === 'function' ? chart.symbol() : null;
    })()
  `);
}

export async function setTimeframe(tf) {
  await evaluate(`
    (function() {
      var chart = ${CHART_API};
      chart.setResolution(${JSON.stringify(tf)}, {});
    })()
  `);
  await waitForBars(2500);
}

async function waitForBars(ms) {
  await new Promise(r => setTimeout(r, ms));
}

export async function getOhlcv(count = 100) {
  const limit = Math.min(count, 500);
  const data = await evaluate(`
    (function() {
      var bars = ${BARS_PATH};
      if (!bars || typeof bars.lastIndex !== 'function') return null;
      var result = [];
      var end = bars.lastIndex();
      var start = Math.max(bars.firstIndex(), end - ${limit} + 1);
      for (var i = start; i <= end; i++) {
        var v = bars.valueAt(i);
        if (v) result.push({ time: v[0], open: v[1], high: v[2], low: v[3], close: v[4], volume: v[5] || 0 });
      }
      return result;
    })()
  `);
  if (!data || data.length === 0) throw new Error('No OHLCV data available — chart may still be loading.');
  return data;
}

export async function getCurrentPrice() {
  const data = await evaluate(`
    (function() {
      var bars = ${BARS_PATH};
      if (!bars) return null;
      var v = bars.valueAt(bars.lastIndex());
      return v ? v[4] : null;
    })()
  `);
  return data;
}
