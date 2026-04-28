/**
 * SMC + Pivot Boss + ICT Strategy Analyzer
 *
 * Synthesized from:
 *   - "Smart Money SMC" by VasilyTrader (BOS, CHoCH, OBs, FVGs, Liquidity Sweeps)
 *   - "Secrets of a Pivot Boss" by F. Ochoa (Floor Pivots, Camarilla, Pivot Trend)
 *   - ICT (Inner Circle Trader) — Killzones, Power of 3, OTE, MSS, BSL/SSL, NDOG, Silver Bullet
 *   - "Inside the Black Box" by Narang (Alpha models, systematic rules)
 *   - "Quantitative Trading" by E. Chan + "Max Dama on Automated Trading" (Kelly, EWMA signals)
 *   - "Building Winning Algo Trading Systems" by K. Davey (robustness, no curve-fitting)
 *
 * Entry TF: 15M | Analysis TFs: Weekly, Daily, 4H, 1H | Min RR: 1:2.5
 */

import { calcFloorPivots, calcCamarilla, getPivotTrend, nearPivot, nearCamarilla } from './pivots.js';

// ─── Market Structure ───────────────────────────────────────────────────────

/**
 * Identify swing highs and lows.
 * Returns array of { idx, price, type: 'high'|'low', bar }
 */
export function findSwings(bars, lookback = 5) {
  const swings = [];
  for (let i = lookback; i < bars.length - lookback; i++) {
    let isHigh = true, isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (bars[j].high >= bars[i].high) isHigh = false;
      if (bars[j].low  <= bars[i].low)  isLow  = false;
    }
    if (isHigh) swings.push({ idx: i, price: bars[i].high, type: 'high', bar: bars[i] });
    if (isLow)  swings.push({ idx: i, price: bars[i].low,  type: 'low',  bar: bars[i] });
  }
  return swings;
}

/**
 * Determine market trend from swing sequence (Higher Highs + Higher Lows = bullish).
 */
export function detectTrend(bars, lookback = 5) {
  const swings = findSwings(bars, lookback);
  const highs = swings.filter(s => s.type === 'high').slice(-4);
  const lows  = swings.filter(s => s.type === 'low').slice(-4);
  if (highs.length < 2 || lows.length < 2) return 'ranging';

  const hhCount = highs.filter((h, i) => i > 0 && h.price > highs[i - 1].price).length;
  const hlCount = lows.filter((l, i)  => i > 0 && l.price > lows[i - 1].price).length;
  const lhCount = highs.filter((h, i) => i > 0 && h.price < highs[i - 1].price).length;
  const llCount = lows.filter((l, i)  => i > 0 && l.price < lows[i - 1].price).length;

  const bullScore = hhCount + hlCount;
  const bearScore = lhCount + llCount;
  if (bullScore > bearScore + 1) return 'bullish';
  if (bearScore > bullScore + 1) return 'bearish';
  return 'ranging';
}

function clamp(value, low = 0, high = 1) {
  return Math.max(low, Math.min(high, Number.isFinite(value) ? value : low));
}

function trendValue(trend) {
  if (trend === 'bullish') return 1;
  if (trend === 'bearish') return -1;
  return 0;
}

export function derivePivotPoints(bars, lookback = 3) {
  return findSwings(bars, lookback).map(swing => ({
    idx: swing.idx,
    price: swing.price,
    type: swing.type,
    barsAgo: bars.length - 1 - swing.idx,
  }));
}

function fibLocationScore(bars, side) {
  const direction = side === 'BUY' ? 'bullish' : side === 'SELL' ? 'bearish' : null;
  if (!direction) return { score: 0, inOTE: false };
  const ote = checkOTE(bars, direction);
  if (ote.inOTE) return { score: 1, inOTE: true, ...ote };

  const pivots = derivePivotPoints(bars, 3).slice(-8);
  let leg = null;
  for (let i = pivots.length - 1; i > 0; i--) {
    if (pivots[i].type !== pivots[i - 1].type) {
      leg = [pivots[i - 1], pivots[i]];
      break;
    }
  }
  if (!leg) return { score: 0, inOTE: false, ...ote };

  const [start, end] = leg;
  const range = Math.abs(end.price - start.price);
  if (!range) return { score: 0, inOTE: false, ...ote };
  const current = bars[bars.length - 1].close;
  const retrace = side === 'BUY'
    ? (end.price - current) / range
    : (current - end.price) / range;
  const distance = Math.min(Math.abs(retrace - 0.618), Math.abs(retrace - 0.705), Math.abs(retrace - 0.79));
  return {
    score: clamp(1 - distance / 0.2),
    inOTE: false,
    retrace,
    ...ote,
  };
}

export function deriveConfluenceFeatures({ bars15M, bars1H, bars4H, direction, currentPrice, maxBonus = 0.75 }) {
  if (!['BUY', 'SELL'].includes(direction)) {
    return {
      chochScore: 0,
      fibOteScore: 0,
      structureAlignmentScore: 0,
      confluenceBonus: 0,
    };
  }

  const wanted = direction === 'BUY' ? 1 : -1;
  const mss = detectMSS(bars15M, 20);
  const bos = detectBOS(bars15M, 5);
  const pivots = derivePivotPoints(bars15M, 3);
  const lastHigh = [...pivots].reverse().find(p => p.type === 'high');
  const lastLow = [...pivots].reverse().find(p => p.type === 'low');
  const mssAligned = mss.detected && ((direction === 'BUY' && mss.direction === 'bullish') || (direction === 'SELL' && mss.direction === 'bearish'));
  const bosAligned = bos.detected && ((direction === 'BUY' && bos.direction === 'bullish') || (direction === 'SELL' && bos.direction === 'bearish'));
  const levelBreak = direction === 'BUY'
    ? lastHigh && currentPrice > lastHigh.price
    : lastLow && currentPrice < lastLow.price;
  const chochScore = mssAligned ? 1 : bosAligned && levelBreak ? 0.7 : levelBreak ? 0.45 : 0;

  const fib = fibLocationScore(bars15M, direction);
  const fibOteScore = fib.score;

  const trends = [detectTrend(bars15M, 3), detectTrend(bars1H, 4), detectTrend(bars4H, 4)].map(trendValue);
  const aligned = trends.filter(value => value === wanted).length;
  const opposed = trends.filter(value => value === -wanted).length;
  const structureAlignmentScore = clamp((aligned - opposed * 0.5) / trends.length);

  const raw = chochScore * 0.4 + fibOteScore * 0.25 + structureAlignmentScore * 0.35;
  return {
    chochScore: Math.round(chochScore * 1000) / 1000,
    fibOteScore: Math.round(fibOteScore * 1000) / 1000,
    structureAlignmentScore: Math.round(structureAlignmentScore * 1000) / 1000,
    confluenceBonus: Math.round(raw * maxBonus * 10) / 10,
    mssAligned,
    bosAligned,
    levelBreak: !!levelBreak,
    fib,
    trends: { m15: trends[0], h1: trends[1], h4: trends[2] },
  };
}

// ─── BOS / CHoCH / MSS ──────────────────────────────────────────────────────

/**
 * Detect Break of Structure (BOS) — continuation signal.
 * Returns { detected, direction, breakLevel, barsAgo }
 */
export function detectBOS(bars, swingLookback = 5) {
  const swings = findSwings(bars, swingLookback);

  const recentHighs = swings.filter(s => s.type === 'high' && s.idx >= bars.length - 25).slice(-3);
  const recentLows  = swings.filter(s => s.type === 'low'  && s.idx >= bars.length - 25).slice(-3);

  let bos = { detected: false, direction: null, breakLevel: null, barsAgo: null };

  if (recentHighs.length > 0) {
    const lastHigh = recentHighs[recentHighs.length - 1];
    for (let i = bars.length - 5; i < bars.length; i++) {
      if (bars[i] && bars[i].close > lastHigh.price) {
        bos = { detected: true, direction: 'bullish', breakLevel: lastHigh.price, barsAgo: bars.length - 1 - i };
        break;
      }
    }
  }

  if (!bos.detected && recentLows.length > 0) {
    const lastLow = recentLows[recentLows.length - 1];
    for (let i = bars.length - 5; i < bars.length; i++) {
      if (bars[i] && bars[i].close < lastLow.price) {
        bos = { detected: true, direction: 'bearish', breakLevel: lastLow.price, barsAgo: bars.length - 1 - i };
        break;
      }
    }
  }
  return bos;
}

/**
 * ICT Market Structure Shift (MSS) — stricter than BOS.
 * MSS requires a liquidity sweep FIRST, then a close beyond the opposing swing.
 * This is the CHoCH pattern: after a sweep of SSL, price closes above the last swing high → bullish MSS.
 *
 * Returns { detected, direction, sweptLevel, mssLevel, barsAgo }
 */
export function detectMSS(bars, lookback = 15) {
  const n = bars.length;
  if (n < lookback + 5) return { detected: false };

  const swings = findSwings(bars, 3);

  // Look for sweep in last lookback bars, then a close beyond the opposite structure
  for (let i = n - lookback; i < n - 2; i++) {
    const bar = bars[i];
    if (!bar) continue;

    // Check for bullish MSS: wick below a swing low, then close above a swing high
    const priorLows  = swings.filter(s => s.type === 'low'  && s.idx < i && s.idx > i - 20);
    const priorHighs = swings.filter(s => s.type === 'high' && s.idx < i && s.idx > i - 20);

    if (priorLows.length > 0 && priorHighs.length > 0) {
      const nearestLow  = priorLows[priorLows.length - 1];
      const nearestHigh = priorHighs[priorHighs.length - 1];

      // Bullish MSS: sweep below swing low (wick), then close above swing high
      if (bar.low < nearestLow.price && bar.close > nearestLow.price) {
        // Look for close above swing high in subsequent bars
        for (let j = i + 1; j < Math.min(i + 6, n); j++) {
          if (bars[j] && bars[j].close > nearestHigh.price) {
            return {
              detected: true, direction: 'bullish',
              sweptLevel: nearestLow.price, mssLevel: nearestHigh.price,
              barsAgo: n - 1 - j
            };
          }
        }
      }

      // Bearish MSS: sweep above swing high (wick), then close below swing low
      if (bar.high > nearestHigh.price && bar.close < nearestHigh.price) {
        for (let j = i + 1; j < Math.min(i + 6, n); j++) {
          if (bars[j] && bars[j].close < nearestLow.price) {
            return {
              detected: true, direction: 'bearish',
              sweptLevel: nearestHigh.price, mssLevel: nearestLow.price,
              barsAgo: n - 1 - j
            };
          }
        }
      }
    }
  }

  return { detected: false };
}

// ─── Order Blocks ────────────────────────────────────────────────────────────

/**
 * Detect Order Blocks (VasilyTrader + ICT refinement):
 * Bullish OB = last bearish candle before a BOS to the upside
 * Bearish OB = last bullish candle before a BOS to the downside
 *
 * ICT refinement: OB is valid only if it caused a displacement (large FVG candle after it).
 */
export function findOrderBlocks(bars, swingLookback = 5) {
  const obs = [];
  const swings = findSwings(bars, swingLookback);

  for (let i = 10; i < bars.length; i++) {
    // Bullish BOS
    const prevHighs = swings.filter(s => s.type === 'high' && s.idx < i && s.idx > i - 30);
    if (prevHighs.length > 0) {
      const lastSwingHigh = prevHighs[prevHighs.length - 1].price;
      if (bars[i].close > lastSwingHigh) {
        for (let j = i - 1; j >= Math.max(0, i - 15); j--) {
          if (bars[j].close < bars[j].open) {
            // ICT: check for displacement candle (large body) after this OB
            const displacementBar = bars[j + 1] || bars[j + 2];
            const isDisplacement = displacementBar &&
              (displacementBar.close - displacementBar.open) > calcATR(bars.slice(0, j + 2), 5) * 0.7;
            obs.push({
              type: 'bullish',
              high: bars[j].high, low: bars[j].low,
              open: bars[j].open, close: bars[j].close,
              idx: j, barsAgo: bars.length - 1 - j,
              isICT: isDisplacement, mitigation: false
            });
            break;
          }
        }
      }
    }

    // Bearish BOS
    const prevLows = swings.filter(s => s.type === 'low' && s.idx < i && s.idx > i - 30);
    if (prevLows.length > 0) {
      const lastSwingLow = prevLows[prevLows.length - 1].price;
      if (bars[i].close < lastSwingLow) {
        for (let j = i - 1; j >= Math.max(0, i - 15); j--) {
          if (bars[j].close > bars[j].open) {
            const displacementBar = bars[j + 1] || bars[j + 2];
            const isDisplacement = displacementBar &&
              (displacementBar.open - displacementBar.close) > calcATR(bars.slice(0, j + 2), 5) * 0.7;
            obs.push({
              type: 'bearish',
              high: bars[j].high, low: bars[j].low,
              open: bars[j].open, close: bars[j].close,
              idx: j, barsAgo: bars.length - 1 - j,
              isICT: isDisplacement, mitigation: false
            });
            break;
          }
        }
      }
    }
  }

  return obs.map(ob => {
    const mitigated = ob.type === 'bullish'
      ? bars.some((b, i) => i > ob.idx && b.close < ob.low)
      : bars.some((b, i) => i > ob.idx && b.close > ob.high);
    return { ...ob, mitigation: mitigated };
  }).filter(ob => !ob.mitigation && ob.barsAgo < 50);
}

/**
 * ICT Breaker Blocks: a mitigated OB that price returns to from the other side.
 * Bullish Breaker: was a bearish OB, got mitigated (price closed above), now acts as support.
 * Returns array of { type, high, low, barsAgo }
 */
export function findBreakerBlocks(bars, swingLookback = 5) {
  const breakers = [];
  const swings = findSwings(bars, swingLookback);

  for (let i = 10; i < bars.length; i++) {
    // Similar OB detection but track mitigated ones
    const prevHighs = swings.filter(s => s.type === 'high' && s.idx < i && s.idx > i - 30);
    if (prevHighs.length > 0) {
      const lastSwingHigh = prevHighs[prevHighs.length - 1].price;
      if (bars[i].close > lastSwingHigh) {
        for (let j = i - 1; j >= Math.max(0, i - 15); j--) {
          if (bars[j].close < bars[j].open) {
            // This was a bullish OB - check if it was later mitigated (traded through)
            const mitigated = bars.some((b, bi) => bi > i && b.close < bars[j].low);
            if (mitigated) {
              // Now check if price is returning to it from above (bullish breaker support)
              const currentPrice = bars[bars.length - 1].close;
              if (currentPrice >= bars[j].low && currentPrice <= bars[j].high * 1.002) {
                breakers.push({
                  type: 'bullish_breaker', high: bars[j].high, low: bars[j].low,
                  idx: j, barsAgo: bars.length - 1 - j
                });
              }
            }
            break;
          }
        }
      }
    }

    const prevLows = swings.filter(s => s.type === 'low' && s.idx < i && s.idx > i - 30);
    if (prevLows.length > 0) {
      const lastSwingLow = prevLows[prevLows.length - 1].price;
      if (bars[i].close < lastSwingLow) {
        for (let j = i - 1; j >= Math.max(0, i - 15); j--) {
          if (bars[j].close > bars[j].open) {
            const mitigated = bars.some((b, bi) => bi > i && b.close > bars[j].high);
            if (mitigated) {
              const currentPrice = bars[bars.length - 1].close;
              if (currentPrice >= bars[j].low * 0.998 && currentPrice <= bars[j].high) {
                breakers.push({
                  type: 'bearish_breaker', high: bars[j].high, low: bars[j].low,
                  idx: j, barsAgo: bars.length - 1 - j
                });
              }
            }
            break;
          }
        }
      }
    }
  }

  return breakers.filter(b => b.barsAgo < 100);
}

// ─── Fair Value Gaps ──────────────────────────────────────────────────────────

/**
 * FVG / Imbalance detection.
 * Bullish FVG: candle[i-2].high < candle[i].low
 * Bearish FVG: candle[i-2].low  > candle[i].high
 */
export function findFVGs(bars, minSizePct = 0.0005) {
  const fvgs = [];
  for (let i = 2; i < bars.length; i++) {
    const prev2 = bars[i - 2];
    const curr  = bars[i];
    const midClose = curr.close;

    if (prev2.high < curr.low) {
      const size = (curr.low - prev2.high) / midClose;
      if (size >= minSizePct) {
        fvgs.push({
          type: 'bullish', high: curr.low, low: prev2.high,
          mid: (curr.low + prev2.high) / 2,
          size, idx: i, barsAgo: bars.length - 1 - i
        });
      }
    }
    if (prev2.low > curr.high) {
      const size = (prev2.low - curr.high) / midClose;
      if (size >= minSizePct) {
        fvgs.push({
          type: 'bearish', high: prev2.low, low: curr.high,
          mid: (prev2.low + curr.high) / 2,
          size, idx: i, barsAgo: bars.length - 1 - i
        });
      }
    }
  }
  return fvgs.map(fvg => {
    const filled = fvg.type === 'bullish'
      ? bars.some((b, i) => i > fvg.idx && b.close < fvg.low)
      : bars.some((b, i) => i > fvg.idx && b.close > fvg.high);
    return { ...fvg, filled };
  }).filter(f => !f.filled && f.barsAgo < 50);
}

// ─── ICT: Buy-side / Sell-side Liquidity (Equal Highs/Lows) ──────────────────

/**
 * ICT BSL/SSL: equal highs are buy-side liquidity (stop hunts above),
 * equal lows are sell-side liquidity (stop hunts below).
 * Returns { bsl: [levels], ssl: [levels] }
 */
export function findLiquidityLevels(bars, lookback = 30, tolerance = 0.001) {
  const window = bars.slice(-lookback);
  const bsl = []; // equal highs = buy-side liquidity
  const ssl = []; // equal lows = sell-side liquidity

  for (let i = 0; i < window.length - 1; i++) {
    for (let j = i + 2; j < window.length; j++) {
      const highDiff = Math.abs(window[i].high - window[j].high) / window[i].high;
      if (highDiff < tolerance) {
        const level = (window[i].high + window[j].high) / 2;
        if (!bsl.find(l => Math.abs(l - level) / level < tolerance)) {
          bsl.push(level);
        }
      }
      const lowDiff = Math.abs(window[i].low - window[j].low) / window[i].low;
      if (lowDiff < tolerance) {
        const level = (window[i].low + window[j].low) / 2;
        if (!ssl.find(l => Math.abs(l - level) / level < tolerance)) {
          ssl.push(level);
        }
      }
    }
  }

  return { bsl, ssl };
}

// ─── Liquidity Sweep Detection ────────────────────────────────────────────────

/**
 * Detect liquidity sweep (enhanced with ICT BSL/SSL awareness).
 * Bullish sweep: wick below swing low or SSL level + closes back above → buy signal.
 * Returns { detected, direction, sweptLevel, sweepBar, confirmBar, isSSL, isBSL }
 */
export function detectLiquiditySweep(bars, lookback = 10) {
  const n = bars.length;
  if (n < lookback + 3) return { detected: false };

  const windowBars = bars.slice(n - lookback - 3, n - 3);
  const swingHighs = [];
  const swingLows  = [];
  for (let i = 2; i < windowBars.length - 2; i++) {
    if (windowBars[i].high > windowBars[i-1].high && windowBars[i].high > windowBars[i+1].high &&
        windowBars[i].high > windowBars[i-2].high && windowBars[i].high > windowBars[i+2].high) {
      swingHighs.push(windowBars[i].high);
    }
    if (windowBars[i].low < windowBars[i-1].low && windowBars[i].low < windowBars[i+1].low &&
        windowBars[i].low < windowBars[i-2].low && windowBars[i].low < windowBars[i+2].low) {
      swingLows.push(windowBars[i].low);
    }
  }

  // Also add BSL/SSL equal high/low levels
  const { bsl, ssl } = findLiquidityLevels(windowBars, lookback, 0.001);
  const allHighs = [...new Set([...swingHighs, ...bsl])];
  const allLows  = [...new Set([...swingLows, ...ssl])];

  for (let offset = 0; offset <= 2; offset++) {
    const sweepBar = bars[n - 3 + offset];
    const confirmBar = offset < 2 ? bars[n - 2 + offset] : bars[n - 1];
    if (!sweepBar || !confirmBar) continue;

    for (const sl of allLows) {
      if (sweepBar.low < sl && sweepBar.close > sl) {
        const isBullishConfirm = confirmBar.close > confirmBar.open && confirmBar.close > sweepBar.close;
        return {
          detected: true, direction: 'bullish',
          sweptLevel: sl, sweepBar, confirmBar,
          hasImbalance: isBullishConfirm, barsAgo: offset,
          isSSL: ssl.includes(sl), isBSL: false
        };
      }
    }

    for (const sh of allHighs) {
      if (sweepBar.high > sh && sweepBar.close < sh) {
        const isBearishConfirm = confirmBar.close < confirmBar.open && confirmBar.close < sweepBar.close;
        return {
          detected: true, direction: 'bearish',
          sweptLevel: sh, sweepBar, confirmBar,
          hasImbalance: isBearishConfirm, barsAgo: offset,
          isBSL: bsl.includes(sh), isSSL: false
        };
      }
    }
  }

  return { detected: false };
}

// ─── Premium / Discount Zones ─────────────────────────────────────────────────

export function getPremiumDiscount(bars, lookback = 20) {
  const window = bars.slice(-lookback);
  const rangeHigh = Math.max(...window.map(b => b.high));
  const rangeLow  = Math.min(...window.map(b => b.low));
  const current = bars[bars.length - 1].close;
  const position = (current - rangeLow) / (rangeHigh - rangeLow);

  let zone = 'equilibrium';
  if (position < 0.45) zone = 'discount';
  else if (position > 0.55) zone = 'premium';

  return { zone, position: Math.round(position * 100), rangeHigh, rangeLow, mid: (rangeHigh + rangeLow) / 2 };
}

// ─── ATR ──────────────────────────────────────────────────────────────────────

export function calcATR(bars, period = 14) {
  if (bars.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < bars.length; i++) {
    const hl = bars[i].high - bars[i].low;
    const hc = Math.abs(bars[i].high - bars[i - 1].close);
    const lc = Math.abs(bars[i].low  - bars[i - 1].close);
    trs.push(Math.max(hl, hc, lc));
  }
  return trs.slice(-period).reduce((a, b) => a + b, 0) / period;
}

// ─── ICT: New Day Opening Gap (NDOG) ─────────────────────────────────────────

/**
 * NDOG: gap between yesterday's close and today's open (midnight open).
 * Price tends to be magnetic to the midpoint of this gap.
 * Returns { detected, gapHigh, gapLow, gapMid, direction }
 */
export function findNDOG(barsDaily, bars15M) {
  if (barsDaily.length < 2 || bars15M.length < 2) return { detected: false };

  const yesterday = barsDaily[barsDaily.length - 2];
  const today     = barsDaily[barsDaily.length - 1];
  const currentPrice = bars15M[bars15M.length - 1].close;

  // Gap up: today's open > yesterday's close
  if (today.open > yesterday.close) {
    const gap = today.open - yesterday.close;
    const gapPct = gap / yesterday.close;
    if (gapPct > 0.0003) { // at least 0.03% gap
      return {
        detected: true, direction: 'up',
        gapHigh: today.open, gapLow: yesterday.close,
        gapMid: (today.open + yesterday.close) / 2,
        filledToday: currentPrice <= today.open && currentPrice >= yesterday.close
      };
    }
  }

  // Gap down: today's open < yesterday's close
  if (today.open < yesterday.close) {
    const gap = yesterday.close - today.open;
    const gapPct = gap / yesterday.close;
    if (gapPct > 0.0003) {
      return {
        detected: true, direction: 'down',
        gapHigh: yesterday.close, gapLow: today.open,
        gapMid: (yesterday.close + today.open) / 2,
        filledToday: currentPrice >= today.open && currentPrice <= yesterday.close
      };
    }
  }

  return { detected: false };
}

// ─── ICT: Optimal Trade Entry (OTE) Fibonacci Zone ───────────────────────────

/**
 * ICT OTE: after a swing impulse move, look for price to retrace to 62%-79% fib level.
 * This is the premium/discount entry after MSS or BOS.
 * Returns { inOTE, oteHigh, oteLow, ote618, ote705, ote79 }
 */
export function checkOTE(bars, direction) {
  const swings = findSwings(bars, 3);
  const n = bars.length;
  const currentPrice = bars[n - 1].close;

  if (direction === 'bullish') {
    // For long OTE: find the most recent swing low → swing high impulse, then retrace to 62-79%
    const recentLows  = swings.filter(s => s.type === 'low').slice(-3);
    const recentHighs = swings.filter(s => s.type === 'high').slice(-3);

    if (recentLows.length > 0 && recentHighs.length > 0) {
      // Most recent completed swing: low before the high
      const swingLow  = recentLows[recentLows.length - 1];
      const swingHigh = recentHighs.find(h => h.idx > swingLow.idx);
      if (swingHigh) {
        const range = swingHigh.price - swingLow.price;
        const ote618 = swingHigh.price - range * 0.618;
        const ote705 = swingHigh.price - range * 0.705;
        const ote79  = swingHigh.price - range * 0.79;
        const inOTE  = currentPrice >= ote79 && currentPrice <= ote618;
        return { inOTE, oteHigh: ote618, oteLow: ote79, ote618, ote705, ote79 };
      }
    }
  } else if (direction === 'bearish') {
    const recentHighs = swings.filter(s => s.type === 'high').slice(-3);
    const recentLows  = swings.filter(s => s.type === 'low').slice(-3);

    if (recentHighs.length > 0 && recentLows.length > 0) {
      const swingHigh = recentHighs[recentHighs.length - 1];
      const swingLow  = recentLows.find(l => l.idx > swingHigh.idx);
      if (swingLow) {
        const range = swingHigh.price - swingLow.price;
        const ote618 = swingLow.price + range * 0.618;
        const ote705 = swingLow.price + range * 0.705;
        const ote79  = swingLow.price + range * 0.79;
        const inOTE  = currentPrice <= ote79 && currentPrice >= ote618;
        return { inOTE, oteHigh: ote79, oteLow: ote618, ote618, ote705, ote79 };
      }
    }
  }

  return { inOTE: false };
}

// ─── ICT: Killzone Detection ──────────────────────────────────────────────────

/**
 * ICT Killzones — corrected EST times (EST = UTC-5).
 *
 * Asia Range    : 8PM–12AM EST  = 01:00–05:00 UTC
 * London Open   : 2AM–5AM EST   = 07:00–10:00 UTC  (SB: 3–4AM EST = 08:00–09:00 UTC)
 * NY Open       : 7AM–10AM EST  = 12:00–15:00 UTC
 * AM Silver Bullet : 10–11AM EST = 15:00–16:00 UTC
 * London Close  : 10AM–12PM EST = 15:00–17:00 UTC  (position squaring / reversals)
 * PM Silver Bullet : 2–3PM EST  = 19:00–20:00 UTC
 *
 * ICT Macros — 20-min high-probability windows (IPDA seeks liquidity / fills FVGs):
 *   London:  2:33–3:00 EST (07:33–08:00 UTC), 4:03–4:30 EST (09:03–09:30 UTC)
 *   NY AM:   8:50–9:10 EST (13:50–14:10 UTC), 9:50–10:10 EST (14:50–15:10 UTC),
 *            10:50–11:10 EST (15:50–16:10 UTC)
 *   NY Lunch/PM: 11:50–12:10 EST (16:50–17:10 UTC), 13:10–13:40 EST (18:10–18:40 UTC),
 *                15:15–15:45 EST (20:15–20:45 UTC)
 *
 * Returns { name: string|null, isSilverBullet: bool, isMacro: bool }
 */
export function getCurrentKillzone(date = new Date()) {
  const utcHour = date.getUTCHours();
  const utcMin  = date.getUTCMinutes();
  const utcTime = utcHour + utcMin / 60;

  // ICT Macro windows (high-precision 20-min liquidity-seeking intervals)
  const isMacro = (
    (utcTime >= 7.55  && utcTime < 8.0)   ||  // London 2:33–3:00 EST
    (utcTime >= 9.05  && utcTime < 9.5)   ||  // London 4:03–4:30 EST
    (utcTime >= 13.83 && utcTime < 14.17) ||  // NY AM  8:50–9:10 EST
    (utcTime >= 14.83 && utcTime < 15.17) ||  // NY AM  9:50–10:10 EST
    (utcTime >= 15.83 && utcTime < 16.17) ||  // NY AM  10:50–11:10 EST
    (utcTime >= 16.83 && utcTime < 17.17) ||  // NY Lunch 11:50–12:10 EST
    (utcTime >= 18.17 && utcTime < 18.67) ||  // NY PM  13:10–13:40 EST
    (utcTime >= 20.25 && utcTime < 20.75)     // NY PM  15:15–15:45 EST
  );

  // Asia Range: 8PM–12AM EST = 01:00–05:00 UTC
  if (utcTime >= 1 && utcTime < 5) return { name: 'Asia', isSilverBullet: false, isMacro };

  // London Open: 2AM–5AM EST = 07:00–10:00 UTC
  // London Silver Bullet: 3–4AM EST = 08:00–09:00 UTC (highest probability within London)
  if (utcTime >= 7 && utcTime < 10) {
    const isSilverBullet = utcTime >= 8 && utcTime < 9;
    return { name: 'London', isSilverBullet, isMacro };
  }

  // NY Open: 7AM–10AM EST = 12:00–15:00 UTC
  if (utcTime >= 12 && utcTime < 15) return { name: 'NY_AM', isSilverBullet: false, isMacro };

  // AM Silver Bullet: 10–11AM EST = 15:00–16:00 UTC (overlaps London Close — SB takes priority)
  if (utcTime >= 15 && utcTime < 16) return { name: 'SilverBullet', isSilverBullet: true, isMacro };

  // London Close: 11AM–12PM EST = 16:00–17:00 UTC (profit-taking / potential reversals)
  if (utcTime >= 16 && utcTime < 17) return { name: 'London_Close', isSilverBullet: false, isMacro };

  // PM Silver Bullet: 2–3PM EST = 19:00–20:00 UTC
  if (utcTime >= 19 && utcTime < 20) return { name: 'PM_SilverBullet', isSilverBullet: true, isMacro };

  return { name: null, isSilverBullet: false, isMacro };
}

// ─── ICT: Power of 3 Phase Detection ─────────────────────────────────────────

/**
 * ICT Power of 3: Accumulation → Manipulation → Distribution.
 * On 15M bars relative to the current session:
 *   - Accumulation: Asia session (tight range, low volatility)
 *   - Manipulation: London open sweep of Asia range (fake move)
 *   - Distribution: NY session (real directional move)
 *
 * Returns { phase: 'accumulation'|'manipulation'|'distribution'|'unknown', asiaRange }
 */
export function detectPowerOf3(bars15M) {
  const now = new Date();
  const kz = getCurrentKillzone(now);
  const n = bars15M.length;

  if (n < 20) return { phase: 'unknown' };

  // Calculate Asia range (last ~16 bars = 4 hours of 15M)
  const asiaBars = bars15M.slice(-32, -16); // rough proxy for Asia session bars
  if (asiaBars.length === 0) return { phase: 'unknown' };

  const asiaHigh = Math.max(...asiaBars.map(b => b.high));
  const asiaLow  = Math.min(...asiaBars.map(b => b.low));
  const asiaRange = asiaHigh - asiaLow;
  const currentPrice = bars15M[n - 1].close;

  let phase = 'unknown';
  if (kz.name === 'Asia') phase = 'accumulation';
  else if (kz.name === 'London') {
    // Check if London is sweeping the Asia range
    const londonBars = bars15M.slice(-12);
    const londonSweepedHigh = londonBars.some(b => b.high > asiaHigh);
    const londonSweepedLow  = londonBars.some(b => b.low  < asiaLow);
    phase = (londonSweepedHigh || londonSweepedLow) ? 'manipulation' : 'accumulation';
  } else if (kz.name === 'NY_AM' || kz.name === 'SilverBullet') {
    phase = 'distribution';
  }

  return { phase, asiaHigh, asiaLow, asiaRange, kz: kz.name };
}

// ─── ICT: Daily Bias (Trader Zed method) ─────────────────────────────────────

/**
 * Daily Bias: compare current daily close to previous.
 * Bullish bias → only take BUY signals. Bearish → only SELL.
 * From "Daily Bias MADE EASY" by Trader Zed.
 */
export function detectDailyBias(barsDaily) {
  if (barsDaily.length < 2) return { bias: 'neutral' };
  const prev = barsDaily[barsDaily.length - 2];
  const curr = barsDaily[barsDaily.length - 1];
  if (curr.close > prev.close) return { bias: 'bullish', currClose: curr.close, prevClose: prev.close };
  if (curr.close < prev.close) return { bias: 'bearish', currClose: curr.close, prevClose: prev.close };
  return { bias: 'neutral', currClose: curr.close, prevClose: prev.close };
}

// ─── ICT: Weekly Template (Wednesday Inflection) ──────────────────────────────

/**
 * ICT Weekly Template: Mon-Tue form the Low of Week (bullish) or High of Week (bearish).
 * Wednesday is the inflection day — the true directional move starts there.
 * Only meaningful Wed–Fri (dow 3–5).
 */
export function detectWeeklyTemplate(barsDaily) {
  if (barsDaily.length < 5) return { template: 'none' };
  const dow = new Date().getUTCDay(); // 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri
  if (dow < 3 || dow > 5) return { template: 'none' };

  const weekBars = barsDaily.slice(-5);
  const weekHigh = Math.max(...weekBars.map(b => b.high));
  const weekLow  = Math.min(...weekBars.map(b => b.low));
  const weekMid  = (weekHigh + weekLow) / 2;

  // Mon-Tue = first 2 bars of the week slice
  const monTueBars = weekBars.slice(0, 2);
  const monTueHigh = Math.max(...monTueBars.map(b => b.high));
  const monTueLow  = Math.min(...monTueBars.map(b => b.low));

  // Mon-Tue formed the low → bullish Low of Week (price expected to rally)
  if (monTueLow === weekLow || monTueLow < weekMid) {
    return { template: 'low_of_week', bias: 'bullish', monTueHigh, monTueLow };
  }
  // Mon-Tue formed the high → bearish High of Week (price expected to fall)
  if (monTueHigh === weekHigh || monTueHigh > weekMid) {
    return { template: 'high_of_week', bias: 'bearish', monTueHigh, monTueLow };
  }
  return { template: 'none' };
}

// ─── Main Signal Engine ───────────────────────────────────────────────────────

/**
 * Full SMC + Pivot Boss + ICT signal engine.
 * Returns { direction, score, breakdown, entry, sl, tp1, tp2, rr, details }
 */
export function computeSignal({
  symbol, bars4H, bars1H, bars15M, barsDaily, barsWeekly, weights
}) {
  const score = { total: 0, breakdown: {} };
  const details = {};

  // Compute early — used for direction conditions and post-direction scoring
  const dailyBias      = detectDailyBias(barsDaily);
  const weeklyTemplate = detectWeeklyTemplate(barsDaily);
  details.dailyBias      = dailyBias;
  details.weeklyTemplate = weeklyTemplate;

  // ── 1. HTF Trend: Weekly + Daily ────────────────────────────────────────
  const weeklyTrend = detectTrend(barsWeekly, 3);
  const dailyTrend  = detectTrend(barsDaily, 4);
  const fourHTrend  = detectTrend(bars4H, 4);

  const dailyPivotBar = barsDaily[barsDaily.length - 2];
  const dailyPivots   = calcFloorPivots(dailyPivotBar.high, dailyPivotBar.low, dailyPivotBar.close);
  const dailyCam      = calcCamarilla(dailyPivotBar.high, dailyPivotBar.low, dailyPivotBar.close);
  const currentPrice  = bars15M[bars15M.length - 1].close;
  const pivotTrend    = getPivotTrend(currentPrice, dailyPivots);

  details.weeklyTrend = weeklyTrend;
  details.dailyTrend  = dailyTrend;
  details.fourHTrend  = fourHTrend;
  details.pivotTrend  = pivotTrend;
  details.dailyPivots = dailyPivots;

  const bullishHTF = weeklyTrend === 'bullish' &&
    (dailyTrend === 'bullish' || pivotTrend === 'bullish' || pivotTrend === 'bullish_breakout') &&
    fourHTrend !== 'bearish';
  const bearishHTF = weeklyTrend === 'bearish' &&
    (dailyTrend === 'bearish' || pivotTrend === 'bearish') &&
    fourHTrend !== 'bullish';
  const htfScore   = (bullishHTF || bearishHTF) ? weights.htfAlignment : weights.htfAlignment * 0.4;
  score.total += htfScore;
  score.breakdown.htfAlignment = htfScore;

  // ── 2. 4H Zone: OBs, FVGs, Breakers ────────────────────────────────────
  const obs4H      = findOrderBlocks(bars4H, 4);
  const fvgs4H     = findFVGs(bars4H, 0.0005);
  const breakers4H = findBreakerBlocks(bars4H, 4);
  const pd4H       = getPremiumDiscount(bars4H, 30);

  const priceTolerance = currentPrice * 0.005;
  const nearBullishOB  = obs4H.find(ob => ob.type === 'bullish' && currentPrice <= ob.high + priceTolerance && currentPrice >= ob.low - priceTolerance);
  const nearBearishOB  = obs4H.find(ob => ob.type === 'bearish' && currentPrice >= ob.low - priceTolerance && currentPrice <= ob.high + priceTolerance);
  const nearBullishFVG = fvgs4H.find(f => f.type === 'bullish' && currentPrice >= f.low - priceTolerance && currentPrice <= f.high + priceTolerance);
  const nearBearishFVG = fvgs4H.find(f => f.type === 'bearish' && currentPrice <= f.high + priceTolerance && currentPrice >= f.low - priceTolerance);
  const nearBullishBreaker = breakers4H.find(b => b.type === 'bullish_breaker');
  const nearBearishBreaker = breakers4H.find(b => b.type === 'bearish_breaker');

  // ICT: prefer ICT-confirmed OBs (with displacement candle) for full score
  const hasBullishZone = nearBullishOB || nearBullishFVG || nearBullishBreaker;
  const hasBearishZone = nearBearishOB || nearBearishFVG || nearBearishBreaker;
  const hasICTZone = (hasBullishZone && (nearBullishOB?.isICT || nearBullishBreaker)) ||
                     (hasBearishZone && (nearBearishOB?.isICT || nearBearishBreaker));

  // ICT Bible: do NOT want price to violate the 50% Mean Threshold (MTH) of an OB
  // If price is below MTH of bullish OB, or above MTH of bearish OB, the OB is weaker
  const bullishMTHValid = !nearBullishOB || currentPrice >= (nearBullishOB.high + nearBullishOB.low) / 2;
  const bearishMTHValid = !nearBearishOB || currentPrice <= (nearBearishOB.high + nearBearishOB.low) / 2;
  const obMTHValid = bullishMTHValid || bearishMTHValid;

  details.pd4H = pd4H;
  details.nearBullishOB = nearBullishOB; details.nearBearishOB = nearBearishOB;
  details.nearBullishFVG = nearBullishFVG; details.nearBearishFVG = nearBearishFVG;
  details.nearBullishBreaker = nearBullishBreaker; details.nearBearishBreaker = nearBearishBreaker;
  details.obMTHValid = { bullish: bullishMTHValid, bearish: bearishMTHValid };

  // Full score for ICT OB/Breaker with valid MTH; 75% for regular OB/FVG; 50% if MTH violated
  const obFVGScore = hasICTZone && obMTHValid ? weights.orderBlockFVG
    : (hasBullishZone || hasBearishZone) ? weights.orderBlockFVG * (obMTHValid ? 0.75 : 0.5) : 0;
  score.total += obFVGScore;
  score.breakdown.orderBlockFVG = obFVGScore;

  // ── 3. Liquidity Sweep + ICT MSS ────────────────────────────────────────
  const sweep15M = detectLiquiditySweep(bars15M, 12);
  const mss15M   = detectMSS(bars15M, 20);

  details.sweep15M = sweep15M;
  details.mss15M   = mss15M;

  // ICT: full score for SSL/BSL sweep, 80% for generic swing sweep
  // MSS is better quality than a sweep alone → bonus applied later
  const sweepMultiplier = sweep15M.detected
    ? (sweep15M.isSSL || sweep15M.isBSL ? 1.0 : 0.8) : 0;
  const sweepScore = sweepMultiplier * weights.liquiditySweep;
  score.total += sweepScore;
  score.breakdown.liquiditySweep = sweepScore;

  // ── 4. 1H + 15M Structure (BOS / MSS) ───────────────────────────────────
  const bos1H = detectBOS(bars1H, 5);
  details.bos1H = bos1H;

  // ICT: MSS on 15M (sweep → structure break) scores higher than plain BOS on 1H
  const structureScore = mss15M.detected ? weights.structureConfirm
    : bos1H.detected ? weights.structureConfirm * 0.75 : 0;
  score.total += structureScore;
  score.breakdown.structureConfirm = structureScore;

  // ── 5. Floor Pivot + Camarilla Confluence ────────────────────────────────
  const pivotMatches = nearPivot(currentPrice, dailyPivots, 0.003);
  const camMatches   = nearCamarilla(currentPrice, dailyCam, 0.003);
  const hasPivotConfl = pivotMatches.length > 0 || camMatches.length > 0;
  details.pivotConfl = { pivotMatches, camMatches };

  const pivotScore = hasPivotConfl ? weights.pivotConfluence : 0;
  score.total += pivotScore;
  score.breakdown.pivotConfluence = pivotScore;

  // ── 6. ICT: Killzone Timing ──────────────────────────────────────────────
  const kz = getCurrentKillzone();
  details.killzone = kz;

  // Killzone scoring: Silver Bullet > London/NY_AM > London_Close > Asia
  // London Close is reversal/profit-taking — lower priority for new entries
  const kzBase = weights.ictKillzone || 1.0;
  const kzBonus = kz.isSilverBullet                              ? kzBase
    : kz.name === 'NY_AM' || kz.name === 'London'               ? kzBase * 0.8
    : kz.name === 'Asia'                                         ? kzBase * 0.5
    : kz.name === 'London_Close'                                 ? kzBase * 0.3
    : kz.name                                                    ? kzBase * 0.5 : 0;

  // ICT Macro bonus: 20-min windows where IPDA specifically seeks liquidity
  const macroBonus = kz.isMacro ? 0.25 : 0;

  score.total += kzBonus + macroBonus;
  score.breakdown.ictKillzone = kzBonus;
  if (macroBonus) score.breakdown.ictMacro = macroBonus;

  // ── 7. ICT: OTE Fibonacci Level ──────────────────────────────────────────
  // Determine tentative direction from structure before checking OTE
  const tentativeDir = (bullishHTF && (hasBullishZone || (sweep15M.detected && sweep15M.direction === 'bullish'))) ? 'bullish'
    : (bearishHTF && (hasBearishZone || (sweep15M.detected && sweep15M.direction === 'bearish'))) ? 'bearish'
    : null;

  let ote = { inOTE: false };
  if (tentativeDir) {
    ote = checkOTE(bars15M, tentativeDir);
  }
  details.ote = ote;

  const oteBonus = ote.inOTE ? (weights.ictOTE || 0.75) : 0;
  score.total += oteBonus;
  score.breakdown.ictOTE = oteBonus;

  // ── 8. ICT: NDOG Magnetism ───────────────────────────────────────────────
  const ndog = findNDOG(barsDaily, bars15M);
  details.ndog = ndog;

  // Small bonus if price is near the NDOG midpoint (filling the gap)
  let ndogBonus = 0;
  if (ndog.detected && !ndog.filledToday) {
    const distToMid = Math.abs(currentPrice - ndog.gapMid) / currentPrice;
    if (distToMid < 0.003) ndogBonus = (weights.ictNDOG || 0.5);
  }
  score.total += ndogBonus;
  score.breakdown.ictNDOG = ndogBonus;

  // ── 9. ICT: Power of 3 Phase ─────────────────────────────────────────────
  const po3 = detectPowerOf3(bars15M);
  details.po3 = po3;
  // Distribution phase = NY session = ideal entry time
  const po3Bonus = po3.phase === 'distribution' ? 0.25
    : po3.phase === 'manipulation' ? 0.15 : 0;
  score.total += po3Bonus;
  score.breakdown.po3 = po3Bonus;

  // ── 10. Direction Determination ──────────────────────────────────────────
  let direction = 'NONE';

  const longConditions = [
    bullishHTF,
    fourHTrend === 'bullish',
    pd4H.zone === 'discount',
    !!hasBullishZone,
    sweep15M.detected && sweep15M.direction === 'bullish',
    mss15M.detected && mss15M.direction === 'bullish',
    !bos1H.detected || bos1H.direction === 'bullish',
    ote.inOTE && tentativeDir === 'bullish',
    dailyBias.bias === 'bullish',
    weeklyTemplate.bias === 'bullish',
  ];
  const shortConditions = [
    bearishHTF,
    fourHTrend === 'bearish',
    pd4H.zone === 'premium',
    !!hasBearishZone,
    sweep15M.detected && sweep15M.direction === 'bearish',
    mss15M.detected && mss15M.direction === 'bearish',
    !bos1H.detected || bos1H.direction === 'bearish',
    ote.inOTE && tentativeDir === 'bearish',
    dailyBias.bias === 'bearish',
    weeklyTemplate.bias === 'bearish',
  ];

  const longScore  = longConditions.filter(Boolean).length;
  const shortScore = shortConditions.filter(Boolean).length;
  details.directionScores = { longScore, shortScore };

  if (longScore >= 4 && score.total >= 4.5) direction = 'BUY';
  else if (shortScore >= 4 && score.total >= 4.5) direction = 'SELL';

  // Post-direction: score bonuses for alignment (only count when they agree with final direction)
  if (direction !== 'NONE') {
    if ((direction === 'BUY' && dailyBias.bias === 'bullish') ||
        (direction === 'SELL' && dailyBias.bias === 'bearish')) {
      const dbScore = weights.dailyBias || 0.5;
      score.total += dbScore;
      score.breakdown.dailyBias = dbScore;
    }
    if ((direction === 'BUY' && fourHTrend === 'bullish') ||
        (direction === 'SELL' && fourHTrend === 'bearish')) {
      const fhScore = weights.fourHBias || 1.0;
      score.total += fhScore;
      score.breakdown.fourHBias = fhScore;
    }
    if ((direction === 'BUY' && weeklyTemplate.bias === 'bullish') ||
        (direction === 'SELL' && weeklyTemplate.bias === 'bearish')) {
      const wtScore = weights.weeklyTemplate || 0.5;
      score.total += wtScore;
      score.breakdown.weeklyTemplate = wtScore;
    }
  }

  // ── 10b. Research confluence layer: CHOCH/MSS + Fib/OTE + multi-scale structure
  details.additionalConfluence = deriveConfluenceFeatures({
    bars15M,
    bars1H,
    bars4H,
    direction,
    currentPrice,
    maxBonus: weights.additionalConfluence || 0.75,
  });
  if (details.additionalConfluence.confluenceBonus > 0) {
    score.total += details.additionalConfluence.confluenceBonus;
    score.breakdown.additionalConfluence = details.additionalConfluence.confluenceBonus;
  }

  // ── 11. SL / TP Calculation ───────────────────────────────────────────────
  const atr15M = calcATR(bars15M, 14);
  let entry, sl, tp1, tp2, rr;

  function actualRR(direction, entry, sl, target) {
    const risk = Math.abs(Number(entry) - Number(sl));
    const reward = direction === 'BUY'
      ? Number(target) - Number(entry)
      : Number(entry) - Number(target);
    return risk > 0 && reward > 0 ? reward / risk : 0;
  }

  if (direction === 'BUY') {
    entry = currentPrice;
    // Prefer OTE level or OB/FVG zone for SL anchor
    const obZone = nearBullishOB || nearBullishFVG || nearBullishBreaker;
    const slBase = ote.inOTE ? ote.oteLow
      : obZone ? Math.min(obZone.low, sweep15M.detected ? sweep15M.sweptLevel * 0.999 : obZone.low)
      : currentPrice - atr15M * 1.5;
    sl = slBase - atr15M * 0.5;
    const risk = entry - sl;
    tp1 = entry + risk * 2.5;
    tp2 = entry + risk * 4.0;
    if (dailyPivots.R1 > entry && dailyPivots.R1 < tp2) tp1 = dailyPivots.R1;
    if (dailyPivots.R2 > tp1) tp2 = dailyPivots.R2;
    // ICT Fib extensions (1.272/1.618) of manipulation swing → TP targets
    if (sweep15M.detected && mss15M.detected && mss15M.direction === 'bullish' &&
        sweep15M.sweptLevel < mss15M.mssLevel) {
      const range = mss15M.mssLevel - sweep15M.sweptLevel;
      const fib127 = mss15M.mssLevel + range * 0.272;
      const fib168 = mss15M.mssLevel + range * 0.618;
      details.fibExtTPs = { fib127, fib168 };
      if (fib168 > tp1) tp2 = fib168;
    }
  } else if (direction === 'SELL') {
    entry = currentPrice;
    const obZone = nearBearishOB || nearBearishFVG || nearBearishBreaker;
    const slBase = ote.inOTE ? ote.oteHigh
      : obZone ? Math.max(obZone.high, sweep15M.detected ? sweep15M.sweptLevel * 1.001 : obZone.high)
      : currentPrice + atr15M * 1.5;
    sl = slBase + atr15M * 0.5;
    const risk = sl - entry;
    tp1 = entry - risk * 2.5;
    tp2 = entry - risk * 4.0;
    if (dailyPivots.S1 < entry && dailyPivots.S1 > tp2) tp1 = dailyPivots.S1;
    if (dailyPivots.S2 < tp1) tp2 = dailyPivots.S2;
    // ICT Fib extensions for bearish
    if (sweep15M.detected && mss15M.detected && mss15M.direction === 'bearish' &&
        sweep15M.sweptLevel > mss15M.mssLevel) {
      const range = sweep15M.sweptLevel - mss15M.mssLevel;
      const fib127 = mss15M.mssLevel - range * 0.272;
      const fib168 = mss15M.mssLevel - range * 0.618;
      details.fibExtTPs = { fib127, fib168 };
      if (fib168 < tp1) tp2 = fib168;
    }
  }

  // ── 12. RR Filter + Bonus ─────────────────────────────────────────────────
  rr = direction === 'NONE' ? 0 : actualRR(direction, entry, sl, tp1);
  details.riskReward = {
    actualRR: rr,
    risk: Math.abs(Number(entry) - Number(sl)),
    reward: direction === 'BUY' ? Number(tp1) - Number(entry) : Number(entry) - Number(tp1),
  };
  if (rr < 2.5) direction = 'NONE';
  if (rr >= 3) {
    score.total += weights.rrBonus;
    score.breakdown.rrBonus = weights.rrBonus;
  }

  return {
    symbol, direction,
    score: Math.round(score.total * 10) / 10,
    maxScore: Object.values(weights).reduce((a, b) => a + b, 0),
    breakdown: score.breakdown,
    entry, sl, tp1, tp2,
    rr: rr ? Math.round(rr * 10) / 10 : null,
    atr: atr15M,
    details,
  };
}
