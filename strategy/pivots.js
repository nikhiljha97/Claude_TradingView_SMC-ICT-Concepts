/**
 * Floor Pivot and Camarilla calculations.
 * Formulas from "Secrets of a Pivot Boss" by Franklin O. Ochoa.
 */

/**
 * Standard + Expanded Floor Pivots
 * Input: previous period's High, Low, Close
 */
export function calcFloorPivots(high, low, close) {
  const P  = (high + low + close) / 3;
  const BC = (high + low) / 2;
  const TC = (P - BC) + P;
  const R1 = 2 * P - low;
  const R2 = P + (high - low);
  const R3 = R1 + (high - low);
  const R4 = R3 + (R2 - R1);
  const S1 = 2 * P - high;
  const S2 = P - (high - low);
  const S3 = S1 - (high - low);
  const S4 = S3 - (S1 - S2);
  return { P, BC, TC, R1, R2, R3, R4, S1, S2, S3, S4 };
}

/**
 * Camarilla Equation levels
 * H3/L3 = reversal zones (trade back toward midpoint)
 * H4/L4 = breakout levels (strong trend continuation)
 */
export function calcCamarilla(high, low, close) {
  const diff = high - low;
  const H4 = close + diff * 1.1 / 2;
  const H3 = close + diff * 1.1 / 4;
  const H2 = close + diff * 1.1 / 6;
  const H1 = close + diff * 1.1 / 12;
  const L1 = close - diff * 1.1 / 12;
  const L2 = close - diff * 1.1 / 6;
  const L3 = close - diff * 1.1 / 4;
  const L4 = close - diff * 1.1 / 2;
  return { H4, H3, H2, H1, L1, L2, L3, L4 };
}

/**
 * Pivot Trend Analysis (Ochoa):
 * Bullish  = price above S1 → look for buys at P/S1
 * Bearish  = price below R1 → look for sells at R1/P
 * Ranging  = price between R1 and S1
 */
export function getPivotTrend(price, pivots) {
  if (price > pivots.R1) return 'bullish_breakout';
  if (price > pivots.S1) return 'bullish';
  if (price < pivots.S1) return 'bearish';
  return 'ranging';
}

/**
 * Check if price is near a pivot level (within pct%)
 */
export function nearPivot(price, pivots, pctThreshold = 0.003) {
  const levels = [
    { name: 'R4', val: pivots.R4 }, { name: 'R3', val: pivots.R3 },
    { name: 'R2', val: pivots.R2 }, { name: 'R1', val: pivots.R1 },
    { name: 'TC', val: pivots.TC }, { name: 'P',  val: pivots.P  },
    { name: 'BC', val: pivots.BC }, { name: 'S1', val: pivots.S1 },
    { name: 'S2', val: pivots.S2 }, { name: 'S3', val: pivots.S3 },
    { name: 'S4', val: pivots.S4 },
  ];
  const matches = [];
  for (const lvl of levels) {
    if (!lvl.val || isNaN(lvl.val)) continue;
    const dist = Math.abs(price - lvl.val) / price;
    if (dist <= pctThreshold) matches.push({ name: lvl.name, level: lvl.val, distPct: dist });
  }
  return matches.sort((a, b) => a.distPct - b.distPct);
}

/**
 * Near Camarilla H3/L3 (best reversal zones)
 */
export function nearCamarilla(price, cam, pctThreshold = 0.003) {
  const levels = [
    { name: 'H3', val: cam.H3 }, { name: 'H4', val: cam.H4 },
    { name: 'L3', val: cam.L3 }, { name: 'L4', val: cam.L4 },
  ];
  const matches = [];
  for (const lvl of levels) {
    if (!lvl.val || isNaN(lvl.val)) continue;
    const dist = Math.abs(price - lvl.val) / price;
    if (dist <= pctThreshold) matches.push({ name: lvl.name, level: lvl.val, distPct: dist });
  }
  return matches.sort((a, b) => a.distPct - b.distPct);
}
