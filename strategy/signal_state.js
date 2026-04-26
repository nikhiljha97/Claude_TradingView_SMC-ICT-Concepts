function directionToBias(direction) {
  return direction === 'BUY' ? 'bullish' : direction === 'SELL' ? 'bearish' : null;
}

function rounded(value, decimals = 4) {
  if (!Number.isFinite(Number(value))) return null;
  const factor = 10 ** decimals;
  return Math.round(Number(value) * factor) / factor;
}

export function setupFingerprint(signal) {
  const d = signal.details || {};
  const parts = [
    signal.symbol,
    signal.direction,
    d.weeklyTrend,
    d.dailyTrend,
    d.fourHTrend,
    d.pivotTrend,
    d.dailyBias?.bias,
    d.weeklyTemplate?.bias,
    d.sweep15M?.direction,
    rounded(d.sweep15M?.sweptLevel),
    d.mss15M?.direction,
    rounded(d.mss15M?.mssLevel),
    d.bos1H?.direction,
    d.pd4H?.zone,
    d.killzone?.name,
  ];
  return parts.map(p => p ?? '').join('|');
}

export function htfAlignment(signal) {
  const d = signal.details || {};
  const wanted = directionToBias(signal.direction);
  if (!wanted) return { aligned: false, count: 0, total: 0 };

  const checks = [
    d.weeklyTrend,
    d.dailyTrend,
    d.fourHTrend,
    d.dailyBias?.bias,
    d.weeklyTemplate?.bias,
  ].filter(Boolean).filter(v => v !== 'neutral' && v !== 'ranging');

  const count = checks.filter(v => v === wanted).length;
  const opposite = checks.filter(v => v !== wanted).length;
  return {
    aligned: count >= Math.max(2, opposite + 1),
    count,
    opposite,
    total: checks.length,
  };
}

export function materialMove(current, previous, atr, minAtr = 0.35) {
  if (!Number.isFinite(current) || !Number.isFinite(previous)) return false;
  if (!Number.isFinite(atr) || atr <= 0) return Math.abs(current - previous) / Math.max(1, Math.abs(previous)) >= 0.001;
  return Math.abs(current - previous) >= atr * minAtr;
}

function recentHours(trade) {
  return (Date.now() - new Date(trade.timestamp).getTime()) / 3600000;
}

export function shouldAlert(signal, trades, config = {}) {
  const rules = config.alertContinuity || {};
  const sameHours = Number(rules.sameDirectionMemoryHours ?? 4);
  const flipHours = Number(rules.oppositeDirectionCooldownHours ?? 2);
  const resendMinutes = Number(rules.resendSkippedAfterMinutes ?? 45);
  const minAtrRefresh = Number(rules.minAtrRefresh ?? 0.35);
  const flipScoreMargin = Number(rules.flipScoreMargin ?? 1.25);
  const flipMinHtfAligned = Number(rules.flipMinHtfAligned ?? 3);
  const nowPrice = Number(signal.entry);

  signal.setupFingerprint = setupFingerprint(signal);
  signal.htfAlignment = htfAlignment(signal);

  const symbolTrades = trades
    .filter(t => t.symbol === signal.symbol && t.direction && t.direction !== 'NONE')
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  const same = symbolTrades.find(t => t.direction === signal.direction && recentHours(t) <= sameHours && !t.outcome);
  if (same) {
    const ageMinutes = recentHours(same) * 60;
    const sameFingerprint = same.setupFingerprint && same.setupFingerprint === signal.setupFingerprint;
    const moved = materialMove(nowPrice, Number(same.entry), Number(signal.atr), minAtrRefresh);
    const improved = Number(signal.score) >= Number(same.score || 0) + Number(rules.scoreRefreshMargin ?? 0.5);
    if (same.confirmed === true) {
      return { allow: false, reason: 'same_direction_taken_cooldown', previousTradeId: same.tradeId };
    }
    if (sameFingerprint && ageMinutes < resendMinutes && !moved && !improved) {
      return { allow: false, reason: 'same_setup_not_refreshed', previousTradeId: same.tradeId };
    }
  }

  const opposite = symbolTrades.find(t => t.direction !== signal.direction && recentHours(t) <= flipHours && !t.outcome);
  if (opposite) {
    const align = signal.htfAlignment;
    const strongEnough = Number(signal.score) >= Number(opposite.score || 0) + flipScoreMargin;
    const htfEnough = align.count >= flipMinHtfAligned && align.aligned;
    if (!strongEnough || !htfEnough) {
      return {
        allow: false,
        reason: 'opposite_direction_needs_more_conviction',
        previousTradeId: opposite.tradeId,
        previousDirection: opposite.direction,
        requiredScore: rounded(Number(opposite.score || 0) + flipScoreMargin, 2),
        htfAligned: align.count,
      };
    }
  }

  return { allow: true, reason: 'passed_continuity' };
}
