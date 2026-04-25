/**
 * Telegram Bot integration — uses Node's built-in https (no extra deps).
 *
 * Sends formatted trade alerts and polls for user feedback (TP HIT / SL HIT replies).
 * Bot token stored in config.json. Chat ID discovered on first user message via setup.js.
 */
import https from 'https';

export function tgRequest(token, method, params = {}) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(params);
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${token}/${method}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (!parsed.ok) reject(new Error(`Telegram API error: ${parsed.description || data}`));
          else resolve(parsed.result);
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

export async function sendMessage(token, chatId, text, options = {}) {
  return tgRequest(token, 'sendMessage', {
    chat_id: chatId,
    text,
    parse_mode: 'Markdown',
    disable_web_page_preview: true,
    disable_notification: false, // always notify even if chat is muted on device
    ...options,
  });
}

export async function getUpdates(token, offset = 0) {
  return tgRequest(token, 'getUpdates', {
    offset, limit: 100, timeout: 0,
  });
}

/**
 * Format a trade alert message in Markdown.
 */
export function formatAlert(signal) {
  const dir = signal.direction === 'BUY' ? '🟢 BUY' : '🔴 SELL';
  const breakdown = Object.entries(signal.breakdown)
    .map(([k, v]) => `  • ${k}: ${v}`)
    .join('\n');

  const fmt = (n) => n != null ? Number(n).toFixed(signal.symbol.includes('JPY') ? 3 : (signal.symbol === 'XAUUSD' ? 2 : 4)) : 'n/a';

  const reasoning = [];
  const d = signal.details;
  if (d.weeklyTrend) reasoning.push(`Weekly: ${d.weeklyTrend}`);
  if (d.dailyTrend)  reasoning.push(`Daily: ${d.dailyTrend}`);
  if (d.pivotTrend)  reasoning.push(`Pivot: ${d.pivotTrend}`);
  if (d.pd4H)        reasoning.push(`4H zone: ${d.pd4H.zone} (${d.pd4H.position}%)`);
  if (d.nearBullishOB) reasoning.push(`✓ 4H Bull OB at ${fmt(d.nearBullishOB.low)}-${fmt(d.nearBullishOB.high)}`);
  if (d.nearBearishOB) reasoning.push(`✓ 4H Bear OB at ${fmt(d.nearBearishOB.low)}-${fmt(d.nearBearishOB.high)}`);
  if (d.nearBullishFVG) reasoning.push(`✓ 4H Bull FVG at ${fmt(d.nearBullishFVG.low)}-${fmt(d.nearBullishFVG.high)}`);
  if (d.nearBearishFVG) reasoning.push(`✓ 4H Bear FVG at ${fmt(d.nearBearishFVG.low)}-${fmt(d.nearBearishFVG.high)}`);
  if (d.sweep15M?.detected) {
    const sweepTag = d.sweep15M.isSSL ? ' [SSL]' : d.sweep15M.isBSL ? ' [BSL]' : '';
    reasoning.push(`✓ Liquidity sweep${sweepTag} ${d.sweep15M.direction} @ ${fmt(d.sweep15M.sweptLevel)}`);
  }
  if (d.mss15M?.detected) reasoning.push(`✓ 15M MSS ${d.mss15M.direction} (sweep→structure)`);
  if (d.bos1H?.detected) reasoning.push(`✓ 1H BOS ${d.bos1H.direction} @ ${fmt(d.bos1H.breakLevel)}`);
  if (d.nearBullishBreaker) reasoning.push(`✓ 4H Bull Breaker Block ${fmt(d.nearBullishBreaker.low)}-${fmt(d.nearBullishBreaker.high)}`);
  if (d.nearBearishBreaker) reasoning.push(`✓ 4H Bear Breaker Block ${fmt(d.nearBearishBreaker.low)}-${fmt(d.nearBearishBreaker.high)}`);
  if (d.killzone?.name) {
    const kzName = d.killzone.name.replace(/_/g, ' ');
    const kzLabel = d.killzone.isSilverBullet ? `${kzName} 🥈 Silver Bullet` : kzName;
    const macroTag = d.killzone.isMacro ? ' [MACRO]' : '';
    reasoning.push(`✓ ICT Killzone: ${kzLabel}${macroTag}`);
  }
  if (d.ote?.inOTE) reasoning.push(`✓ OTE zone ${fmt(d.ote.oteLow)}-${fmt(d.ote.oteHigh)} (62-79% Fib)`);
  if (d.ndog?.detected && !d.ndog.filledToday) reasoning.push(`✓ NDOG gap ${fmt(d.ndog.gapLow)}-${fmt(d.ndog.gapHigh)} (mid ${fmt(d.ndog.gapMid)})`);
  if (d.po3?.phase === 'distribution') reasoning.push(`✓ PO3 distribution phase (NY session)`);
  if (d.po3?.phase === 'manipulation') reasoning.push(`✓ PO3 manipulation phase (London sweep)`);
  if (d.dailyBias?.bias && d.dailyBias.bias !== 'neutral') reasoning.push(`✓ Daily bias: ${d.dailyBias.bias} (prev close ${fmt(d.dailyBias.prevClose)} → ${fmt(d.dailyBias.currClose)})`);
  if (d.weeklyTemplate?.template !== 'none' && d.weeklyTemplate?.template) reasoning.push(`✓ Weekly template: ${d.weeklyTemplate.template.replace(/_/g, ' ')} (${d.weeklyTemplate.bias})`);
  if (d.pivotConfl?.pivotMatches?.length) {
    const m = d.pivotConfl.pivotMatches[0];
    reasoning.push(`✓ Pivot ${m.name} @ ${fmt(m.level)}`);
  }
  if (d.pivotConfl?.camMatches?.length) {
    const m = d.pivotConfl.camMatches[0];
    reasoning.push(`✓ Camarilla ${m.name} @ ${fmt(m.level)}`);
  }

  const fibLines = [];
  if (signal.details?.fibExtTPs) {
    const { fib127, fib168 } = signal.details.fibExtTPs;
    fibLines.push(`*Fib 1.272:* \`${fmt(fib127)}\``);
    fibLines.push(`*Fib 1.618:* \`${fmt(fib168)}\``);
  }

  return [
    `${dir} *${signal.symbol}*  (15M)`,
    ``,
    `*Entry:* \`${fmt(signal.entry)}\``,
    `*Stop:*  \`${fmt(signal.sl)}\``,
    `*TP1:*   \`${fmt(signal.tp1)}\`  _(1:${signal.rr})_`,
    `*TP2:*   \`${fmt(signal.tp2)}\``,
    ...(fibLines.length ? fibLines : []),
    ``,
    `*Confidence:* ${signal.score}/${signal.maxScore.toFixed(1)}`,
    ``,
    `*Setup reasoning:*`,
    reasoning.map(r => `  ${r}`).join('\n'),
    ``,
    `*Score breakdown:*`,
    breakdown,
    ``,
    `_Did you take this trade?_`,
    `\`YES ${signal.tradeId}\` _→ activates cooldown (won't re-alert same setup)_`,
    `\`NO ${signal.tradeId}\` _→ skipped (will re-alert if setup holds)_`,
    `_Once in trade:_ \`TP HIT ${signal.tradeId}\` _or_ \`SL HIT ${signal.tradeId}\``,
  ].join('\n');
}

/**
 * Parse a feedback message: "TP HIT abc123" or "SL HIT xyz789".
 * Returns { outcome: 'TP'|'SL', tradeId } or null.
 */
export function parseFeedback(text) {
  if (!text) return null;
  const m = text.trim().match(/^(TP|SL)\s+HIT\s+([a-z0-9]+)/i);
  if (!m) return null;
  return { outcome: m[1].toUpperCase(), tradeId: m[2].toLowerCase() };
}
