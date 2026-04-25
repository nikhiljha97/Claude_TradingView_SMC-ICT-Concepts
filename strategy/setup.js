#!/usr/bin/env node
/**
 * One-time setup: discovers your Telegram chat_id by polling for the next message.
 * Run this once, then send any message (e.g. "/start") to your bot from Telegram.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { tgRequest, sendMessage } from './telegram.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.join(__dirname, 'config.json');

async function main() {
  const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  const token = config.telegram.token;

  console.log('Fetching bot info...');
  const me = await tgRequest(token, 'getMe');
  console.log(`Bot connected: @${me.username} (${me.first_name})\n`);

  if (config.telegram.chat_id) {
    console.log(`✓ chat_id already configured: ${config.telegram.chat_id}`);
    console.log('Sending test message...');
    await sendMessage(token, config.telegram.chat_id,
      '🤖 *Trading Scanner connected!*\n\nYou will receive 15M signal alerts here.\n\n' +
      '*Reply commands:*\n' +
      '`TP HIT <id>` — log a winning trade\n' +
      '`SL HIT <id>` — log a losing trade\n' +
      '`/stats` — performance breakdown\n' +
      '`/trades` — recent alerts');
    console.log('✓ Test message sent. Setup complete.');
    return;
  }

  console.log('Open Telegram and send any message to your bot now (e.g. "/start").');
  console.log('Polling for incoming messages...\n');

  for (let attempt = 0; attempt < 60; attempt++) {
    const updates = await tgRequest(token, 'getUpdates', { offset: 0, limit: 1, timeout: 5 });
    if (updates.length > 0) {
      const chatId = updates[updates.length - 1].message?.chat?.id;
      if (chatId) {
        config.telegram.chat_id = chatId;
        config.telegram.last_update_id = updates[updates.length - 1].update_id;
        fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
        console.log(`✓ Captured chat_id: ${chatId}`);
        await sendMessage(token, chatId,
          '🤖 *Trading Scanner connected!*\n\n' +
          'You will receive 15M alerts for: BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, XAUUSD\n\n' +
          '*Reply commands:*\n' +
          '`TP HIT <id>` / `SL HIT <id>` — log outcome\n' +
          '`/stats` — performance breakdown\n' +
          '`/trades` — recent alerts');
        console.log('✓ Setup complete!');
        return;
      }
    }
    process.stdout.write('.');
    await new Promise(r => setTimeout(r, 2000));
  }
  console.log('\n✗ Timed out — no message received. Try again.');
  process.exit(1);
}

main().catch(e => { console.error('Error:', e.message); process.exit(1); });
