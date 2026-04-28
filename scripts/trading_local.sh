#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCANNER_PLIST="$HOME/Library/LaunchAgents/com.tradingview.scanner.plist"
KEEP_AWAKE_PLIST="$HOME/Library/LaunchAgents/com.tradingview.keepawake.plist"
DOMAIN="gui/$(id -u)"

find_node() {
  if command -v node >/dev/null 2>&1; then
    command -v node
  elif [ -x /usr/local/bin/node ]; then
    printf '%s\n' /usr/local/bin/node
  elif [ -x /opt/homebrew/bin/node ]; then
    printf '%s\n' /opt/homebrew/bin/node
  else
    printf 'node not found. Install Node.js first.\n' >&2
    exit 1
  fi
}

write_scanner_plist() {
  local node_bin="$1"
  mkdir -p "$HOME/Library/LaunchAgents"
  /usr/bin/python3 - "$SCANNER_PLIST" "$node_bin" "$REPO_ROOT" <<'PY'
import plistlib
import sys
from pathlib import Path

plist_path, node_bin, repo_root = sys.argv[1:4]
payload = {
    "Label": "com.tradingview.scanner",
    "ProgramArguments": [node_bin, f"{repo_root}/strategy/scanner.js"],
    "StartCalendarInterval": [
        {"Minute": 0},
        {"Minute": 15},
        {"Minute": 30},
        {"Minute": 45},
    ],
    "WorkingDirectory": f"{repo_root}/strategy",
    "StandardOutPath": f"{repo_root}/strategy/scanner.log",
    "StandardErrorPath": f"{repo_root}/strategy/scanner.error.log",
    "RunAtLoad": False,
    "EnvironmentVariables": {
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    },
}
Path(plist_path).write_bytes(plistlib.dumps(payload))
PY
}

write_keep_awake_plist() {
  mkdir -p "$HOME/Library/LaunchAgents"
  /usr/bin/python3 - "$KEEP_AWAKE_PLIST" "$REPO_ROOT" <<'PY'
import plistlib
import sys
from pathlib import Path

plist_path, repo_root = sys.argv[1:3]
payload = {
    "Label": "com.tradingview.keepawake",
    "ProgramArguments": ["/usr/bin/caffeinate", "-i"],
    "RunAtLoad": True,
    "KeepAlive": True,
    "StandardOutPath": f"{repo_root}/strategy/keepawake.log",
    "StandardErrorPath": f"{repo_root}/strategy/keepawake.error.log",
}
Path(plist_path).write_bytes(plistlib.dumps(payload))
PY
}

start_tradingview() {
  if pgrep -f "/Applications/TradingView.app/Contents/MacOS/TradingView.*remote-debugging-port=9222" >/dev/null 2>&1; then
    printf 'TradingView is already running with CDP port 9222.\n'
    return
  fi
  "$REPO_ROOT/scripts/launch_tv_debug_mac.sh" >/dev/null 2>&1 &
  printf 'Started TradingView with debug port 9222.\n'
}

start_scanner() {
  local node_bin
  node_bin="$(find_node)"
  write_scanner_plist "$node_bin"
  launchctl bootout "$DOMAIN" "$SCANNER_PLIST" >/dev/null 2>&1 || true
  launchctl bootstrap "$DOMAIN" "$SCANNER_PLIST"
  launchctl enable "$DOMAIN/com.tradingview.scanner"
  launchctl kickstart -k "$DOMAIN/com.tradingview.scanner"
  printf 'Scanner loaded and kicked once. Future runs happen at :00/:15/:30/:45.\n'
}

stop_scanner() {
  launchctl bootout "$DOMAIN" "$SCANNER_PLIST" >/dev/null 2>&1 || true
  printf 'Scanner stopped.\n'
}

keep_awake_on() {
  write_keep_awake_plist
  launchctl bootout "$DOMAIN" "$KEEP_AWAKE_PLIST" >/dev/null 2>&1 || true
  launchctl bootstrap "$DOMAIN" "$KEEP_AWAKE_PLIST"
  launchctl enable "$DOMAIN/com.tradingview.keepawake"
  printf 'Keep-awake enabled. Display may turn off; the Mac should not idle-sleep while logged in.\n'
}

keep_awake_off() {
  launchctl bootout "$DOMAIN" "$KEEP_AWAKE_PLIST" >/dev/null 2>&1 || true
  rm -f "$KEEP_AWAKE_PLIST"
  printf 'Keep-awake disabled.\n'
}

status() {
  printf '%s\n' '--- scanner ---'
  launchctl print "$DOMAIN/com.tradingview.scanner" 2>/dev/null | sed -n '1,70p' || printf 'scanner is not loaded\n'
  printf '%s\n' '--- keep awake ---'
  launchctl print "$DOMAIN/com.tradingview.keepawake" 2>/dev/null | sed -n '1,35p' || printf 'keep-awake is not loaded\n'
  printf '%s\n' '--- TradingView ---'
  pgrep -fl "/Applications/TradingView.app/Contents/MacOS/TradingView.*remote-debugging-port=9222" || printf 'TradingView debug process not found\n'
}

usage() {
  cat <<USAGE
Usage: scripts/trading_local.sh <command>

Commands:
  start          Start TradingView debug mode, load scanner, and run one scan now
  stop           Stop the 15-minute scanner
  restart        Stop then start the scanner
  run-once       Run one scanner cycle immediately
  status         Show scanner, keep-awake, and TradingView status
  logs           Follow scanner logs
  keep-awake-on  Prevent true system sleep while allowing display sleep
  keep-awake-off Remove the keep-awake agent

USAGE
}

case "${1:-}" in
  start)
    start_tradingview
    start_scanner
    ;;
  stop)
    stop_scanner
    ;;
  restart)
    stop_scanner
    start_tradingview
    start_scanner
    ;;
  run-once)
    launchctl kickstart -k "$DOMAIN/com.tradingview.scanner"
    ;;
  status)
    status
    ;;
  logs)
    tail -f "$REPO_ROOT/strategy/scanner.log" "$REPO_ROOT/strategy/scanner.error.log"
    ;;
  keep-awake-on)
    keep_awake_on
    ;;
  keep-awake-off)
    keep_awake_off
    ;;
  *)
    usage
    exit 1
    ;;
esac
