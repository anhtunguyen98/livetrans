#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi
LOG_DIR="${LIVETRANS_LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "$LOG_DIR"

info() { printf '[livetrans] %s\n' "$*"; }
fail() { printf '[livetrans] ERROR: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"; }
session_exists() { tmux has-session -t "$1" 2>/dev/null; }
port_open() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"; }

start_tmux() {
  local session="$1" command="$2" log_file="$3"
  if session_exists "$session"; then info "$session is already running"; return 0; fi
  tmux new-session -d -s "$session" "cd '$PROJECT_DIR' && set -o pipefail && $command 2>&1 | tee -a '$log_file'"
  info "started tmux session: $session"
  info "log: $log_file"
}

wait_http() {
  local name="$1" url="$2" api_key="$3" wait_seconds="${4:-10}" elapsed=0
  while (( elapsed < wait_seconds )); do
    if curl -fsS -H "Authorization: Bearer $api_key" "$url" >/dev/null 2>&1; then
      info "$name is ready: $url"; return 0
    fi
    sleep 1; elapsed=$((elapsed + 1))
  done
  info "$name is still loading; follow its tmux log"
  return 1
}
