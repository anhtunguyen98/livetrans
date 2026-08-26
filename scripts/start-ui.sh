#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SCRIPT_PATH")/common.sh"
require_command tmux; require_command python

UI_SESSION="${LIVETRANS_UI_SESSION:-livetrans-ui}"
UI_PORT="${LIVETRANS_UI_PORT:-8007}"
RUN_MODE="${LIVETRANS_MODE:-real}"
ASR_URL="${LIVETRANS_ASR_BASE_URL:-http://127.0.0.1:8101/v1}"
MT_URL="${LIVETRANS_MT_BASE_URL:-http://127.0.0.1:8102/v1}"
TTS_URL="${LIVETRANS_TTS_BASE_URL:-http://127.0.0.1:8103}"
API_KEY="${LIVETRANS_VLLM_API_KEY:-local}"

if port_open "$UI_PORT"; then
  info "UI port $UI_PORT is already open; no duplicate process started"
  info "Open: http://localhost:$UI_PORT"
  exit 0
fi

start_tmux "$UI_SESSION" "LIVETRANS_MODE='$RUN_MODE' LIVETRANS_ASR_BASE_URL='$ASR_URL' LIVETRANS_MT_BASE_URL='$MT_URL' LIVETRANS_TTS_BASE_URL='$TTS_URL' LIVETRANS_VLLM_API_KEY='$API_KEY' python -m uvicorn app.main:app --host 0.0.0.0 --port '$UI_PORT'" "$LOG_DIR/ui.log"
wait_http UI "http://127.0.0.1:${UI_PORT}/api/status" "" 10
info "Open: http://localhost:$UI_PORT"
info "Use: tmux attach -t $UI_SESSION"
