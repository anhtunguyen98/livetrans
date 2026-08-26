#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SCRIPT_PATH")/common.sh"
require_command tmux
for session in "${LIVETRANS_UI_SESSION:-livetrans-ui}" "${LIVETRANS_TTS_SESSION:-livetrans-tts}" "${LIVETRANS_MT_SESSION:-livetrans-mt}" "${LIVETRANS_ASR_SESSION:-livetrans-asr}"; do
  if session_exists "$session"; then tmux kill-session -t "$session"; info "stopped $session"; else info "$session is not running"; fi
done
