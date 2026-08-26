#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SCRIPT_PATH")/common.sh"
require_command tmux
printf '%-18s %-10s %-8s\n' SERVICE TMUX PORT
for item in "livetrans-asr:8101" "livetrans-mt:8102" "livetrans-tts:8103" "livetrans-ui:8007"; do
  session="${item%%:*}"; port="${item##*:}"
  if session_exists "$session"; then tmux_state=running; else tmux_state=stopped; fi
  if port_open "$port"; then port_state=open; else port_state=closed; fi
  printf '%-18s %-10s %-8s\n' "$session" "$tmux_state" "$port_state"
done
printf '\nGPU:\n'; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
printf '\nUI/vLLM health:\n'; curl -fsS http://127.0.0.1:8007/api/vllm/health 2>/dev/null || printf 'UI unavailable'
printf '\n'
