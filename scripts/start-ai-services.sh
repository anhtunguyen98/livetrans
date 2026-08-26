#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SCRIPT_PATH")/common.sh"
require_command tmux; require_command vllm; require_command curl

ASR_SESSION="${LIVETRANS_ASR_SESSION:-livetrans-asr}"
MT_SESSION="${LIVETRANS_MT_SESSION:-livetrans-mt}"
TTS_SESSION="${LIVETRANS_TTS_SESSION:-livetrans-tts}"
ASR_MODEL="${LIVETRANS_ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}"
MT_MODEL="${LIVETRANS_MT_MODEL:-tencent/Hy-MT2-1.8B}"
ASR_PORT="${LIVETRANS_ASR_PORT:-8101}"
MT_PORT="${LIVETRANS_MT_PORT:-8102}"
TTS_PORT="${LIVETRANS_TTS_PORT:-8103}"
API_KEY="${LIVETRANS_VLLM_API_KEY:-local}"
GPU_ID="${LIVETRANS_GPU_ID:-0}"
ASR_GPU_MEMORY="${LIVETRANS_ASR_GPU_MEMORY:-0.18}"
MT_GPU_MEMORY="${LIVETRANS_MT_GPU_MEMORY:-0.42}"
WAIT_SECONDS="${LIVETRANS_START_WAIT_SECONDS:-180}"

# Start sequentially. Starting both engines together can OOM a single GPU while
# torch.compile / CUDA graph capture temporarily uses much more than steady state.
if ! port_open "$MT_PORT"; then
  start_tmux "$MT_SESSION" "CUDA_VISIBLE_DEVICES='$GPU_ID' vllm serve '$MT_MODEL' --host 0.0.0.0 --port '$MT_PORT' --served-model-name '$MT_MODEL' --trust-remote-code --enforce-eager --gpu-memory-utilization '$MT_GPU_MEMORY' --max-model-len 8192 --api-key '$API_KEY'" "$LOG_DIR/mt.log"
else info "MT port $MT_PORT is already open"; fi
wait_http MT "http://127.0.0.1:${MT_PORT}/v1/models" "$API_KEY" "$WAIT_SECONDS"

if ! port_open "$ASR_PORT"; then
  start_tmux "$ASR_SESSION" "CUDA_VISIBLE_DEVICES='$GPU_ID' vllm serve '$ASR_MODEL' --host 0.0.0.0 --port '$ASR_PORT' --served-model-name '$ASR_MODEL' --trust-remote-code --enforce-eager --gpu-memory-utilization '$ASR_GPU_MEMORY' --max-model-len 4096 --api-key '$API_KEY'" "$LOG_DIR/asr.log"
else info "ASR port $ASR_PORT is already open"; fi
wait_http ASR "http://127.0.0.1:${ASR_PORT}/v1/models" "$API_KEY" "$WAIT_SECONDS"

if ! port_open "$TTS_PORT"; then
  start_tmux "$TTS_SESSION" "CUDA_VISIBLE_DEVICES='$GPU_ID' OMNIVOICE_DEVICE='cuda:0' OMNIVOICE_MODEL='${LIVETRANS_TTS_MODEL:-k2-fsa/OmniVoice}' OMNIVOICE_NUM_STEP='${LIVETRANS_TTS_NUM_STEP:-16}' OMNIVOICE_DTYPE='${LIVETRANS_TTS_DTYPE:-float32}' python -m uvicorn app.tts_service:app --host 0.0.0.0 --port '$TTS_PORT'" "$LOG_DIR/tts.log"
else info "TTS port $TTS_PORT is already open"; fi
wait_http TTS "http://127.0.0.1:${TTS_PORT}/healthz" "" "$WAIT_SECONDS"

info "Use: tmux attach -t $ASR_SESSION, $MT_SESSION, or $TTS_SESSION"
