# LiveTrans

Near-live speech-to-speech translation with a continuous browser audio stream.

```text
Microphone (PCM16/16 kHz)
  -> OmniVAD utterance endpointing
  -> Qwen3-ASR-0.6B on vLLM
  -> Hy-MT2-1.8B on vLLM
  -> OmniVoice
  -> ordered browser audio segments
```

The UI runs on port `8007`. ASR, translation, and TTS are separate persistent
services, so model weights are loaded once and kept on the GPU.

## Features

- Start/stop continuous microphone streaming; VAD cuts utterances without
  closing the session.
- Token-level ASR dictation using vLLM transcription SSE deltas.
- Stable draft/final translation feed.
- Sentence/utterance-level TTS queue with TTFT, TTFA, synthesis, and queue
  latency shown in the UI.
- OmniVoice with a cached 4.68-second English voice-clone prompt and 16
  decoding steps.
- Input/output audio previews for debugging.
- Mock mode for opening the UI without GPU models.

## Tested environment

- Ubuntu Linux
- Python 3.12
- NVIDIA RTX 3090 (24 GB)
- CUDA-capable PyTorch
- `vllm 0.25.1`
- `omnivoice 0.2.1`
- `omnivad 0.2.13`

The default GPU allocation is designed to colocate all three model services on
one 24 GB GPU. Smaller GPUs require lower vLLM memory limits or separate GPUs.

## Installation

Install system tools:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tmux curl git python3.12-venv
```

Create a virtual environment:

```bash
git clone https://github.com/anhtunguyen98/livetrans.git
cd livetrans

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

`vllm`, PyTorch, and CUDA wheels must be compatible with the host CUDA driver.
If the pinned AI requirements do not match the machine, install an appropriate
PyTorch/vLLM build first, then install `omnivoice`, `omnivad`, and `soundfile`.

## Configuration

```bash
cp .env.example .env
```

The launcher scripts automatically load `.env`. Important defaults:

```dotenv
LIVETRANS_MODE=real
LIVETRANS_GPU_ID=0
LIVETRANS_ASR_GPU_MEMORY=0.18
LIVETRANS_MT_GPU_MEMORY=0.42
LIVETRANS_TTS_NUM_STEP=16
LIVETRANS_TTS_DTYPE=float32
```

Default ports:

| Service | Port | Runtime |
| --- | ---: | --- |
| Qwen3-ASR | 8101 | vLLM |
| Hy-MT2 | 8102 | vLLM |
| OmniVoice | 8103 | FastAPI/PyTorch |
| UI/API | 8007 | FastAPI |

The default voice reference is stored at
`assets/voices/default_en.wav`, with its exact transcript in
`assets/voices/default_en.txt`. OmniVoice encodes this reference once during
service startup and reuses the cached prompt for every request.

## Run the application

Activate the environment and start all services:

```bash
source .venv/bin/activate
./scripts/start-all.sh
```

Open <http://localhost:8007>.

Start components separately when debugging:

```bash
./scripts/start-ai-services.sh
./scripts/start-ui.sh
```

Inspect status and logs:

```bash
./scripts/status.sh

tmux attach -t livetrans-asr
tmux attach -t livetrans-mt
tmux attach -t livetrans-tts
tmux attach -t livetrans-ui
```

Stop everything:

```bash
./scripts/stop.sh
```

Logs are written under `logs/` and are excluded from Git.

## Mock UI

To open the interface without loading models:

```bash
source .venv/bin/activate
LIVETRANS_MODE=mock python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8007
```

## Live session behavior

1. Clicking **Start stream** opens one browser WebSocket and continuously sends
   mono PCM16 audio at 16 kHz.
2. OmniVAD identifies speech boundaries. RMS/peak values are displayed only as
   input diagnostics and never decide whether ASR runs.
3. While an utterance is active, Qwen3-ASR is called on rolling windows. vLLM
   SSE transcription deltas are forwarded to the UI word by word.
4. At a VAD endpoint, final ASR and MT output is committed. The microphone and
   WebSocket remain open for the next utterance.
5. Translated text is queued to the persistent OmniVoice service and returned
   as ordered WAV segments.
6. Only clicking **Stop stream** closes capture and drains the remaining TTS
   queue.

Current OmniVAD defaults:

```dotenv
LIVETRANS_VAD_THRESHOLD=0.8
LIVETRANS_VAD_MIN_SILENCE_FRAMES=60
```

## HTTP and WebSocket API

- `GET /api/status`
- `GET /api/vllm/health`
- `POST /api/translate` — multipart file translation
- `WS /api/live` — continuous PCM16 live session

File translation example:

```bash
curl -X POST http://localhost:8007/api/translate \
  -F audio=@sample.wav \
  -F source_language=vi \
  -F target_language=en \
  -F speed=1.0
```

Uploaded audio is limited to the first 600 seconds by default. Change
`LIVETRANS_MAX_FILE_SECONDS` in `.env` to use a different limit.

## Troubleshooting

- **Microphone unavailable:** browsers require HTTPS unless the UI is opened
  from `localhost`.
- **A model port is already open:** run `./scripts/status.sh`, then stop the old
  tmux session or use `./scripts/stop.sh`.
- **CUDA out of memory:** reduce `LIVETRANS_ASR_GPU_MEMORY` and
  `LIVETRANS_MT_GPU_MEMORY`, or place services on separate GPUs.
- **No automatic playback:** browser autoplay policy may require clicking the
  output play button once.
- **Inspect captured audio:** use the input preview/download controls in the UI
  before changing ASR or VAD settings.

## Acknowledgements

The interface is adapted from the interaction and visual direction of
[X-Translator](https://github.com/zhaoyx239/X-Translator) (MIT). Model and
runtime projects retain their respective licenses: Qwen3-ASR, Hy-MT2,
OmniVoice, OmniVAD, and vLLM.
