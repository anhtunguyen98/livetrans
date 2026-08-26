from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL_NAME = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.getenv("OMNIVOICE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "16"))
DTYPE_NAME = os.getenv("OMNIVOICE_DTYPE", "float32").lower()
VOICE_PROMPT_AUDIO = os.getenv("OMNIVOICE_PROMPT_AUDIO", "assets/voices/default_en.wav")
VOICE_PROMPT_TEXT = os.getenv(
    "OMNIVOICE_PROMPT_TEXT",
    "Hello, this is a clear and natural voice for real-time speech translation.",
)

model: Any = None
generation_config: Any = None
short_generation_config: Any = None
voice_clone_prompt: Any = None
generation_lock = asyncio.Lock()


def load_model() -> None:
    global model, generation_config, short_generation_config, voice_clone_prompt
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(DTYPE_NAME)
    if dtype is None:
        raise ValueError(f"Unsupported OMNIVOICE_DTYPE: {DTYPE_NAME}")
    model = OmniVoice.from_pretrained(MODEL_NAME, device_map=DEVICE, dtype=dtype)
    generation_config = OmniVoiceGenerationConfig(num_step=NUM_STEP)
    short_generation_config = OmniVoiceGenerationConfig(num_step=NUM_STEP)
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=VOICE_PROMPT_AUDIO,
        ref_text=VOICE_PROMPT_TEXT,
        preprocess_prompt=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(load_model)
    yield


app = FastAPI(title="LiveTrans OmniVoice", lifespan=lifespan)


class SpeechRequest(BaseModel):
    text: str
    speed: float = 1.0
    language: str | None = None


@app.get("/healthz")
def health() -> dict:
    if model is None:
        raise HTTPException(503, "OmniVoice is loading")
    return {"status": "ok", "model": MODEL_NAME, "sampling_rate": int(model.sampling_rate),
            "num_step": NUM_STEP, "dtype": DTYPE_NAME}


def generate_wav(text: str, speed: float, language: str | None) -> bytes:
    words = text.split()
    is_short = len(words) <= 3
    config = short_generation_config if is_short else generation_config
    spoken_text = text
    if is_short and text[-1:] not in ".!?。！？":
        spoken_text = text[0].upper() + text[1:] + "!"
    kwargs = {"text": spoken_text, "language": language, "speed": speed,
              "voice_clone_prompt": voice_clone_prompt,
              "generation_config": config}
    with torch.inference_mode():
        audios = model.generate(**kwargs)
    if not audios:
        raise RuntimeError("OmniVoice returned no audio")
    output = io.BytesIO()
    sf.write(output, np.asarray(audios[0]).squeeze(), int(model.sampling_rate), format="WAV", subtype="PCM_16")
    return output.getvalue()


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Text is empty")
    if not 0.7 <= request.speed <= 1.3:
        raise HTTPException(400, "Speed must be between 0.7 and 1.3")
    started = time.perf_counter()
    async with generation_lock:
        wav = await asyncio.to_thread(generate_wav, text, request.speed, request.language)
    return Response(
        wav,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(int(model.sampling_rate)), "X-Synthesis-Ms": str(round((time.perf_counter() - started) * 1000))},
    )
