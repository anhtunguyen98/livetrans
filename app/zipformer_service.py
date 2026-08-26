from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("LIVETRANS_ASR_MODEL", "hynt/Zipformer-30M-RNNT-6000h")
MODEL_DIR = os.getenv("LIVETRANS_ZIPFORMER_MODEL_DIR", "")
NUM_THREADS = int(os.getenv("LIVETRANS_ZIPFORMER_THREADS", "4"))
USE_INT8 = os.getenv("LIVETRANS_ZIPFORMER_INT8", "1").lower() not in {"0", "false", "no"}


class ZipformerASR:
    def __init__(self) -> None:
        import sherpa_onnx

        if MODEL_DIR:
            model_dir = Path(MODEL_DIR).expanduser().resolve()
        else:
            model_dir = Path(snapshot_download(
                MODEL_ID,
                allow_patterns=["*.onnx", "config.json", "bpe.model"],
            ))
        suffix = ".int8" if USE_INT8 else ""
        self.model_dir = model_dir
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(model_dir / "config.json"),
            encoder=str(model_dir / f"encoder-epoch-20-avg-10{suffix}.onnx"),
            decoder=str(model_dir / "decoder-epoch-20-avg-10.onnx"),
            joiner=str(model_dir / f"joiner-epoch-20-avg-10{suffix}.onnx"),
            num_threads=NUM_THREADS,
            sample_rate=16000,
            feature_dim=80,
            blank_penalty=0.25,
            decoding_method="greedy_search",
        )
        self._lock = threading.Lock()

    def transcribe(self, payload: bytes) -> str:
        try:
            with wave.open(io.BytesIO(payload), "rb") as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000:
                    raise ValueError("expected mono PCM16 WAV at 16 kHz")
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
        except (wave.Error, EOFError, ValueError) as exc:
            raise ValueError(f"invalid ASR audio: {exc}") from exc
        with self._lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, samples.astype(np.float32) / 32768.0)
            self.recognizer.decode_stream(stream)
            return stream.result.text.strip()


asr: ZipformerASR | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global asr
    asr = await asyncio.to_thread(ZipformerASR)
    yield
    asr = None


app = FastAPI(title="Vietnamese Zipformer ASR", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict:
    return {"ready": asr is not None, "model": MODEL_ID, "language": "vi"}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "hynt"}]}


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str = Form("vi"),
    stream: str = Form("false"),
):
    del model
    if language not in {"vi", "vie", "Vietnamese", ""}:
        raise HTTPException(400, "This ASR service supports Vietnamese only.")
    if asr is None:
        raise HTTPException(503, "Zipformer is still loading.")
    payload = await file.read()
    try:
        text = await asyncio.to_thread(asr.transcribe, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if stream.lower() not in {"1", "true", "yes"}:
        return {"text": text, "language": "vi"}

    async def events():
        # The model is offline RNNT. SSE preserves the existing UI contract;
        # rolling VAD windows provide evolving hypotheses during live capture.
        parts = text.split(" ")
        for index, word in enumerate(parts):
            token = word if index == 0 else f" {word}"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
