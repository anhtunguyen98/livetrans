from __future__ import annotations

import base64
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .live_session import LiveSessionHandler
from .pipeline import MockPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
settings = get_settings()
mock_pipeline = MockPipeline()
live_handler = LiveSessionHandler(settings)
app = FastAPI(title=settings.app_name, version="0.2.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "mode": settings.mode, "loaded": settings.mode == "real",
        "vllm": {"asr_url": settings.asr_base_url, "mt_url": settings.mt_base_url, "tts_url": settings.tts_base_url},
        "models": {"asr": settings.asr_model, "translation": settings.mt_model, "speech": settings.tts_model},
    }


@app.get("/api/vllm/health")
async def vllm_health() -> dict:
    health = await live_handler.vllm.health()
    health["speech"] = await live_handler.tts.health()
    return health


@app.post("/api/translate")
async def translate(
    audio: UploadFile = File(...), source_language: str = Form("auto"), target_language: str = Form("en"),
    speed: float = Form(1.0),
) -> dict:
    started = time.perf_counter(); payload = await audio.read()
    if not payload: raise HTTPException(400, "Audio is empty.")
    if len(payload) > settings.max_upload_mb * 1024 * 1024: raise HTTPException(413, f"Audio exceeds {settings.max_upload_mb} MB.")
    if target_language == source_language and source_language != "auto": raise HTTPException(400, "Source and target languages must differ.")
    if not 0.7 <= speed <= 1.3: raise HTTPException(400, "Speed must be between 0.7 and 1.3.")
    try:
        if settings.mode == "mock":
            result = await mock_pipeline.translate(source_language, target_language)
            transcript, translation, detected, speech, mime = result.transcript, result.translation, result.detected_language, result.audio_bytes, result.audio_mime
        else:
            transcript, detected = await live_handler.vllm.transcribe_audio(
                payload, audio.filename or "recording.webm", audio.content_type or "application/octet-stream", source_language
            )
            translation, mt_ttft_ms = await live_handler.vllm.translate_with_metrics(
                transcript, detected if source_language == "auto" else source_language, target_language
            )
            speech_started = time.perf_counter()
            speech, _ = await live_handler.tts.synthesize(translation, speed, target_language)
            ttfa_ms = round((time.perf_counter() - speech_started) * 1000)
            mime = "audio/wav"
    except Exception as exc:
        raise HTTPException(503, f"Pipeline failed: {exc}") from exc
    return {"transcript": transcript, "translation": translation, "detected_language": detected,
            "audio": base64.b64encode(speech).decode("ascii"), "audio_mime": mime,
            "latency_ms": round((time.perf_counter() - started) * 1000), "mode": settings.mode,
            "time_to_first_text_ms": mt_ttft_ms if settings.mode != "mock" else 0,
            "time_to_first_audio_ms": ttfa_ms if settings.mode != "mock" else 0}


@app.websocket("/api/live")
async def live_translate(websocket: WebSocket) -> None:
    await live_handler.handle(websocket)
