from __future__ import annotations

import asyncio
import io
import json
import os
import re
import threading
import time
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
PROVIDER = os.getenv("LIVETRANS_ZIPFORMER_PROVIDER", "cpu").strip().lower()
CAPU_ENABLED = os.getenv("LIVETRANS_CAPU_ENABLED", "1").lower() not in {"0", "false", "no"}
CAPU_MODEL_ID = os.getenv("LIVETRANS_CAPU_MODEL", "dragonSwing/vibert-capu")
CAPU_REVISION = os.getenv(
    "LIVETRANS_CAPU_REVISION", "261c60f2c30b02455dfce21a43c3ef14fc26992c"
)
CAPU_DEVICE = os.getenv("LIVETRANS_CAPU_DEVICE", "cuda")
CAPU_KEEP_BIAS = float(os.getenv("LIVETRANS_CAPU_KEEP_BIAS", "0.10"))
CAPU_CASE_BIAS = float(os.getenv("LIVETRANS_CAPU_CASE_BIAS", "0.10"))
CAPU_CONTEXT_WORDS = int(os.getenv("LIVETRANS_CAPU_CONTEXT_WORDS", "30"))
ITN_CACHE_DIR = Path(
    os.getenv("LIVETRANS_VI_ITN_CACHE_DIR", ".cache/livetrans/vi_itn")
).expanduser().resolve()


class ZipformerASR:
    def __init__(self) -> None:
        import sherpa_onnx
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
            InverseNormalizer,
        )

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
            provider=PROVIDER,
        )
        ITN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.text_normalizer = InverseNormalizer(
            lang="vi",
            input_case="lower_cased",
            cache_dir=str(ITN_CACHE_DIR),
        )
        self.capu = None
        if CAPU_ENABLED:
            from app.vendor.vibert_capu import GecBERTModel

            capu_dir = Path(snapshot_download(
                CAPU_MODEL_ID,
                revision=CAPU_REVISION,
                allow_patterns=["config.json", "pytorch_model.bin"],
            ))
            vocabulary_dir = Path(__file__).parent / "vendor" / "vibert_capu" / "vocabulary"
            self.capu = GecBERTModel(
                vocab_path=str(vocabulary_dir),
                model_paths=str(capu_dir),
                device=CAPU_DEVICE,
                split_chunk=True,
                max_len=80,
                chunk_size=56,
                overlap_size=16,
                iterations=3,
                min_error_probability=0.25,
                confidence=CAPU_KEEP_BIAS,
                case_confidence=CAPU_CASE_BIAS,
            )
            # Pay CUDA kernel/tokenizer warm-up during startup, not on the first user.
            self.capu("xin chào bạn")
        self._lock = threading.Lock()

    @staticmethod
    def _post_process(text: str, capitalize_initial: bool) -> str:
        text = re.sub(r",+", ",", text)
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r",\s*\.", ".", text)
        text = re.sub(r"\s+([,.:?])", r"\1", text)
        text = re.sub(r"([,.:?])([^\s\d])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text).strip()
        if capitalize_initial and text:
            text = text[0].upper() + text[1:]
        return re.sub(
            r"([.?]\s+)([^\W_])",
            lambda match: match.group(1) + match.group(2).upper(),
            text,
        )

    def transcribe(
        self,
        payload: bytes,
        apply_capu: bool = True,
        capu_context: str = "",
    ) -> dict:
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
            raw_text = stream.result.text.strip()
            if not raw_text:
                return {"raw_text": "", "normalized_text": "", "text": "", "capu_ms": 0}
            normalized_text = self.text_normalizer.inverse_normalize(
                raw_text.lower(), verbose=False
            ).strip()
            text = normalized_text
            capu_ms = 0
            context_used = False
            context_boundary = ""
            if apply_capu and self.capu is not None and normalized_text:
                started = time.perf_counter()
                context_words = capu_context.split()[-CAPU_CONTEXT_WORDS:]
                context_used = bool(context_words)
                capu_input = " ".join([*context_words, normalized_text]).strip()
                formatted_words = self.capu(capu_input)[0].strip().split()
                if context_words and len(formatted_words) >= len(context_words):
                    boundary_match = re.search(r"[,.:?]+$", formatted_words[len(context_words) - 1])
                    context_boundary = boundary_match.group(0) if boundary_match else ""
                text = " ".join(formatted_words[len(context_words):])
                text = self._post_process(text, capitalize_initial=not context_words)
                capu_ms = round((time.perf_counter() - started) * 1000)
            return {
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "text": text,
                "capu_ms": capu_ms,
                "context_used": context_used,
                "context_boundary": context_boundary,
            }


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
    return {
        "ready": asr is not None,
        "model": MODEL_ID,
        "language": "vi",
        "inverse_text_normalization": "nemo-vi",
        "provider": PROVIDER,
        "capu": CAPU_MODEL_ID if CAPU_ENABLED else None,
    }


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "hynt"}]}


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str = Form("vi"),
    stream: str = Form("false"),
    capu: str = Form("true"),
    capu_context: str = Form(""),
):
    del model
    if language not in {"vi", "vie", "Vietnamese", ""}:
        raise HTTPException(400, "This ASR service supports Vietnamese only.")
    if asr is None:
        raise HTTPException(503, "Zipformer is still loading.")
    payload = await file.read()
    try:
        result = await asyncio.to_thread(
            asr.transcribe,
            payload,
            capu.lower() in {"1", "true", "yes"},
            capu_context,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if stream.lower() not in {"1", "true", "yes"}:
        return {**result, "language": "vi"}

    async def events():
        # The model is offline RNNT. SSE preserves the existing UI contract;
        # rolling VAD windows provide evolving hypotheses during live capture.
        parts = result["text"].split(" ")
        for index, word in enumerate(parts):
            token = word if index == 0 else f" {word}"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
