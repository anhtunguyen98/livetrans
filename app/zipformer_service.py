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

MODEL_ID = os.getenv("LIVETRANS_ASR_MODEL", "checkpoint-122000-onnx-int8")
MODEL_DIR = os.getenv(
    "LIVETRANS_ZIPFORMER_MODEL_DIR",
    "/workspace/livetrans/models/checkpoint-122000-onnx-int8",
)
NUM_THREADS = int(os.getenv("LIVETRANS_ZIPFORMER_THREADS", "4"))
USE_INT8 = os.getenv("LIVETRANS_ZIPFORMER_INT8", "1").lower() not in {"0", "false", "no"}
PROVIDER = os.getenv("LIVETRANS_ZIPFORMER_PROVIDER", "cpu").strip().lower()
CAPU_ENABLED = os.getenv("LIVETRANS_CAPU_ENABLED", "1").lower() not in {"0", "false", "no"}
CAPU_MODEL_DIR = os.getenv(
    "LIVETRANS_CAPU_MODEL_DIR",
    "/workspace/vicapu/outputs/videberta-xsmall-capu-bilingual-pruned-onnx-int8",
)
CAPU_MODEL_FILE = os.getenv("LIVETRANS_CAPU_MODEL_FILE", "model.int8.onnx")
CAPU_THREADS = int(os.getenv("LIVETRANS_CAPU_THREADS", "4"))
CAPU_PUNCT_NONE_BIAS = float(os.getenv("LIVETRANS_CAPU_PUNCT_NONE_BIAS", "0"))
CAPU_CASE_KEEP_BIAS = float(os.getenv("LIVETRANS_CAPU_CASE_KEEP_BIAS", "0"))
CAPU_CONTEXT_WORDS = int(os.getenv("LIVETRANS_CAPU_CONTEXT_WORDS", "30"))
KENLM_ENABLED = os.getenv("LIVETRANS_KENLM_ENABLED", "1").lower() not in {"0", "false", "no"}
KENLM_MODEL = os.getenv(
    "LIVETRANS_KENLM_MODEL",
    "/workspace/livetrans/.cache/kenlm/3-gram-lm.binary",
)
KENLM_MAX_ACTIVE_PATHS = int(os.getenv("LIVETRANS_KENLM_MAX_ACTIVE_PATHS", "4"))
KENLM_MIN_GAIN = float(os.getenv("LIVETRANS_KENLM_MIN_GAIN", "0.05"))
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
                allow_patterns=["*.onnx", "config.json", "tokens.txt", "bpe.model"],
            ))

        def model_file(component: str) -> Path:
            candidates = sorted(model_dir.glob(f"{component}-*.onnx"))
            preferred = [path for path in candidates if (".int8." in path.name) == USE_INT8]
            choices = preferred or candidates
            if not choices:
                raise FileNotFoundError(f"No {component} ONNX file found in {model_dir}")
            return choices[0]

        tokens = model_dir / "tokens.txt"
        if not tokens.exists():
            tokens = model_dir / "config.json"
        if not tokens.exists():
            raise FileNotFoundError(f"No tokens.txt or config.json found in {model_dir}")
        encoder = model_file("encoder")
        decoder = model_file("decoder")
        joiner = model_file("joiner")
        self.model_dir = model_dir
        self.model_files = {
            "tokens": tokens.name,
            "encoder": encoder.name,
            "decoder": decoder.name,
            "joiner": joiner.name,
        }
        recognizer_args = dict(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=NUM_THREADS,
            sample_rate=16000,
            feature_dim=80,
            blank_penalty=0.25,
            provider=PROVIDER,
        )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            **recognizer_args,
            decoding_method="greedy_search",
        )
        self.beam_recognizer = None
        self.kenlm = None
        kenlm_path = Path(KENLM_MODEL).expanduser().resolve()
        if KENLM_ENABLED:
            if not kenlm_path.is_file():
                raise FileNotFoundError(f"KenLM model not found: {kenlm_path}")
            import kenlm

            self.kenlm = kenlm.Model(str(kenlm_path))
            # sherpa's `lm` option is for an ONNX neural LM, not KenLM. Decode
            # a second candidate and let the word-level 3-gram rerank finals.
            self.beam_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                **recognizer_args,
                decoding_method="modified_beam_search",
                max_active_paths=KENLM_MAX_ACTIVE_PATHS,
            )
        ITN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from app.custom_itn import build_normalizer
        self.text_normalizer, self.itn_directory = build_normalizer(ITN_CACHE_DIR)
        self.capu = None
        if CAPU_ENABLED:
            from app.videberta_capu import ViDeBERTaCapu

            self.capu = ViDeBERTaCapu(
                CAPU_MODEL_DIR,
                model_name=CAPU_MODEL_FILE,
                threads=CAPU_THREADS,
                punctuation_none_bias=CAPU_PUNCT_NONE_BIAS,
                case_keep_bias=CAPU_CASE_KEEP_BIAS,
            )
            # Pay tokenizer/ORT graph warm-up during startup, not on the first user.
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
            waveform = samples.astype(np.float32) / 32768.0

            def decode(recognizer) -> str:
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, waveform)
                recognizer.decode_stream(stream)
                return stream.result.text.strip()

            raw_candidates = [decode(self.recognizer)]
            lm_ms = 0
            if apply_capu and self.beam_recognizer is not None:
                lm_started = time.perf_counter()
                beam_text = decode(self.beam_recognizer)
                if beam_text and beam_text not in raw_candidates:
                    raw_candidates.append(beam_text)
                lm_ms = round((time.perf_counter() - lm_started) * 1000)
            raw_text = raw_candidates[0]
            if not raw_text:
                return {"raw_text": "", "normalized_text": "", "text": "", "capu_ms": 0}

            normalized_candidates = [
                self.text_normalizer.inverse_normalize(candidate.lower(), verbose=False).strip()
                for candidate in raw_candidates
            ]
            normalized_text = normalized_candidates[0]
            text = normalized_text
            capu_ms = 0
            context_used = False
            context_boundary = ""
            selected_candidate = 0
            lm_scores: list[float] = []
            if apply_capu and self.capu is not None and normalized_text:
                started = time.perf_counter()
                context_words = capu_context.split()[-CAPU_CONTEXT_WORDS:]
                context_used = bool(context_words)
                formatted_candidates: list[list[str]] = []
                for candidate in normalized_candidates:
                    capu_input = " ".join([*context_words, candidate]).strip()
                    formatted_candidates.append(self.capu(capu_input)[0].strip().split())

                if self.kenlm is not None and len(formatted_candidates) > 1:
                    for formatted in formatted_candidates:
                        segment = " ".join(formatted[len(context_words):])
                        lm_input = re.sub(r"[,.:?]", "", segment).lower()
                        word_count = max(1, len(lm_input.split()))
                        lm_scores.append(self.kenlm.score(lm_input, bos=True, eos=True) / word_count)
                    best = int(np.argmax(lm_scores))
                    base_words = max(1, len(normalized_candidates[0].split()))
                    best_words = len(normalized_candidates[best].split())
                    similar_length = 0.8 <= best_words / base_words <= 1.2
                    if similar_length and lm_scores[best] >= lm_scores[0] + KENLM_MIN_GAIN:
                        selected_candidate = best

                raw_text = raw_candidates[selected_candidate]
                normalized_text = normalized_candidates[selected_candidate]
                formatted_words = formatted_candidates[selected_candidate]
                if context_words and len(formatted_words) >= len(context_words):
                    boundary_match = re.search(r"[,.:?]+$", formatted_words[len(context_words) - 1])
                    context_boundary = boundary_match.group(0) if boundary_match else ""
                text = " ".join(formatted_words[len(context_words):])
                text = self._post_process(text, capitalize_initial=not context_words)
                from app.custom_itn import restore_written_forms
                text = restore_written_forms(text, source=normalized_text)
                capu_ms = round((time.perf_counter() - started) * 1000)
            return {
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "text": text,
                "capu_ms": capu_ms,
                "context_used": context_used,
                "context_boundary": context_boundary,
                "lm": {
                    "enabled": self.kenlm is not None,
                    "candidates": len(raw_candidates),
                    "selected": selected_candidate,
                    "scores": [round(score, 4) for score in lm_scores],
                    "ms": lm_ms,
                },
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
        "capu": {
            "backend": "videberta-onnx-int8",
            "directory": CAPU_MODEL_DIR,
            "model": CAPU_MODEL_FILE,
        } if CAPU_ENABLED else None,
        "kenlm": {
            "enabled": asr is not None and asr.kenlm is not None,
            "model": KENLM_MODEL,
            "order": asr.kenlm.order if asr is not None and asr.kenlm is not None else None,
            "stage": "post-capu-final-reranker",
        },
        "files": asr.model_files if asr is not None else None,
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": MODEL_ID.split("/", 1)[0]}],
    }


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
