from __future__ import annotations

import asyncio
import io
import json
import time
import wave
from collections.abc import Awaitable, Callable
import httpx

from .config import Settings

LANGUAGE_NAMES = {
    "auto": "the detected language", "vi": "Vietnamese", "en": "English",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French",
    "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "th": "Thai", "id": "Indonesian",
}


def pcm16_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def is_pcm16_mono_16k(audio: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            return (
                wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getframerate() == 16000
                and wav.getcomptype() == "NONE"
            )
    except (wave.Error, EOFError):
        return False


async def normalize_for_asr(audio: bytes) -> bytes:
    """Normalize browser uploads to the WAV format accepted by vLLM ASR."""
    if is_pcm16_mono_16k(audio):
        return audio
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(audio)
    if process.returncode != 0 or not stdout:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Audio conversion failed: {detail or 'unsupported input format'}")
    return stdout


class VLLMClients:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.vllm_api_key}"}, timeout=120
        )

    async def health(self) -> dict[str, bool]:
        async def check(url: str) -> bool:
            try:
                return (await self.client.get(f"{url.rstrip('/')}/models", timeout=2)).is_success
            except httpx.HTTPError:
                return False
        return {"asr": await check(self.settings.asr_base_url), "translation": await check(self.settings.mt_base_url)}

    async def transcribe(self, pcm: bytes, language: str) -> tuple[str, str]:
        return await self.transcribe_audio(pcm16_wav(pcm), "live.wav", "audio/wav", language)

    async def transcribe_audio(self, audio: bytes, filename: str, mime: str, language: str) -> tuple[str, str]:
        audio = await normalize_for_asr(audio)
        filename, mime = "audio-16k.wav", "audio/wav"
        data = {"model": self.settings.asr_model, "response_format": "json"}
        if language != "auto": data["language"] = language
        try:
            response = await self.client.post(
                f"{self.settings.asr_base_url.rstrip('/')}/audio/transcriptions",
                files={"file": (filename, audio, mime)}, data=data,
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"ASR vLLM is unavailable at {self.settings.asr_base_url}"
            ) from exc
        if response.is_error:
            try:
                detail = response.json().get("error", {}).get("message")
            except ValueError:
                detail = response.text
            raise RuntimeError(f"ASR vLLM rejected the request: {detail or response.status_code}")
        result = response.json()
        return (result.get("text") or result.get("transcription") or "").strip(), str(result.get("language", language))

    async def transcribe_audio_stream(
        self,
        audio: bytes,
        language: str,
        on_delta: Callable[[str], Awaitable[None]],
    ) -> tuple[str, str]:
        audio = await normalize_for_asr(audio)
        data = {"model": self.settings.asr_model, "response_format": "json", "stream": "true"}
        if language != "auto":
            data["language"] = language
        chunks: list[str] = []
        try:
            async with self.client.stream(
                "POST",
                f"{self.settings.asr_base_url.rstrip('/')}/audio/transcriptions",
                files={"file": ("live-16k.wav", audio, "audio/wav")},
                data=data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    payload = json.loads(line[6:])
                    token = payload.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                    if token:
                        chunks.append(token)
                        await on_delta("".join(chunks))
        except httpx.ConnectError as exc:
            raise RuntimeError(f"ASR vLLM is unavailable at {self.settings.asr_base_url}") from exc
        return "".join(chunks).strip(), language

    async def translate(self, text: str, source: str, target: str) -> str:
        translation, _ = await self.translate_with_metrics(text, source, target)
        return translation

    async def translate_with_metrics(self, text: str, source: str, target: str) -> tuple[str, int]:
        pair_guidance = ""
        if source == "vi" and target == "en":
            pair_guidance = (
                " In Vietnamese conversational speech, translate 'alo' as 'hello' "
                "(or 'hey' when more natural); never leave it as 'alo'."
            )
        prompt = (
            f"Translate the following text from {LANGUAGE_NAMES.get(source, source)} "
            f"to {LANGUAGE_NAMES.get(target, target)}. Output only the translation. "
            "Translate every translatable expression, including greetings, fillers, "
            "interjections, and spoken number sequences, into a natural target-language "
            "equivalent. Do not copy source-language words unless they are proper names, "
            "technical terms, or have no natural equivalent. Preserve the meaning and "
            f"appropriate punctuation.{pair_guidance}\n\n{text}"
        )
        started = time.perf_counter()
        first_token_ms = 0
        chunks: list[str] = []
        try:
            async with self.client.stream(
                "POST", f"{self.settings.mt_base_url.rstrip('/')}/chat/completions",
                json={"model": self.settings.mt_model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2, "top_p": 0.6, "max_tokens": 512, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    token = data.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                    if token:
                        if not first_token_ms:
                            first_token_ms = round((time.perf_counter() - started) * 1000)
                        chunks.append(token)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Translation vLLM is unavailable at {self.settings.mt_base_url}"
            ) from exc
        return "".join(chunks).strip(), first_token_ms


def stable_prefix(previous: str, current: str) -> str:
    index = 0
    while index < min(len(previous), len(current)) and previous[index] == current[index]: index += 1
    prefix = current[:index]
    if index < len(current) and prefix and not prefix[-1].isspace():
        prefix = prefix.rsplit(" ", 1)[0] if " " in prefix else ""
    return prefix.strip()
