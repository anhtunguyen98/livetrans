from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineResult:
    transcript: str
    translation: str
    detected_language: str
    audio_bytes: bytes
    audio_mime: str = "audio/wav"


class MockPipeline:
    """Deterministic UI preview that does not load model weights."""

    async def translate(self, source: str, target: str) -> PipelineResult:
        await asyncio.sleep(0.25)
        transcript = "Chúng ta đang thử nghiệm hệ thống phiên dịch giọng nói trực tiếp."
        translations = {
            "en": "We are testing a live speech translation system.",
            "zh": "我们正在测试实时语音翻译系统。",
            "ja": "ライブ音声翻訳システムをテストしています。",
            "ko": "실시간 음성 번역 시스템을 테스트하고 있습니다.",
            "fr": "Nous testons un système de traduction vocale en direct.",
        }
        return PipelineResult(
            transcript=transcript,
            translation=translations.get(target, f"[{target}] {transcript}"),
            detected_language="vi" if source == "auto" else source,
            audio_bytes=_silence_wav(),
        )


def _silence_wav(seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return output.getvalue()
