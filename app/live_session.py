from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .config import Settings
from .tts_client import OmniVoiceClient
from .vllm_client import VLLMClients, pcm16_wav, stable_prefix


@dataclass
class SessionState:
    source: str = "auto"; target: str = "en"; speed: float = 1.0
    pcm: bytearray = field(default_factory=bytearray)
    vad_buffer: bytearray = field(default_factory=bytearray)
    last_processed_bytes: int = 0
    previous_transcript: str = ""
    previous_translation: str = ""
    committed_translation: str = ""
    tts_queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    tts_worker: asyncio.Task[Any] | None = None
    segment_index: int = 0
    first_text_ms: int = 0
    first_audio_ms: int = 0
    speech_started: bool = False
    utterance_count: int = 0
    energy_speech_frames: int = 0
    energy_silence_frames: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    vad: Any = None


def silence_wav(seconds: float = 0.5) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes(b"\0\0" * int(24000 * seconds))
    return output.getvalue()


class LiveSessionHandler:
    def __init__(self, settings: Settings):
        self.settings = settings; self.vllm = VLLMClients(settings); self.tts = OmniVoiceClient(settings.tts_base_url)

    @staticmethod
    def _new_vad(settings: Settings) -> Any:
        try:
            from omnivad import OmniStreamVAD
            return OmniStreamVAD(threshold=settings.vad_threshold, min_silence_frame=settings.vad_min_silence_frames)
        except TypeError:
            from omnivad import OmniStreamVAD
            return OmniStreamVAD()

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept(); state = SessionState(vad=self._new_vad(self.settings))
        state.tts_worker = asyncio.create_task(self._tts_worker(websocket, state))
        await websocket.send_json({"type": "session.ready", "sample_rate": 16000, "mode": self.settings.mode, "vad": "OmniStreamVAD"})
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    state.pcm.extend(chunk); state.vad_buffer.extend(chunk)
                    if len(state.pcm) % 32000 < len(chunk):
                        rms, peak = self._pcm_levels(bytes(state.pcm[-32000:]))
                        await websocket.send_json({"type": "audio.received", "seconds": len(state.pcm) / 32000,
                                                   "speech": state.speech_started, "rms": rms, "peak": peak})
                    endpoint = await self._feed_vad(websocket, state)
                    threshold = int(16000 * 2 * self.settings.live_window_seconds)
                    if state.speech_started and len(state.pcm) - state.last_processed_bytes >= threshold:
                        state.last_processed_bytes = len(state.pcm)
                        await self._update(websocket, state, final=False)
                    if endpoint:
                        if state.energy_speech_frames < self.settings.vad_min_voiced_frames:
                            await websocket.send_json({
                                "type": "utterance.discarded",
                                "reason": "not_enough_voiced_audio",
                                "voiced_ms": state.energy_speech_frames * 10,
                            })
                        else:
                            await self._update(websocket, state, final=True)
                            state.utterance_count += 1
                            await websocket.send_json({"type": "utterance.done", "index": state.utterance_count})
                        self._reset_utterance(state)
                        state.vad = self._new_vad(self.settings)
                elif message.get("text"):
                    event = json.loads(message["text"])
                    if event["type"] == "session.configure":
                        for key in ("source", "target", "speed"):
                            if key in event: setattr(state, key, event[key])
                    elif event["type"] == "input.commit":
                        if not state.speech_started:
                            rms, peak = self._pcm_levels(bytes(state.pcm))
                            if len(state.pcm) >= 9600 and rms >= 160:
                                state.speech_started = True
                                state.started_at = time.perf_counter()
                                await websocket.send_json({"type": "vad.speech_start", "time": 0,
                                                           "fallback": "energy", "rms": rms, "peak": peak})
                            elif state.utterance_count == 0:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Không nhận được giọng nói (RMS {rms}, peak {peak}). Kiểm tra mic/input preview.",
                                })
                                return
                        if state.speech_started:
                            await self._update(websocket, state, final=True)
                            state.utterance_count += 1
                        await self._finish_tts(websocket, state)
                        return
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.close()
        finally:
            if state.tts_worker and not state.tts_worker.done():
                state.tts_worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await state.tts_worker

    @staticmethod
    def _pcm_levels(pcm: bytes) -> tuple[int, int]:
        if len(pcm) < 2:
            return 0, 0
        import numpy as np
        samples = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32)
        if not len(samples):
            return 0, 0
        return round(float(np.sqrt(np.mean(samples * samples)))), int(np.max(np.abs(samples)))

    @staticmethod
    def _reset_utterance(state: SessionState) -> None:
        state.pcm.clear()
        state.last_processed_bytes = 0
        state.previous_transcript = ""
        state.previous_translation = ""
        state.committed_translation = ""
        state.first_text_ms = 0
        state.first_audio_ms = 0
        state.speech_started = False
        state.energy_speech_frames = 0
        state.energy_silence_frames = 0
        state.started_at = time.perf_counter()

    async def _finish_tts(self, ws: WebSocket, state: SessionState) -> None:
        await state.tts_queue.join()
        await state.tts_queue.put(None)
        await state.tts_queue.join()
        if state.tts_worker:
            await state.tts_worker
        await ws.send_json({"type": "speech.done", "segments": state.segment_index})
        await ws.send_json({"type": "session.done", "utterances": state.utterance_count})

    async def _tts_worker(self, ws: WebSocket, state: SessionState) -> None:
        while True:
            item = await state.tts_queue.get()
            if item is None:
                state.tts_queue.task_done()
                return
            text, speed, language, index, enqueued_at = item
            try:
                queue_wait_ms = round((time.perf_counter() - enqueued_at) * 1000)
                await ws.send_json({
                    "type": "speech.started", "index": index, "text": text,
                    "queued_at_ms": round((time.perf_counter() - state.started_at) * 1000),
                    "queue_wait_ms": queue_wait_ms,
                })
                if self.settings.mode == "mock":
                    audio, synthesis_ms = silence_wav(), 0
                else:
                    audio, synthesis_ms = await self.tts.synthesize(text, speed, language)
                now_ms = round((time.perf_counter() - state.started_at) * 1000)
                if not state.first_audio_ms:
                    state.first_audio_ms = now_ms
                await ws.send_json({
                    "type": "speech.segment", "index": index, "text": text,
                    "audio": base64.b64encode(audio).decode(), "mime": "audio/wav",
                    "synthesis_ms": synthesis_ms, "time_to_first_audio_ms": state.first_audio_ms,
                })
            except Exception as exc:
                await ws.send_json({"type": "error", "message": f"TTS segment {index} failed: {exc}"})
            finally:
                state.tts_queue.task_done()

    @staticmethod
    def _committable_translation(state: SessionState, translation: str, final: bool) -> str:
        stable = translation if final else stable_prefix(state.previous_translation, translation)
        state.previous_translation = translation
        if not stable.startswith(state.committed_translation):
            return ""
        tail = stable[len(state.committed_translation):]
        if not final:
            matches = list(re.finditer(r"[.!?。！？]+(?:\s+|$)", tail))
            if not matches:
                return ""
            tail = tail[:matches[-1].end()]
        segment = tail.strip()
        if segment:
            state.committed_translation = stable[:len(state.committed_translation) + len(tail)]
        return segment

    async def _feed_vad(self, ws: WebSocket, state: SessionState) -> bool:
        import numpy as np
        endpoint = False
        while len(state.vad_buffer) >= 320:
            frame = bytes(state.vad_buffer[:320]); del state.vad_buffer[:320]
            samples = np.frombuffer(frame, dtype=np.int16)
            frame_rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            if frame_rms >= self.settings.vad_energy_threshold:
                state.energy_speech_frames += 1
                state.energy_silence_frames = 0
            elif state.speech_started:
                state.energy_silence_frames += 1
            else:
                state.energy_speech_frames = 0
            result = state.vad.process(samples)
            if (not state.speech_started
                    and state.energy_speech_frames >= self.settings.vad_energy_start_frames):
                state.speech_started = True
                state.started_at = time.perf_counter()
                await ws.send_json({"type": "vad.speech_start", "time": 0, "fallback": "energy"})
            if result is None: continue
            if result.is_speech_start:
                if not state.speech_started:
                    state.speech_started = True
                    state.started_at = time.perf_counter()
                    await ws.send_json({"type": "vad.speech_start", "time": result.speech_start_frame * 0.01})
            if result.is_speech_end and state.speech_started:
                await ws.send_json({"type": "vad.speech_end", "time": result.speech_end_frame * 0.01})
                endpoint = True
            elif (state.speech_started
                  and state.energy_silence_frames >= self.settings.vad_energy_silence_frames):
                await ws.send_json({"type": "vad.speech_end", "time": 0, "fallback": "energy"})
                endpoint = True
                state.energy_silence_frames = 0
        return endpoint

    async def _update(self, ws: WebSocket, state: SessionState, final: bool) -> None:
        if not state.pcm: return
        mt_ttft_ms = 0
        mt_started_ms = round((time.perf_counter() - state.started_at) * 1000)
        if self.settings.mode == "mock":
            seconds = len(state.pcm) / 32000
            words = "Chúng ta đang thử nghiệm hệ thống phiên dịch giọng nói theo thời gian gần thực".split()
            transcript = " ".join(words[:max(1, min(len(words), int(seconds * 2)))])
            detected = "vi" if state.source == "auto" else state.source
            translation = f"Live translation preview · {transcript}"
        else:
            async def emit_dictation(text: str) -> None:
                await ws.send_json({
                    "type": "asr.dictation.delta",
                    "text": text,
                    "stable_text": stable_prefix(state.previous_transcript, text),
                    "final_pass": final,
                })

            transcript, detected = await self.vllm.transcribe_audio_stream(
                pcm16_wav(bytes(state.pcm)), state.source, emit_dictation
            )
            translation, mt_ttft_ms = await self.vllm.translate_with_metrics(
                transcript, detected if state.source == "auto" else state.source, state.target
            )
            if translation and not state.first_text_ms:
                state.first_text_ms = mt_started_ms + mt_ttft_ms
        stable = transcript if final else stable_prefix(state.previous_transcript, transcript)
        state.previous_transcript = transcript
        stable_translation = (
            translation if final else stable_prefix(state.previous_translation, translation)
        )
        await ws.send_json({"type": "translation.update", "transcript": transcript, "translation": translation,
                            "stable_transcript": stable, "stable_translation": stable_translation,
                            "detected_language": detected, "final": final,
                            "latency_ms": round((time.perf_counter() - state.started_at) * 1000),
                            "time_to_first_text_ms": state.first_text_ms})
        segment = self._committable_translation(state, translation, final)
        if segment:
            state.segment_index += 1
            await state.tts_queue.put((segment, float(state.speed), state.target,
                                       state.segment_index, time.perf_counter()))
