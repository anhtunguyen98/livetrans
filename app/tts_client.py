from __future__ import annotations

import httpx


class OmniVoiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    async def health(self) -> bool:
        try:
            return (await self.client.get(f"{self.base_url}/healthz", timeout=2)).is_success
        except httpx.HTTPError:
            return False

    async def synthesize(self, text: str, speed: float, language: str | None = None) -> tuple[bytes, int]:
        try:
            response = await self.client.post(f"{self.base_url}/v1/audio/speech",
                                              json={"text": text, "speed": speed, "language": language})
        except httpx.ConnectError as exc:
            raise RuntimeError(f"OmniVoice is unavailable at {self.base_url}") from exc
        if response.is_error:
            raise RuntimeError(f"OmniVoice rejected the request: {response.text or response.status_code}")
        return response.content, int(response.headers.get("X-Synthesis-Ms", "0"))
