from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LiveTrans Desk"
    mode: str = "mock"
    asr_model: str = "g-group-ai-lab/gipformer1.5-65M-rnnt"
    mt_model: str = "tencent/Hy-MT2-1.8B"
    tts_model: str = "k2-fsa/OmniVoice"
    device: str = "auto"
    max_upload_mb: int = 30
    max_file_seconds: int = 600
    asr_base_url: str = "http://127.0.0.1:8101/v1"
    mt_base_url: str = "http://127.0.0.1:8102/v1"
    tts_base_url: str = "http://127.0.0.1:8103"
    vllm_api_key: str = "local"
    live_window_seconds: float = 2.0
    live_min_stable_chars: int = 18
    vad_threshold: float = 0.8
    vad_min_silence_frames: int = 60

    model_config = SettingsConfigDict(env_prefix="LIVETRANS_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
