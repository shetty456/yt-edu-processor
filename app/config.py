from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Sarvam-m
    sarvam_api_key: str
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    sarvam_model: str = "sarvam-m"

    # OpenAI (eval pass only)
    openai_api_key: str
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_eval_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # App behaviour
    max_concurrent_jobs: int = 5
    request_timeout_seconds: int = 360
    max_video_duration_seconds: int = 7200   # 2 hours
    chunk_word_limit: int = 6000             # trigger chunking above this
    chunk_target_words: int = 5000           # target words per chunk
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()