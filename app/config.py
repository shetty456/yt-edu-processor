from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Sarvam-m
    sarvam_api_key: str
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    sarvam_model: str = "sarvam-30b"

    # Cloudinary
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str

    # PDF limits
    max_pdf_size_mb: int = 5
    max_pdf_pages: int = 20

    # App behaviour
    max_concurrent_jobs: int = 5
    request_timeout_seconds: int = 360
    max_video_duration_seconds: int = 7200   # 2 hours
    chunk_word_limit: int = 6000             # trigger chunking above this
    chunk_target_words: int = 5000           # target words per chunk

    # PDF pipeline (lower than YouTube — PDF text tokenizes much more densely)
    pdf_chunk_word_limit: int = 1600         # force chunking at this threshold
    pdf_chunk_target_words: int = 1600       # target words per chunk
    pdf_quiz_word_limit: int = 2100          # max safe words for quiz prompt at 3 tokens/word
    log_level: str = "INFO"

    # Webshare residential proxy (optional — falls back to direct if not set)
    webshare_proxy_username: str | None = None
    webshare_proxy_password: str | None = None

    # Web URL pipeline
    web_min_word_count: int = 200        # reject pages below this
    web_fetch_timeout_seconds: int = 20  # httpx timeout
    web_quiz_word_limit: int = 2000      # max safe words for quiz prompt at ~1.88 tokens/word
    web_chunk_word_limit: int = 3500     # trigger chunking above this word count
    web_chunk_target_words: int = 3000   # target words per chunk (~5800 tokens, under 7168 limit)

    # Response cache
    cache_ttl_seconds: int = 86400  # 24 hours; set 0 to disable
    cache_max_size: int = 500       # max cached responses (each ~5–20 KB)


@lru_cache
def get_settings() -> Settings:
    return Settings()