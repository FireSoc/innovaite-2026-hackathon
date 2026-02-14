from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # CommonStack (Anthropic-style Messages API)
    commonstack_api_key: str = ""
    commonstack_base_url: str = "https://api.commonstack.ai/v1"
    commonstack_model: str = "gpt-4o-mini"

    # LLM provider: "commonstack" or "gemini". If commonstack and key set, use CommonStack; else Gemini.
    llm_provider: str = "commonstack"

    # OpenFEMA
    fema_api_base: str = "https://www.fema.gov/api/open/v2"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": ["../.env", ".env"], "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
