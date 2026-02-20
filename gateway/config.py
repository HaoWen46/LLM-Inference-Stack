from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Upstream vLLM
    vllm_url: str = Field("http://localhost:8000", alias="VLLM_URL", env="VLLM_URL")
    vllm_api_key: str = Field(..., alias="VLLM_API_KEY", env="VLLM_API_KEY")
    served_model_name: str = Field("llama-7b", env="SERVED_MODEL_NAME")

    # Gateway listen
    gateway_host: str = Field("0.0.0.0", env="GATEWAY_HOST")
    gateway_port: int = Field(8080, env="GATEWAY_PORT")

    # Auth — comma-separated list of valid bearer tokens
    gateway_api_keys: str = Field("dev-key-1", env="GATEWAY_API_KEYS")

    # Rate limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")

    # Timeouts
    request_timeout_seconds: float = Field(300.0, env="REQUEST_TIMEOUT_SECONDS")
    connect_timeout_seconds: float = Field(5.0, env="CONNECT_TIMEOUT_SECONDS")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}


settings = Settings()
