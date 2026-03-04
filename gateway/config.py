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

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = Field("http://localhost:4317", env="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_enabled: bool = Field(True, env="OTEL_ENABLED")

    # Circuit breaker
    cb_failure_threshold: int = Field(5, env="CB_FAILURE_THRESHOLD")
    cb_recovery_timeout: float = Field(30.0, env="CB_RECOVERY_TIMEOUT")
    cb_half_open_max_calls: int = Field(1, env="CB_HALF_OPEN_MAX_CALLS")

    # Token quota (0 = unlimited)
    daily_token_quota: int = Field(0, env="DAILY_TOKEN_QUOTA")

    # Directory that holds downloaded model weights / GGUF files
    model_cache_dir: str = Field("models", env="MODEL_CACHE_DIR")

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}


settings = Settings()
