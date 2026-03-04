import logging
import logging.handlers
import os
import sys
import structlog
from gateway.config import settings

LOG_FILE = os.environ.get("GATEWAY_LOG_FILE", "/var/log/vllm/gateway.log")
LOG_FILE_ENABLED = os.environ.get("GATEWAY_LOG_FILE_ENABLED", "true").lower() == "true"


def _inject_otel_context(logger, method, event_dict):
    """Inject OpenTelemetry trace_id and span_id into every log record."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except ImportError:
        pass
    return event_dict


def configure_logging() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Build handlers list
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if LOG_FILE_ENABLED:
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=50 * 1024 * 1024,  # 50 MB
                backupCount=5,
            )
            handlers.append(file_handler)
        except OSError:
            # Can't write to log file (e.g., no permission in dev) — stdout only
            pass

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        _inject_otel_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [structlog.processors.JSONRenderer()],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
