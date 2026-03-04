#!/usr/bin/env python3
"""Entry point: run the gateway with uvicorn."""
import asyncio
import os
import signal
import sys

# Ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from gateway.config import settings


def _make_signal_handler(server: uvicorn.Server):
    """Return a signal handler that sets the uvicorn shutdown event and the gateway drain event."""
    def handler(sig, frame):
        import structlog
        from gateway.app import _shutting_down
        log = structlog.get_logger("gateway.server")
        log.info("shutdown_signal_received", signal=sig)
        # Signal the app to stop accepting new requests
        _shutting_down.set()
        # Tell uvicorn to exit after draining
        server.should_exit = True

    return handler


if __name__ == "__main__":
    config = uvicorn.Config(
        "gateway.app:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        workers=1,          # single worker — async handles concurrency
        loop="uvloop",
        http="httptools",
        log_config=None,    # we handle logging via structlog
        access_log=False,
        server_header=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    server = uvicorn.Server(config)

    # Register SIGTERM/SIGINT handlers before starting
    signal.signal(signal.SIGTERM, _make_signal_handler(server))
    signal.signal(signal.SIGINT, _make_signal_handler(server))

    server.run()
