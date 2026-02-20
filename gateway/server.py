#!/usr/bin/env python3
"""Entry point: run the gateway with uvicorn."""
import os
import sys

# Ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from gateway.config import settings

if __name__ == "__main__":
    uvicorn.run(
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
