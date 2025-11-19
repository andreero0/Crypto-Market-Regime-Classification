#!/usr/bin/env python3
"""
Start the FastAPI server.

Usage:
    python scripts/run_api.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.utils.config import settings


def main():
    """Start the API server."""
    print("=" * 60)
    print("Starting Crypto Market Regime Classification API")
    print("=" * 60)
    print(f"Environment: {settings.app_env}")
    print(f"Host: {settings.api_host}:{settings.api_port}")
    print(f"Model: {settings.model_path}")
    print(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
