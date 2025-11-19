"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path
from src.utils.config import settings


def setup_logging():
    """Configure application logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log"),
        ],
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    logging.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
