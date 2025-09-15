import logging
import sys
from logging import Logger

from src.config.config import LOGGING_LEVEL


def get_logger(name: str) -> Logger:
    """
    Configures and returns a logger with the configured logging level.

    Args:
        name: The name of the logger, typically __name__ of the calling module

    Returns:
        A configured logger instance with the specified logging level
    """
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        # Set the logging level from config, defaulting to ERROR if invalid
        try:
            level = getattr(logging, LOGGING_LEVEL, logging.ERROR)
            if not isinstance(level, int):
                level = logging.ERROR
        except (TypeError, AttributeError):
            level = logging.ERROR

        logger.setLevel(level)

        # Configure formatter and handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
