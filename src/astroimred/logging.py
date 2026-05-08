"""Shared logging configuration for astroimred."""

import logging

__all__ = ["logger", "set_log_level", "enable_console_logging"]

logger = logging.getLogger("astroimred")
logger.addHandler(logging.NullHandler())


def set_log_level(level):
    """Set the log level for astroimred."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


def enable_console_logging(level=logging.INFO, format=None):
    """Enable console logging for interactive use."""
    if format is None:
        format = "[%(levelname)s] %(message)s"

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(level)
