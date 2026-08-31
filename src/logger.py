import logging
from logging import Logger
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name=None, level="INFO"):
    """Return a logger that writes to the console.

    Calling this more than once with the same name reuses the same handler
    instead of stacking duplicates.
    """
    logger = logging.getLogger(name if name else "AWS-AI-Agent")
    logger.setLevel(level.upper() if level else DEFAULT_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False  # Avoid the root logger printing the record again
    return logger


if __name__ == "__main__":
    log = get_logger(__name__, level="DEBUG")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("Exception message with traceback")
