import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    propagate: bool = True,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(handler)

    if log_file and not any(
        isinstance(h, logging.FileHandler)
        and h.baseFilename == os.path.abspath(log_file)
        for h in logger.handlers
    ):
        add_file_handler(logger, log_file, level, log_format, date_format)

    return logger


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: str = "INFO",
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(handler)
    return logger
