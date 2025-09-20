import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_json_logging(level: int | str) -> None:
    """
    Configure the root logger to emit JSON-formatted logs to stdout.
    Idempotent: reconfigures unconditionally (clears previous handlers).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(funcName)s %(message)s"
    json_formatter = jsonlogger.JsonFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(json_formatter)
    root.addHandler(handler)


def configure_text_logging(level: int | str) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d -> %(funcName)s() - %(message)s",
        stream=sys.stdout,
    )


def configure_logging(log_level: int | str, log_type: str) -> None:
    match log_type.upper():
        case "JSON":
            configure_json_logging(log_level)
        case "TEXT":
            configure_text_logging(log_level)
        case _:
            raise ValueError(f" Unknown logging configuration type: {log_type}")
