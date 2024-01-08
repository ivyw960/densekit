"""Lightweight logging setup used by the CLI and trainers."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "densekit") -> logging.Logger:
    """Return a package logger, configuring a simple handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root = logging.getLogger("densekit")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
