"""Compatibility helpers for supported Python versions."""
from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 fallback for enum.StrEnum."""


__all__ = ["StrEnum"]
