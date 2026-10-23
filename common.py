"""Small helpers shared across CrossVerify modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def norm_path(path: str | Path) -> str:
    """Normalise Windows-style separators so paths work on any OS."""

    return str(path).replace("\\", "/")


def resolve_image_path(path: str | Path) -> Path:
    """Resolve a (possibly backslashed, project-relative) DB image path."""

    candidate = Path(norm_path(path))
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate