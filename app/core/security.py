from __future__ import annotations

from pathlib import Path

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_admin_key(x_api_key: str = Header(...)) -> str:
    if not settings.ADMIN_API_KEY or x_api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return x_api_key


def validate_file_path(path: str) -> Path:
    """Ensure *path* falls under one of the allowed input roots."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    roots = settings.allowed_roots
    if not roots:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No ALLOWED_INPUT_ROOTS configured — file_path mode disabled",
        )
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Path is outside allowed roots",
    )
