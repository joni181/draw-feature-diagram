"""ID generation utilities for feature entries."""

from __future__ import annotations

import re
from typing import Iterable


_DISALLOWED_RE = re.compile(r"[^a-z0-9_ ]+")
_SPACE_RE = re.compile(r"\s+")


def slugify_feature_name(name: str) -> str:
    """Create a base feature ID from the name."""
    normalized = _DISALLOWED_RE.sub("", name.strip().lower())
    normalized = _SPACE_RE.sub("_", normalized)
    normalized = normalized.strip("_")
    return normalized or "feature"


def suggest_unique_id(name: str, existing_ids: Iterable[str]) -> str:
    """Suggest a unique feature ID from name and existing IDs."""
    base = slugify_feature_name(name)
    existing = set(existing_ids)
    if base not in existing:
        return base
    index = 1
    while f"{base}-{index}" in existing:
        index += 1
    return f"{base}-{index}"

