"""Shared tier snapshot sanitization for agents and master registry."""

from __future__ import annotations

from typing import Any


def sanitize_tiers(tiers: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop static plan placeholders and absurd engine tier counts."""
    if not tiers:
        return None
    total = int(tiers.get("vram") or 0) + int(tiers.get("ram") or 0) + int(tiers.get("disk") or 0)
    if total > 1_000_000:
        return None
    ram_gb = float(tiers.get("ram_gb") or 0)
    vram_gb = float(tiers.get("vram_gb") or 0)
    if ram_gb > 1024 or vram_gb > 1024:
        cleaned = dict(tiers)
        if ram_gb > 1024:
            cleaned["ram_gb"] = 0.0
        if vram_gb > 1024:
            cleaned["vram_gb"] = 0.0
        return cleaned
    return dict(tiers)
