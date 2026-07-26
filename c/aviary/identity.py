"""Stable node UUID persisted beside the model directory."""

from __future__ import annotations

import uuid
from pathlib import Path

NODE_ID_FILE = ".aviary_node_id"


def node_id_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / NODE_ID_FILE


def load_or_create_node_id(model_dir: str | Path) -> str:
    path = node_id_path(model_dir)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    node_id = str(uuid.uuid4())
    path.write_text(node_id + "\n", encoding="utf-8")
    return node_id
