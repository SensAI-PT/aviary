"""Line-based cluster control-plane framing (see docs/cluster_protocol.md)."""

from __future__ import annotations

import json
import os
import socket
from typing import Any

DEFAULT_RPC_TIMEOUT_MS = int(os.environ.get("AVIARY_RPC_TIMEOUT_MS", "150"))


class ProtocolError(Exception):
    pass


def _read_line(conn: socket.socket, timeout_sec: float | None) -> str | None:
    if timeout_sec is not None:
        conn.settimeout(timeout_sec)
    buf = bytearray()
    while True:
        try:
            chunk = conn.recv(1)
        except socket.timeout:
            raise ProtocolError("read timeout") from None
        if not chunk:
            return None
        if chunk == b"\n":
            break
        buf.extend(chunk)
    return buf.decode("utf-8", "replace")


def _read_payload(conn: socket.socket, size: int, timeout_sec: float | None) -> bytes:
    if timeout_sec is not None:
        conn.settimeout(timeout_sec)
    chunks = []
    remaining = size
    while remaining:
        try:
            chunk = conn.recv(remaining)
        except socket.timeout:
            raise ProtocolError("payload read timeout") from None
        if not chunk:
            raise ProtocolError("truncated payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    term = conn.recv(1)
    if term != b"\n":
        raise ProtocolError("missing payload terminator")
    return data


def read_frame(conn: socket.socket, timeout_ms: int | None = None) -> tuple[str, list[str], Any | None]:
    timeout_sec = None if timeout_ms is None else timeout_ms / 1000.0
    line = _read_line(conn, timeout_sec)
    if line is None:
        return ("EOF", [], None)
    fields = line.strip().split()
    if not fields:
        return read_frame(conn, timeout_ms)
    kind = fields[0]
    payload = None
    if kind == "REGISTER" and len(fields) >= 5:
        size = int(fields[4])
        raw = _read_payload(conn, size, timeout_sec)
        payload = json.loads(raw.decode("utf-8"))
    elif kind == "HEARTBEAT" and len(fields) >= 4:
        size = int(fields[3])
        raw = _read_payload(conn, size, timeout_sec)
        payload = json.loads(raw.decode("utf-8"))
    elif kind == "PLACEMENT" and len(fields) >= 2:
        size = int(fields[1])
        raw = _read_payload(conn, size, timeout_sec)
        payload = json.loads(raw.decode("utf-8"))
    return (kind, fields, payload)


def write_line(conn: socket.socket, line: str) -> None:
    conn.sendall((line.rstrip("\n") + "\n").encode("utf-8"))


def encode_payload_frame(header: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{header} {len(raw)}\n{raw.decode('utf-8')}\n"


def send_frame(conn: socket.socket, line: str, payload: dict[str, Any] | None = None) -> None:
    if payload is None:
        write_line(conn, line)
        return
    conn.sendall(encode_payload_frame(line, payload).encode("utf-8"))


def register_frame(node_id: str, http_port: int, model_id: str, payload: dict[str, Any]) -> str:
    return encode_payload_frame(f"REGISTER {node_id} {http_port} {model_id}", payload)


def heartbeat_frame(node_id: str, inflight: int, payload: dict[str, Any]) -> str:
    return encode_payload_frame(f"HEARTBEAT {node_id} {inflight}", payload)


def placement_frame(payload: dict[str, Any]) -> str:
    return encode_payload_frame("PLACEMENT", payload)
