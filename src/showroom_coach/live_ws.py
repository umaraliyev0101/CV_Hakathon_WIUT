"""The `/ws/live` protocol, as a handler the web app mounts: `@app.websocket("/ws/live")` calls `serve_live`.

    client -> server  binary   one JPEG frame; the server answers with exactly one text message: the overlay
                               (`{"type": "overlay", ...}`), or `{"type": "error", "error": "bad frame"}` when the bytes
                               are not an image or are larger than `MAX_FRAME_BYTES` (refused without decoding). The
                               session carries on after either.
    client -> server  text     a JSON control: toggle {pid}, name {pid, name}, preset {name}, time {t} (the timestamp of
                               the next frame only, used by file-fed tests) and summary. Controls get no reply except
                               `summary` (`{"type": "summary", ...}`); a control the server cannot act on gets
                               `{"type": "error", "error": ...}` instead, and the session carries on.

The client sends the next frame only after the reply (stop-and-wait), so frames that arrive while the CPU is busy are
dropped on the client and never queue here. Even so, the loop below handles one message at a time, in order.

Each connection builds its own `LiveSession` with `session_factory(preset)` (the preset is the `?preset=` query
parameter, "short" by default); the models behind it are shared between connections. Inference runs in a worker thread,
and so does every control (a preset change replays the whole session, a summary scores it), so the event loop keeps
serving other connections and the REST API meanwhile. A dropped
connection ends the handler quietly.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import PRESETS
from .live import LiveSession
from .sources import VideoError, decode_jpeg

log = logging.getLogger(__name__)

__all__ = ["serve_live"]

DEFAULT_PRESET = "short"
MAX_NAME_CHARS = 40
MAX_FRAME_BYTES = 2 * 1024 * 1024    # a 960 px JPEG is far smaller; anything bigger is refused without decoding

E_BAD_FRAME = {"type": "error", "error": "bad frame"}
E_FRAME_FAILED = {"type": "error", "error": "frame could not be processed"}
E_BAD_CONTROL = {"type": "error", "error": "bad control"}


def _error(message: str) -> dict:
    return {"type": "error", "error": message}


def _process_frame(session: LiveSession, data: bytes, t: float | None) -> dict:
    """Decode and process one frame (runs in a worker thread). Never raises."""
    if len(data) > MAX_FRAME_BYTES:
        return E_BAD_FRAME
    try:
        frame = decode_jpeg(data)
    except VideoError:
        return E_BAD_FRAME
    try:
        return session.process(frame, t)
    except Exception:  # a model or engine failure on one frame must not end the stream
        log.exception("live frame failed")
        return E_FRAME_FAILED


def _pid(msg: dict) -> int | None:
    pid = msg.get("pid")
    return pid if isinstance(pid, int) and not isinstance(pid, bool) else None


def _control(session: LiveSession, text: str, next_t: float | None) -> tuple[dict | None, float | None]:
    """Apply one text control. Returns `(reply or None, timestamp override for the next frame)`. Never raises."""
    try:
        return _apply(session, text, next_t)
    except Exception:
        log.exception("live control failed")
        return E_BAD_CONTROL, next_t


def _apply(session: LiveSession, text: str, next_t: float | None) -> tuple[dict | None, float | None]:
    try:
        msg = json.loads(text)
    except ValueError:
        return E_BAD_CONTROL, next_t
    if not isinstance(msg, dict):
        return E_BAD_CONTROL, next_t
    kind = msg.get("type")
    if kind == "summary":
        return session.summary(), next_t
    if kind == "time":
        t = msg.get("t")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            return E_BAD_CONTROL, next_t
        return None, float(t)
    if kind == "preset":
        name = msg.get("name")
        if not isinstance(name, str) or name not in PRESETS:
            return _error("unknown preset"), next_t
        session.set_preset(name)
        return None, next_t
    if kind in ("toggle", "name"):
        pid = _pid(msg)
        name = msg.get("name")
        if pid is None or (kind == "name" and not isinstance(name, str)):
            return E_BAD_CONTROL, next_t
        try:
            if kind == "toggle":
                session.toggle(pid)
            else:
                session.set_name(pid, name[:MAX_NAME_CHARS])
        except KeyError:
            return _error("unknown person"), next_t
        return None, next_t
    return E_BAD_CONTROL, next_t


async def serve_live(websocket: WebSocket, session_factory: Callable[[str], Any]) -> None:
    """Serve one `/ws/live` connection until the client goes away. `session_factory(preset)` returns the connection's
    `LiveSession`. An unknown preset is refused before the factory runs (an error message, then the socket closes);
    any failure inside the factory is reported as "live mode is unavailable"."""
    preset = websocket.query_params.get("preset") or DEFAULT_PRESET
    await websocket.accept()
    try:
        if preset not in PRESETS:
            await websocket.send_json(_error(f"unknown preset {preset!r}"))
            await websocket.close(code=1008)
            return
        try:
            session = await run_in_threadpool(session_factory, preset)
        except Exception:
            log.exception("cannot start a live session")
            await websocket.send_json(_error("live mode is unavailable"))
            await websocket.close(code=1011)
            return

        next_t: float | None = None
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data, text = message.get("bytes"), message.get("text")
            if data is not None:
                t, next_t = next_t, None
                reply = await run_in_threadpool(_process_frame, session, data, t)
            elif text is not None:
                reply, next_t = await run_in_threadpool(_control, session, text, next_t)
            else:
                continue
            if reply is not None:
                await websocket.send_json(reply)
    except (WebSocketDisconnect, RuntimeError):
        # Only socket I/O can raise these here (frames and controls are handled without raising): the client dropped
        # while we were sending, or the socket was already closed. Nothing left to do.
        return
