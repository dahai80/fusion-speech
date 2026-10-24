from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import struct
import time
from typing import Any

from fusion_speech.metrics import metrics

logger = logging.getLogger(__name__)

# macOS LOCAL_PEERCRED (SOL_LOCAL=0); Linux SO_PEERCRED (SOL_SOCKET=1, opt=17).
# Fail-closed if getsockopt fails: reject connection rather than allow (audit
# lesson from fusion-agent-studio #209).
_SOL_LOCAL = 0
_LOCAL_PEERCRED = 0x001
_SOL_SOCKET = 1
_SO_PEERCRED = 17

# Cap an un-terminated line (no newline yet). A client streaming with no \n
# would otherwise grow the buffer unbounded (audit F3 DoS). 1 MiB is far above
# any legitimate single JSON-RPC message (audio transcribe is base64'd client-
# side and bounded by methods.MAX_AUDIO_B64).
_MAX_LINE = 1 << 20

# H5: backstop timeout on every handler. transcribe edges 120s on mlx; a wedged
# handler (deadlock, runaway loop) gets -32000 instead of hanging forever.
# Module-level so tests monkeypatch it to a small value (a real 120s sleep would
# defeat the test) — production value stays 120s.
DISPATCH_TIMEOUT = 120.0


def _peer_uid_from_socket(sock) -> int | None:
    try:
        cred = sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERCRED, struct.calcsize("iII"))
        _, uid, _ = struct.unpack("iII", cred)
        return uid
    except (OSError, AttributeError):
        pass
    try:
        cred = sock.getsockopt(_SOL_SOCKET, _SO_PEERCRED, struct.calcsize("iII"))
        _, uid, _ = struct.unpack("iII", cred)
        return uid
    except (OSError, AttributeError):
        return None


def _verify_peer_uid(writer: asyncio.StreamWriter) -> bool:
    sock = writer.get_extra_info("socket")
    if sock is None:
        logger.warning("peer-uid check: no socket, fail-closed (reject)")
        return False
    uid = _peer_uid_from_socket(sock)
    if uid is None:
        logger.warning("peer-uid getsockopt failed, fail-closed (reject)")
        return False
    if uid != os.getuid():
        logger.warning("rejected non-owner UID %d on socket (owner %d)", uid, os.getuid())
        return False
    return True


def success_response(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error_response(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _format_handler_error(e: Exception) -> str:
    msg = str(e).strip()
    status = getattr(e, "status", 0)
    if not msg:
        return f"{type(e).__name__} (status {status})" if status else type(e).__name__
    if status:
        return f"{type(e).__name__} (status {status}): {msg}"
    return f"{type(e).__name__}: {msg}"


class JsonRpcDispatcher:
    def __init__(self, handlers: dict[str, Any]):
        self._handlers = handlers

    def get_handler(self, method: str):
        return self._handlers.get(method)

    async def dispatch(self, message: dict) -> dict:
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except (ValueError, TypeError):
                params = {}
            if not isinstance(params, dict):
                params = {}
        if message.get("jsonrpc") != "2.0":
            return error_response(msg_id, -32600, "Invalid Request: missing jsonrpc 2.0")
        if not method:
            return error_response(msg_id, -32601, "Method not found: empty method")
        handler = self.get_handler(method)
        if handler is None:
            return error_response(msg_id, -32601, f"Method not found: {method}")
        # E2: per-method request counter + duration histogram. Fail-loud
        # counter is separate (errors_total) so /metrics can compute success
        # rate. Timeout counts as an error.
        metrics.rpc_requests_total.inc(labels={"method": method})
        started = time.monotonic()
        try:
            # H5: backstop timeout so a wedged handler (deadlock, runaway loop)
            # cannot hold a client connection forever. transcribe edges 120s on
            # mlx; -32000 timeout surfaces to the client rather than hanging.
            # Per-Service locks (Wave 3) prevent the state-mutation races; this
            # only bounds wall time.
            result = await asyncio.wait_for(handler(params), timeout=DISPATCH_TIMEOUT)
            metrics.rpc_duration_seconds.observe(time.monotonic() - started, labels={"method": method})
            return success_response(msg_id, result)
        except TimeoutError:
            metrics.rpc_errors_total.inc(labels={"method": method})
            metrics.rpc_duration_seconds.observe(time.monotonic() - started, labels={"method": method})
            logger.warning("handler %s timed out after %ss", method, DISPATCH_TIMEOUT)
            return error_response(msg_id, -32000, f"handler timeout: {method}")
        except Exception as e:
            metrics.rpc_errors_total.inc(labels={"method": method})
            metrics.rpc_duration_seconds.observe(time.monotonic() - started, labels={"method": method})
            logger.exception("handler error for %s", method)
            return error_response(msg_id, -32000, _format_handler_error(e))


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    dispatcher: JsonRpcDispatcher,
    running_fn,
    notify_sink=None,
    *,
    verify_fn=None,
) -> None:
    verify = verify_fn if verify_fn is not None else _verify_peer_uid
    peer = writer.get_extra_info("peername")
    if not verify(writer):
        logger.warning("closing non-owner connection: %s", peer)
        writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await writer.wait_closed()
        return
    logger.info("client connected: %s", peer)
    buf = bytearray()
    try:
        while running_fn():
            data = await reader.read(65536)
            if not data:
                break
            buf += data
            if len(buf) > _MAX_LINE and b"\n" not in buf:
                logger.warning("client %s exceeded line limit %d with no newline, closing", peer, _MAX_LINE)
                break
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                    response = await dispatcher.dispatch(message)
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
                except json.JSONDecodeError as e:
                    err = error_response(None, -32700, f"Parse error: {e}")
                    writer.write(json.dumps(err).encode() + b"\n")
                    await writer.drain()
                except Exception as e:
                    logger.exception("client handler dispatch error")
                    err = error_response(None, -32603, f"Internal error: {e}")
                    writer.write(json.dumps(err).encode() + b"\n")
                    await writer.drain()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("client handler error: %s", e)
    finally:
        writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await writer.wait_closed()
        logger.info("client disconnected: %s", peer)
