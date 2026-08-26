from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import struct
from typing import Any

logger = logging.getLogger(__name__)

# macOS LOCAL_PEERCRED (SOL_LOCAL=0); Linux SO_PEERCRED (SOL_SOCKET=1, opt=17).
# Fail-closed if getsockopt fails: reject connection rather than allow (audit
# lesson from fusion-agent-studio #209).
_SOL_LOCAL = 0
_LOCAL_PEERCRED = 0x001
_SOL_SOCKET = 1
_SO_PEERCRED = 17


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
        logger.warning("peer-uid check: no socket, fail-open (test path)")
        return True
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
        try:
            result = await handler(params)
            return success_response(msg_id, result)
        except Exception as e:
            logger.exception("handler error for %s", method)
            return error_response(msg_id, -32000, _format_handler_error(e))


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    dispatcher: JsonRpcDispatcher,
    running_fn,
    notify_sink=None,
) -> None:
    peer = writer.get_extra_info("peername")
    if not _verify_peer_uid(writer):
        logger.warning("closing non-owner connection: %s", peer)
        writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await writer.wait_closed()
        return
    logger.info("client connected: %s", peer)
    buf = b""
    try:
        while running_fn():
            data = await reader.read(65536)
            if not data:
                break
            buf += data
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
