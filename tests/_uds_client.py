from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class UdsJsonRpcClient:
    # Real UDS JSON-RPC 2.0 client (newline-delimited). Connects to the live
    # daemon socket; one request -> one response, matched by id.

    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._id = 0

    async def connect(self) -> None:
        # Synthesized WAV base64 can exceed the default 64KiB readline limit;
        # raise the limit so large RPC responses (transcribe/synthesize) read whole.
        logger.info("uds-client: connect %s", self.socket_path)
        self._reader, self._writer = await asyncio.open_unix_connection(self.socket_path, limit=2**22)

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        logger.info("uds-client: closed")

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._writer is None or self._reader is None:
            raise RuntimeError("uds-client: not connected")
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            req["params"] = params
        line = (json.dumps(req) + "\n").encode()
        logger.info("uds-client: -> %s %s", method, params or {})
        self._writer.write(line)
        await self._writer.drain()
        raw = await asyncio.wait_for(self._reader.readline(), timeout=self.timeout)
        if not raw:
            raise RuntimeError("uds-client: empty response (daemon closed)")
        resp = json.loads(raw.decode())
        logger.info("uds-client: <- %s", resp)
        return resp

    async def __aenter__(self) -> UdsJsonRpcClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def default_socket_path() -> str:
    sock_dir = os.environ.get("FUSION_SOCKET_DIR", "").strip()
    if not sock_dir:
        sock_dir = os.path.expanduser("~/.fusion-speech/run")
    return os.path.join(sock_dir, "fusion-speech.sock")
