from __future__ import annotations

import contextlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# E6: persistent daemon state across restarts. state.json lives next to the UDS
# in ~/.fusion-speech/run so a crash-then-restart can report restart_count and
# broadcast speech.restarted. All fields are non-sensitive metadata (no audio,
# no transcript text); last_wake_word is the configured/configured word, not a
# captured utterance. Atomic write via tmp+rename; corrupt file -> start fresh.


def _state_dir(socket_path: str) -> str:
    return os.path.dirname(socket_path) or os.path.expanduser("~/.fusion-speech/run")


def _state_path(socket_path: str) -> str:
    return os.path.join(_state_dir(socket_path), "state.json")


class DaemonState:
    # Thin persistence layer: load() reads (or starts fresh), save() writes
    # atomically. Fields: start_time, restart_count, last_wake_word,
    # uptime_at_stop. The facade calls enter_start() on boot (bump
    # restart_count, persist) and enter_stop() on graceful shutdown (record
    # uptime_at_stop, persist).
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.path = _state_path(socket_path)
        self.start_time: float = 0.0
        self.restart_count: int = 0
        self.last_wake_word: str = ""
        self.uptime_at_stop: float = 0.0

    def load(self) -> dict:
        # Returns the persisted dict (or {} if missing/corrupt). Side-effect:
        # populates self.* from disk so a restart retains restart_count.
        data: dict = {}
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError, ValueError), open(
            self.path, encoding="utf-8"
        ) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("state.json not a dict, starting fresh")
            data = {}
        self.restart_count = int(data.get("restart_count", 0))
        self.last_wake_word = str(data.get("last_wake_word", ""))
        self.uptime_at_stop = float(data.get("uptime_at_stop", 0.0))
        return data

    def enter_start(self) -> int:
        # Called on daemon boot. Bumps restart_count (every start counts,
        # including crash recovery), stamps start_time, persists.
        self.load()
        self.restart_count += 1
        self.start_time = time.time()
        self._save()
        logger.info(
            "state: start restart_count=%d start_time=%.0f prev_uptime=%.1f",
            self.restart_count,
            self.start_time,
            self.uptime_at_stop,
        )
        return self.restart_count

    def enter_stop(self) -> None:
        # Called on graceful shutdown. Records uptime, keeps restart_count.
        if self.start_time:
            self.uptime_at_stop = time.time() - self.start_time
        self._save()
        logger.info("state: stop uptime_at_stop=%.1f", self.uptime_at_stop)

    def record_wake_word(self, word: str) -> None:
        self.last_wake_word = word
        self._save()

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "restart_count": self.restart_count,
            "last_wake_word": self.last_wake_word,
            "uptime_at_stop": self.uptime_at_stop,
        }

    def _save(self) -> None:
        os.makedirs(_state_dir(self.socket_path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)
