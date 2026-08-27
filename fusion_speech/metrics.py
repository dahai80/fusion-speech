from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# E2: zero-dependency metrics (no prometheus_client). dict-backed Counter /
# Gauge / Histogram, GIL-safe under the single asyncio loop + occasional
# /metrics scrape. Prometheus text exposition via render_prometheus().

# Default histogram buckets (seconds) — RPC + transcribe latency profile.
_DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)


class _Counter:
    # Monotonic increment per labelset. float values so inc(n) works for any n.
    def __init__(self, name: str, help_: str, labelnames: tuple[str, ...] = ()):
        self.name = name
        self.help = help_
        self.labelnames = labelnames
        self._values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict | None) -> tuple:
        if not self.labelnames:
            return ()
        if labels is None:
            return ("",) * len(self.labelnames)
        return tuple(str(labels.get(n, "")) for n in self.labelnames)

    def inc(self, n: float = 1.0, labels: dict | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + n

    def snapshot(self) -> dict[tuple, float]:
        with self._lock:
            return dict(self._values)


class _Gauge:
    # Set/inc/dec per labelset. Non-monotonic (listening/subscribers).
    def __init__(self, name: str, help_: str, labelnames: tuple[str, ...] = ()):
        self.name = name
        self.help = help_
        self.labelnames = labelnames
        self._values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict | None) -> tuple:
        if not self.labelnames:
            return ()
        if labels is None:
            return ("",) * len(self.labelnames)
        return tuple(str(labels.get(n, "")) for n in self.labelnames)

    def set(self, value: float, labels: dict | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, n: float = 1.0, labels: dict | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + n

    def dec(self, n: float = 1.0, labels: dict | None = None) -> None:
        self.inc(-n, labels)

    def snapshot(self) -> dict[tuple, float]:
        with self._lock:
            return dict(self._values)


class _Histogram:
    # Latency distribution: bucket counts + sum + count per labelset.
    def __init__(self, name: str, help_: str, labelnames: tuple[str, ...] = (), buckets=_DEFAULT_BUCKETS):
        self.name = name
        self.help = help_
        self.labelnames = labelnames
        self.buckets = tuple(buckets)
        # per-labelset: {"count": int, "sum": float, "buckets": [counts...]}
        self._state: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict | None) -> tuple:
        if not self.labelnames:
            return ()
        if labels is None:
            return ("",) * len(self.labelnames)
        return tuple(str(labels.get(n, "")) for n in self.labelnames)

    def observe(self, value: float, labels: dict | None = None) -> None:
        key = self._key(labels)
        v = float(value)
        with self._lock:
            st = self._state.get(key)
            if st is None:
                st = {"count": 0, "sum": 0.0, "buckets": [0] * len(self.buckets)}
                self._state[key] = st
            st["count"] += 1
            st["sum"] += v
            for i, b in enumerate(self.buckets):
                if v <= b:
                    st["buckets"][i] += 1

    def snapshot(self) -> dict[tuple, dict]:
        with self._lock:
            # deep-copy the per-labelset dicts so callers can iterate unlocked
            out: dict[tuple, dict] = {}
            for key, st in self._state.items():
                out[key] = {"count": st["count"], "sum": st["sum"], "buckets": list(st["buckets"])}
            return out


class MetricsRegistry:
    # Holds the core metrics. Single shared instance `metrics` below. Registry
    # methods are the public inc/observe/set surface used across the daemon.
    def __init__(self):
        self.start_time = time.time()
        self.rpc_requests_total = _Counter(
            "rpc_requests_total", "Total RPC requests by method.", ("method",)
        )
        self.rpc_errors_total = _Counter(
            "rpc_errors_total", "Total RPC handler errors by method.", ("method",)
        )
        self.rpc_duration_seconds = _Histogram(
            "rpc_duration_seconds", "RPC handler duration in seconds.", ("method",)
        )
        self.asr_segments_total = _Counter(
            "asr_segments_total", "Total ASR segments closed (VAD segment-close)."
        )
        self.asr_transcribe_seconds = _Histogram(
            "asr_transcribe_seconds", "ASR per-segment transcribe latency in seconds."
        )
        self.asr_segments_dropped_total = _Counter(
            "asr_segments_dropped_total", "ASR segments dropped by R4 backpressure."
        )
        self.wake_fires_total = _Counter(
            "wake_fires_total", "Total wake-word fires."
        )
        self.tts_speak_total = _Counter(
            "tts_speak_total", "Total TTS speak invocations."
        )
        self.subscribers = _Gauge(
            "subscribers", "Current WS/event subscribers."
        )
        self.listening = _Gauge(
            "listening", "1 when ASR listener active, else 0."
        )
        self.wake_armed = _Gauge(
            "wake_armed", "1 when wake engine armed, else 0."
        )
        self.speaking = _Gauge(
            "speaking", "1 when TTS player active, else 0."
        )

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


metrics = MetricsRegistry()


def _format_labels(labelnames: tuple[str, ...], key: tuple) -> str:
    if not labelnames:
        return ""
    pairs = [f'{n}="{v}"' for n, v in zip(labelnames, key, strict=True)]
    return "{" + ",".join(pairs) + "}"


def render_prometheus() -> str:
    # Prometheus text exposition format (0.0.4). One TYPE + HELP per metric,
    # then one sample line per labelset. Histograms emit bucket cumulative
    # counts + _sum + _count.
    lines: list[str] = []
    reg = metrics

    def emit_counter(c: _Counter) -> None:
        lines.append(f"# HELP {c.name} {c.help}")
        lines.append(f"# TYPE {c.name} counter")
        snap = c.snapshot()
        if not snap:
            lines.append(f"{c.name} 0")
            return
        for key, val in sorted(snap.items()):
            lines.append(f"{c.name}{_format_labels(c.labelnames, key)} {val}")

    def emit_gauge(g: _Gauge) -> None:
        lines.append(f"# HELP {g.name} {g.help}")
        lines.append(f"# TYPE {g.name} gauge")
        snap = g.snapshot()
        if not snap:
            lines.append(f"{g.name} 0")
            return
        for key, val in sorted(snap.items()):
            lines.append(f"{g.name}{_format_labels(g.labelnames, key)} {val}")

    def emit_histogram(h: _Histogram) -> None:
        lines.append(f"# HELP {h.name} {h.help}")
        lines.append(f"# TYPE {h.name} histogram")
        snap = h.snapshot()
        if not snap:
            lines.append(f'{h.name}_bucket{{le="+Inf"}} 0')
            lines.append(f"{h.name}_sum 0")
            lines.append(f"{h.name}_count 0")
            return
        for key, st in sorted(snap.items()):
            base = _format_labels(h.labelnames, key)
            for i, b in enumerate(h.buckets):
                le = f'le="{b}"'
                label = f"{{{le}}}" if not base else f"{{{base[1:-1]},{le}}}"
                lines.append(f"{h.name}_bucket{label} {st['buckets'][i]}")
            inf = 'le="+Inf"'
            label = f"{{{inf}}}" if not base else f"{{{base[1:-1]},{inf}}}"
            lines.append(f"{h.name}_bucket{label} {st['count']}")
            lines.append(f"{h.name}_sum{base} {st['sum']}")
            lines.append(f"{h.name}_count{base} {st['count']}")

    emit_counter(reg.rpc_requests_total)
    emit_counter(reg.rpc_errors_total)
    emit_histogram(reg.rpc_duration_seconds)
    emit_counter(reg.asr_segments_total)
    emit_histogram(reg.asr_transcribe_seconds)
    emit_counter(reg.asr_segments_dropped_total)
    emit_counter(reg.wake_fires_total)
    emit_counter(reg.tts_speak_total)
    emit_gauge(reg.subscribers)
    emit_gauge(reg.listening)
    emit_gauge(reg.wake_armed)
    emit_gauge(reg.speaking)
    lines.append("# HELP uptime_seconds Daemon uptime in seconds.")
    lines.append("# TYPE uptime_seconds gauge")
    lines.append(f"uptime_seconds {reg.uptime_seconds()}")
    lines.append("# HELP process_start_time_seconds Daemon start time (unix epoch).")
    lines.append("# TYPE process_start_time_seconds gauge")
    lines.append(f"process_start_time_seconds {reg.start_time}")
    return "\n".join(lines) + "\n"
