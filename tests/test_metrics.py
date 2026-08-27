from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

from fusion_speech.metrics import MetricsRegistry, render_prometheus

# E2: zero-dependency metrics. Tests cover the primitives (counter monotonic,
# gauge set/inc/dec, histogram bucketing), the Prometheus text exposition
# (HELP/TYPE lines, label format, histogram _bucket/_sum/_count), and the
# dispatch wiring (rpc_requests_total + rpc_duration_seconds + rpc_errors_total
# labeled by method), plus /metrics endpoint + handle_status summary.
#
# The daemon wiring imports `metrics` BY NAME (`from fusion_speech.metrics
# import metrics`) into jsonrpc / each component / stt_stream, and
# handle_status imports it lazily. A fresh registry must be swapped into the
# canonical singleton AND every by-name importer, then restored. The helper
# below does both so a test's increments land on the test registry and
# render_prometheus (which reads the module global) sees them.


_BY_NAME_IMPORTERS = (
    "fusion_speech.ipc.jsonrpc",
    "fusion_speech.ipc.methods",
    "fusion_speech.components.event_bus",
    "fusion_speech.components.asr_service",
    "fusion_speech.components.wake_service",
    "fusion_speech.components.tts_service",
    "fusion_speech.stt_stream",
    "fusion_speech.server",
)


@contextlib.contextmanager
def _swapped_registry(reg: MetricsRegistry):
    import importlib

    import fusion_speech.metrics as metrics_mod

    real = metrics_mod.metrics
    metrics_mod.metrics = reg
    saved = {}
    for modname in _BY_NAME_IMPORTERS:
        mod = importlib.import_module(modname)
        if hasattr(mod, "metrics"):
            saved[modname] = mod.metrics
            mod.metrics = reg
    try:
        yield
    finally:
        metrics_mod.metrics = real
        for modname, val in saved.items():
            mod = importlib.import_module(modname)
            mod.metrics = val


def test_counter_monotonic():
    reg = MetricsRegistry()
    reg.rpc_requests_total.inc(labels={"method": "speech.ping"})
    reg.rpc_requests_total.inc(2.0, labels={"method": "speech.ping"})
    reg.rpc_requests_total.inc(labels={"method": "speech.status"})
    snap = reg.rpc_requests_total.snapshot()
    assert snap[("speech.ping",)] == 3.0
    assert snap[("speech.status",)] == 1.0


def test_gauge_set_inc_dec():
    reg = MetricsRegistry()
    reg.subscribers.set(2)
    reg.subscribers.inc(3)
    reg.subscribers.dec(1)
    assert reg.subscribers.snapshot()[()] == 4.0


def test_histogram_bucketing():
    reg = MetricsRegistry()
    reg.rpc_duration_seconds.observe(0.03, labels={"method": "speech.ping"})
    reg.rpc_duration_seconds.observe(0.2, labels={"method": "speech.ping"})
    snap = reg.rpc_duration_seconds.snapshot()
    st = snap[("speech.ping",)]
    assert st["count"] == 2
    assert st["sum"] == pytest.approx(0.23)
    buckets = st["buckets"]
    b = list(reg.rpc_duration_seconds.buckets)
    assert buckets[b.index(0.05)] == 1
    assert buckets[b.index(0.25)] == 2


def test_render_prometheus_exposition():
    reg = MetricsRegistry()
    reg.rpc_requests_total.inc(labels={"method": "speech.ping"})
    reg.listening.set(1)
    with _swapped_registry(reg):
        out = render_prometheus()
    assert "# HELP rpc_requests_total" in out
    assert "# TYPE rpc_requests_total counter" in out
    assert 'rpc_requests_total{method="speech.ping"} 1.0' in out
    assert "# TYPE listening gauge" in out
    assert "listening 1.0" in out
    assert "# TYPE uptime_seconds gauge" in out
    assert "process_start_time_seconds " in out


def test_render_empty_histogram_has_inf_bucket():
    reg = MetricsRegistry()
    with _swapped_registry(reg):
        out = render_prometheus()
    assert 'asr_transcribe_seconds_bucket{le="+Inf"} 0' in out
    assert "asr_transcribe_seconds_count 0" in out


@pytest.mark.asyncio
async def test_dispatch_wires_rpc_counters():
    # dispatch must inc rpc_requests_total{method} and observe duration on the
    # success path, and inc rpc_errors_total{method} on the error path.
    reg = MetricsRegistry()
    from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher

    async def ok_handler(params):
        return {"pong": True}

    async def bad_handler(params):
        raise ValueError("boom")

    dispatcher = JsonRpcDispatcher({"speech.ok": ok_handler, "speech.bad": bad_handler})
    with _swapped_registry(reg):
        await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.ok"})
        await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.bad"})
        req = reg.rpc_requests_total.snapshot()
        err = reg.rpc_errors_total.snapshot()
        dur = reg.rpc_duration_seconds.snapshot()
    assert req[("speech.ok",)] == 1.0
    assert req[("speech.bad",)] == 1.0
    assert err[("speech.bad",)] == 1.0
    assert ("speech.ok",) not in err
    assert dur[("speech.ok",)]["count"] == 1
    assert dur[("speech.bad",)]["count"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_and_status():
    # Build the facade app, fire pings through dispatch, GET /metrics and call
    # handle_status — both must reflect the counted request on the swapped
    # registry.
    reg = MetricsRegistry()
    from fusion_speech.config import load_config
    from fusion_speech.ipc.methods import handle_status
    from fusion_speech.server import SpeechDaemon

    with _swapped_registry(reg):
        daemon = SpeechDaemon(load_config())
        dispatcher = daemon._dispatcher
        for i in range(3):
            await dispatcher.dispatch({"jsonrpc": "2.0", "id": i, "method": "speech.ping"})
        app = daemon.build_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        assert 'rpc_requests_total{method="speech.ping"} 3.0' in body, body
        assert "# TYPE rpc_duration_seconds histogram" in body
        status = await handle_status({}, daemon)
        assert status["metrics"]["rpc_requests_total"]["('speech.ping',)"] == 3.0
        assert status["metrics"]["uptime_seconds"] >= 0
