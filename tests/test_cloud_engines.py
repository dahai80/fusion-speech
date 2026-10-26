from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fusion_speech.cloud import (
    CLOUD_RETRY_BACKOFFS,
    CloudEngineError,
    StaticCredentials,
    cloud_call,
    cloud_retry,
    config_azure_credentials,
    wav_to_pcm,
)
from fusion_speech.cloud_azure import (
    AzureSTTEngine,
    AzureTTSEngine,
    parse_language_candidates,
    register_azure_engines,
)
from fusion_speech.engines import asr_engine_transcribe_adapter, make_asr_engine, make_tts_engine
from fusion_speech.metrics import metrics
from fusion_speech.stt_stream import pcm_to_wav


class Cfg:
    # Minimal config stub: cloud engines read azure_* + sample_rate only.
    sample_rate = 16000
    language = "en"
    azure_key = "test-key"
    azure_region = "eastus"
    azure_language = "en-US"
    azure_stt_languages = ""
    azure_tts_voice = "en-US-AriaNeural"


def _tone_pcm(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# --- SECTION: pure helpers ---


def test_parse_language_candidates():
    assert parse_language_candidates("ar-EG, ar-SA ,en-US") == ["ar-EG", "ar-SA", "en-US"]
    assert parse_language_candidates("") == []
    assert parse_language_candidates(" , ") == []


def test_cloud_engine_error_embeds_reason():
    e = CloudEngineError("boom", reason="azure_auth", retryable=False)
    assert "reason=azure_auth retryable=False" in str(e)
    assert e.reason == "azure_auth"
    assert e.retryable is False


async def test_static_credentials_fail_closed():
    provider = StaticCredentials("", "")
    with pytest.raises(CloudEngineError) as ei:
        await provider()
    assert ei.value.reason == "azure_not_configured"


async def test_config_credentials_passthrough():
    provider = config_azure_credentials(Cfg())
    assert await provider() == {"key": "test-key", "region": "eastus"}


def test_wav_to_pcm_roundtrip_and_rejection():
    pcm = _tone_pcm(0.25)
    wav = pcm_to_wav(pcm, 16000)
    out = wav_to_pcm(wav)
    assert out.shape == pcm.shape
    assert np.allclose(out, pcm, atol=1e-4)
    with pytest.raises(CloudEngineError) as ei:
        wav_to_pcm(b"not a wav file")
    assert ei.value.reason == "invalid_audio"
    bad = pcm_to_wav(pcm, 8000)  # violates 16k contract
    with pytest.raises(CloudEngineError) as ei:
        wav_to_pcm(bad)
    assert ei.value.reason == "invalid_audio"


# --- SECTION: cloud_retry / cloud_call ---


async def test_cloud_retry_retries_retryable(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise CloudEngineError("transient", reason="azure_unreachable", retryable=True)
        return "ok"

    assert await cloud_retry("test", flaky) == "ok"
    assert calls["n"] == 3
    assert sleeps == list(CLOUD_RETRY_BACKOFFS[:2])


async def test_cloud_retry_raises_non_retryable(monkeypatch):
    async def fake_sleep(d):
        raise AssertionError("must not sleep")

    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise CloudEngineError("no key", reason="azure_auth", retryable=False)

    with pytest.raises(CloudEngineError) as ei:
        await cloud_retry("test", bad)
    assert ei.value.reason == "azure_auth"
    assert calls["n"] == 1


async def test_cloud_retry_gives_up_after_max():
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise CloudEngineError("down", reason="azure_unreachable", retryable=True)

    with pytest.raises(CloudEngineError):
        await cloud_retry("test", always_fail)
    assert calls["n"] == len(CLOUD_RETRY_BACKOFFS) + 1


async def test_cloud_call_metrics():
    before_req = metrics.cloud_requests_total.snapshot().get(("azure", "stt"), 0.0)
    before_err = metrics.cloud_errors_total.snapshot().get(("azure", "stt", "azure_auth"), 0.0)

    async with cloud_call("stt", "azure"):
        pass
    assert metrics.cloud_requests_total.snapshot().get(("azure", "stt"), 0.0) == before_req + 1

    with pytest.raises(CloudEngineError):
        async with cloud_call("stt", "azure"):
            raise CloudEngineError("bad key", reason="azure_auth", retryable=False)
    assert metrics.cloud_errors_total.snapshot().get(("azure", "stt", "azure_auth"), 0.0) == before_err + 1


async def _no_sleep(_d):
    return None


# --- SECTION: azure stt (fake sdk) ---


class FakePushStream:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    def close(self):
        self.closed = True


class FakeRecognizer:
    # Records ctor args so tests assert on LID vs fixed-language wiring.
    last = None

    def __init__(self, speech_config, auto_detect=None, audio_config=None, language=None):
        self.speech_config = speech_config
        self.auto_detect = auto_detect
        self.audio_config = audio_config
        self.language = language
        self.push_stream = audio_config.stream if audio_config else None
        FakeRecognizer.last = self
        # Each recognizer instance consumes the next queued result: retry
        # tests seed [failure, success] and assert one re-attempt happened.
        self.result_to_return = _RESULT_QUEUE.pop(0) if _RESULT_QUEUE else None
        self.exc_to_raise = _EXC_QUEUE.pop(0) if _EXC_QUEUE else None

    def recognize_once_async(self):
        outer = self

        class _Fut:
            def get(self):
                if outer.exc_to_raise is not None:
                    raise outer.exc_to_raise
                return outer.result_to_return

        return _Fut()


_RESULT_QUEUE: list = []
_EXC_QUEUE: list = []


class FakeSdk:
    # Surface used by AzureSTTEngine._recognize/_result_to_dict.
    ResultReason = SimpleNamespace(RecognizedSpeech=1, NoMatch=0)
    PropertyId = SimpleNamespace(
        SpeechServiceResponse_JsonResult="json_result_prop",
        SpeechServiceResponse_AutoDetectDetectedLanguage="lid_prop",
    )

    class SpeechConfig:
        def __init__(self, subscription, region):
            self.subscription = subscription
            self.region = region
            self.word_timestamps = False

        def request_word_level_timestamps(self):
            self.word_timestamps = True

    audio = SimpleNamespace(
        PushAudioInputStream=FakePushStream,
        AudioConfig=lambda stream: SimpleNamespace(stream=stream),
    )
    languageconfig = SimpleNamespace(
        AutoDetectSourceLanguageConfig=lambda languages: SimpleNamespace(languages=languages)
    )
    SpeechRecognizer = FakeRecognizer
    CancellationDetails = None  # set per-test via make_cancellation


def _fake_result(text="hello world", json_detail=None, lid="", reason=1):
    props = {}
    if json_detail is not None:
        props["json_result_prop"] = json.dumps(json_detail)
    if lid:
        props["lid_prop"] = lid
    return SimpleNamespace(reason=reason, text=text, properties=props)


def _make_cancellation(code, details="svc msg"):
    class _Details:
        error_code = code
        error_details = details

    FakeSdk.CancellationDetails = lambda result: _Details()


async def _run_stt(sdk, languages=None, pcm=None):
    engine = AzureSTTEngine(Cfg(), languages=languages, sdk=sdk)
    await engine.start_stream(language="")
    await engine.feed(pcm if pcm is not None else _tone_pcm(0.3))
    return await engine.finish()


async def test_stt_words_and_dialect_from_lid():
    detail = {
        "NBest": [
            {
                "Words": [
                    {"Word": "hello", "Offset": 500000, "Duration": 2500000, "Confidence": 0.97},
                    {"Word": "world", "Offset": 3000000, "Duration": 2000000, "Confidence": 0.91},
                ]
            }
        ]
    }
    _RESULT_QUEUE.append(_fake_result(text="hello world", json_detail=detail, lid="ar-EG"))
    result = await _run_stt(FakeSdk, languages=["ar-EG", "ar-SA", "en-US"])
    assert result["text"] == "hello world"
    assert result["engine"] == "azure"
    assert result["language"] == "ar-EG"
    assert result["dialect"] == "ar-EG"
    assert result["dialect_detected"] is True
    assert result["words"] == [
        {"word": "hello", "offset_ms": 50.0, "duration_ms": 250.0, "confidence": 0.97},
        {"word": "world", "offset_ms": 300.0, "duration_ms": 200.0, "confidence": 0.91},
    ]
    # SDK got the LID candidate list + word-timestamp request + full PCM
    rec = FakeRecognizer.last
    assert rec.auto_detect.languages == ["ar-EG", "ar-SA", "en-US"]
    assert rec.speech_config.word_timestamps is True
    assert rec.push_stream.closed is True
    assert len(rec.push_stream.written) > 0


async def test_stt_fixed_language_no_dialect():
    _RESULT_QUEUE.append(_fake_result(text="hi", json_detail={"NBest": [{"Words": []}]}))
    result = await _run_stt(FakeSdk)  # no LID candidates
    assert result["dialect"] == ""
    assert result["dialect_detected"] is False
    assert result["language"] == "en-US"  # config.azure_language default
    rec = FakeRecognizer.last
    assert rec.auto_detect is None
    assert rec.language == "en-US"


async def test_stt_no_match():
    _RESULT_QUEUE.append(_fake_result(reason=0))
    result = await _run_stt(FakeSdk)
    assert result["text"] == ""
    assert result["no_match"] is True


async def test_stt_auth_failure_non_retryable(monkeypatch):
    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", _no_sleep)
    _make_cancellation("AuthenticationFailure", "bad key")
    _RESULT_QUEUE.append(_fake_result(reason=2))
    with pytest.raises(CloudEngineError) as ei:
        await _run_stt(FakeSdk)
    assert ei.value.reason == "azure_auth"
    assert ei.value.retryable is False


async def test_stt_transient_failure_retried_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", fake_sleep)
    _make_cancellation("ConnectionFailure", "net down")
    _RESULT_QUEUE.append(_fake_result(reason=2))  # attempt 1: canceled, retryable
    _RESULT_QUEUE.append(_fake_result(text="recovered"))  # attempt 2: success
    result = await _run_stt(FakeSdk)
    assert result["text"] == "recovered"
    assert sleeps == [CLOUD_RETRY_BACKOFFS[0]]  # exactly one re-attempt


async def test_stt_empty_stream_short_circuits():
    engine = AzureSTTEngine(Cfg(), sdk=FakeSdk)
    await engine.start_stream()
    result = await engine.finish()
    assert result["text"] == ""
    assert result["duration"] == 0.0


async def test_stt_sdk_missing_is_structured_error(monkeypatch):
    def _raise(name):
        raise ImportError("no azure package")

    monkeypatch.setattr("fusion_speech.cloud_azure.importlib.import_module", _raise)
    engine = AzureSTTEngine(Cfg(), sdk=None)
    await engine.start_stream()
    await engine.feed(_tone_pcm(0.1))
    with pytest.raises(CloudEngineError) as ei:
        await engine.finish()
    assert ei.value.reason == "azure_sdk_missing"
    assert ei.value.retryable is False


# --- SECTION: azure tts (fake httpx) ---


def _wav_bytes(seconds: float = 0.1) -> bytes:
    return pcm_to_wav(_tone_pcm(seconds), 16000)


class FakeStreamCM:
    def __init__(self, response):
        self.response = response
        self.exited = False

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc):
        self.exited = True


class FakeResponse:
    def __init__(self, status_code=200, chunks=()):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.request = SimpleNamespace(headers={})

    async def aread(self):
        return b"err-body"

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []
        self.closed = False

    def stream(self, method, url, headers=None, content=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "content": content})
        return FakeStreamCM(self.response)

    async def aclose(self):
        self.closed = True


def _tts_engine(response, **kw):
    client = FakeHttpClient(response)
    engine = AzureTTSEngine(Cfg(), client_factory=lambda: client, **kw)
    return engine, client


async def test_tts_ssml_headers_and_stream():
    wav = _wav_bytes()
    engine, client = _tts_engine(FakeResponse(200, [wav[:64], wav[64:]]))
    out = b""
    async for chunk in engine.synthesize("hello <world>", voice="en-US-GuyNeural"):
        out += chunk
    assert out == wav
    call = client.calls[0]
    assert call["method"] == "POST"
    assert "eastus.tts.speech.microsoft.com" in call["url"]
    h = call["headers"]
    assert h["Ocp-Apim-Subscription-Key"] == "test-key"
    assert h["X-Microsoft-OutputFormat"] == "riff-16khz-16bit-mono-pcm"
    assert "application/ssml+xml" in h["Content-Type"]
    ssml = call["content"].decode()
    assert "xml:lang='en-US'" in ssml
    assert "<voice name='en-US-GuyNeural'>" in ssml
    assert "hello &lt;world&gt;" in ssml  # text is XML-escaped
    assert client.closed is True
    assert engine.active is False


async def test_tts_auth_failure_non_retryable(monkeypatch):
    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", _no_sleep)
    engine, client = _tts_engine(FakeResponse(401))
    with pytest.raises(CloudEngineError) as ei:
        async for _ in engine.synthesize("hi"):
            pass
    assert ei.value.reason == "azure_auth"
    assert ei.value.status == 401
    assert len(client.calls) == 1  # 4xx never retried


async def test_tts_rate_limited_then_success(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", fake_sleep)
    client = FakeHttpClient(None)
    client.responses = [FakeResponse(429), FakeResponse(200, [_wav_bytes()])]

    def _stream(method, url, headers=None, content=None):
        resp = client.responses.pop(0)
        client.calls.append({"url": url, "headers": headers})
        return FakeStreamCM(resp)

    client.stream = _stream
    engine = AzureTTSEngine(Cfg(), client_factory=lambda: client)
    out = b""
    async for chunk in engine.synthesize("hi"):
        out += chunk
    assert len(out) > 0
    assert len(client.calls) == 2
    assert sleeps == [CLOUD_RETRY_BACKOFFS[0]]


async def test_tts_rejects_non_riff_body(monkeypatch):
    monkeypatch.setattr("fusion_speech.cloud.asyncio.sleep", _no_sleep)
    engine, _ = _tts_engine(FakeResponse(200, [b"garbage-not-wav" * 8]))
    with pytest.raises(CloudEngineError) as ei:
        async for _ in engine.synthesize("hi"):
            pass
    assert ei.value.reason == "azure_invalid_response"


# --- SECTION: registry + rpc routing ---


def test_register_azure_engines():
    register_azure_engines()
    asr = make_asr_engine("azure", Cfg())
    tts = make_tts_engine("azure", Cfg())
    assert isinstance(asr, AzureSTTEngine)
    assert isinstance(tts, AzureTTSEngine)


def test_azure_factory_reads_lid_candidates():
    register_azure_engines()
    cfg = Cfg()
    cfg.azure_stt_languages = "ar-EG,ar-SA"
    asr = make_asr_engine("azure", cfg)
    assert asr._languages == ["ar-EG", "ar-SA"]


async def test_rpc_transcribe_routes_engine():
    from fusion_speech.ipc.methods import handle_transcribe

    class FakeAsr:
        name = "fakecloud"

        async def transcribe_wav(self, wav_bytes, *, language=""):
            return {"text": "cloud text", "engine": "fakecloud", "language": language}

    from fusion_speech import engines

    engines.register_asr_engine("fakecloud", lambda config, **kw: FakeAsr())
    daemon = SimpleNamespace(config=Cfg(), audio_lane={})
    wav_b64 = base64.b64encode(_wav_bytes()).decode()
    result = await handle_transcribe({"audio": wav_b64, "engine": "fakecloud", "language": "ar-EG"}, daemon)
    assert result["text"] == "cloud text"
    assert result["engine"] == "fakecloud"


async def test_rpc_transcribe_unknown_engine_raises():
    from fusion_speech.ipc.methods import handle_transcribe

    daemon = SimpleNamespace(config=Cfg(), audio_lane={})
    wav_b64 = base64.b64encode(_wav_bytes()).decode()
    with pytest.raises(ValueError) as ei:
        await handle_transcribe({"audio": wav_b64, "engine": "nope"}, daemon)
    assert "unknown asr engine" in str(ei.value)


async def test_rpc_engines_lists_registry():
    from fusion_speech.ipc.methods import handle_engines

    register_azure_engines()
    result = await handle_engines({}, SimpleNamespace(config=Cfg()))
    assert "local" in result["asr"]
    assert "azure" in result["asr"]
    assert "local" in result["tts"]
    assert "azure" in result["tts"]


async def test_asr_engine_transcribe_adapter():
    class AdapterAsr:
        name = "fakeadapter"

        async def start_stream(self, *, language=""):
            AdapterAsr.lang = language

        async def feed(self, pcm):
            AdapterAsr.fed = getattr(AdapterAsr, "fed", []) + [pcm.size]

        async def finish(self):
            return {"text": "adapted", "language": AdapterAsr.lang, "engine": "fakeadapter"}

        async def cancel(self):
            pass

    from fusion_speech import engines

    engines.register_asr_engine("fakeadapter", lambda config, **kw: AdapterAsr())
    transcribe = asr_engine_transcribe_adapter("fakeadapter")
    result = await transcribe(Cfg(), pcm_to_wav(_tone_pcm(0.2), 16000), filename="x.wav", language="ar-EG")
    assert result["text"] == "adapted"
    assert result["language"] == "ar-EG"
    assert AdapterAsr.fed == [int(16000 * 0.2)]
