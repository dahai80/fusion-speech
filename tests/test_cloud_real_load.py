from __future__ import annotations

import os

import pytest

# Issue #4 real-load gate: exercises the Azure cloud engines against the REAL
# Azure Speech service. Opt-in only — requires FUSION_SPEECH_REAL_LOAD=1 plus
# live credentials. Azure keys are issued by fusion-identity (issue #4:
# business apps never carry their own); until that issuance path exists for
# this environment the gate skips, citing the blocker. Skipped (not xfail) so
# it flips to a real assertion once creds are provisioned.

pytestmark = pytest.mark.real_load

_AZURE_KEY = os.environ.get("FUSION_SPEECH_AZURE_KEY", "").strip()
_AZURE_REGION = os.environ.get("FUSION_SPEECH_AZURE_REGION", "").strip()

pytestskip_reason = (
    "Azure credentials not provisioned (FUSION_SPEECH_AZURE_KEY / FUSION_SPEECH_AZURE_REGION unset); "
    "blocked on fusion-identity Azure credential issuance"
)

if not (_AZURE_KEY and _AZURE_REGION):
    pytest.skip(pytestskip_reason, allow_module_level=True)

if os.environ.get("FUSION_SPEECH_REAL_LOAD") != "1":
    pytest.skip("FUSION_SPEECH_REAL_LOAD=1 not set; skipping cloud real-load integration", allow_module_level=True)


async def test_real_azure_stt_words_and_dialect():
    from fusion_speech.cloud_azure import AzureSTTEngine
    from fusion_speech.config import load_config
    from fusion_speech.stt_stream import pcm_to_wav

    config = load_config()
    config.azure_stt_languages = "ar-EG,ar-SA,en-US"
    engine = AzureSTTEngine(config, languages=["ar-EG", "ar-SA", "en-US"])
    wav = pcm_to_wav(_sine(16000), 16000)
    result = await engine.transcribe_wav(wav, language="en-US")
    assert result["engine"] == "azure"
    assert isinstance(result["text"], str)
    assert isinstance(result["words"], list)


async def test_real_azure_tts_wav_stream():
    from fusion_speech.cloud_azure import AzureTTSEngine
    from fusion_speech.config import load_config

    config = load_config()
    engine = AzureTTSEngine(config)
    chunks = []
    async for chunk in engine.synthesize("This is a fusion speech cloud engine test."):
        chunks.append(chunk)
    wav = b"".join(chunks)
    assert wav[:4] == b"RIFF"
    assert len(wav) > 1000


def _sine(n: int):
    import numpy as np

    t = np.arange(n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
