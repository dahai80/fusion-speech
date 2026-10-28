from __future__ import annotations

import pytest

# Issue #6: API-key/port/data-dir resolution lives in fusion_core.config only
# (config-conventions §2, fusion-core#31). No local resolvers, one-line call
# sites, dev-missing-key = "" + warning, FUSION_PROFILE=prod = RuntimeError.

_KEY_ENVS = ("FUSION_CLIENT_KEY", "FUSION_MLX_API_KEY", "MLX_API_KEY")


def _clear_key_env(monkeypatch):
    for var in _KEY_ENVS:
        monkeypatch.delenv(var, raising=False)


def test_client_key_env_wins_for_mlx_api_key(monkeypatch):
    from fusion_speech.config import SpeechConfig

    _clear_key_env(monkeypatch)
    monkeypatch.setenv("FUSION_CLIENT_KEY", "client-key-value")
    monkeypatch.setenv("FUSION_MLX_API_KEY", "legacy-value")
    assert SpeechConfig().mlx_api_key == "client-key-value"


def test_legacy_mlx_key_alias_still_resolves(monkeypatch):
    from fusion_speech.config import SpeechConfig

    _clear_key_env(monkeypatch)
    monkeypatch.setenv("FUSION_MLX_API_KEY", "legacy-value")
    assert SpeechConfig().mlx_api_key == "legacy-value"


def test_dev_missing_key_is_empty(monkeypatch):
    from fusion_speech.config import SpeechConfig

    _clear_key_env(monkeypatch)
    monkeypatch.delenv("FUSION_PROFILE", raising=False)
    assert SpeechConfig().mlx_api_key == ""


def test_prod_missing_key_raises(monkeypatch):
    from fusion_speech.config import SpeechConfig

    _clear_key_env(monkeypatch)
    monkeypatch.setenv("FUSION_PROFILE", "prod")
    with pytest.raises(RuntimeError, match="FUSION_CLIENT_KEY not set"):
        SpeechConfig()


def test_port_via_service_port(monkeypatch):
    from fusion_speech.config import SpeechConfig

    monkeypatch.setenv("FUSION_SPEECH_PORT", "12345")
    assert SpeechConfig().port == 12345


def test_port_non_integer_fails_loud(monkeypatch):
    from fusion_speech.config import SpeechConfig

    monkeypatch.setenv("FUSION_SPEECH_PORT", "not-a-port")
    with pytest.raises(ValueError, match="FUSION_SPEECH_PORT"):
        SpeechConfig()


def test_data_dir_env_override(monkeypatch, tmp_path):
    from fusion_speech.config import _data_dir

    monkeypatch.setenv("FUSION_SPEECH_DATA_DIR", str(tmp_path))
    assert _data_dir() == tmp_path
