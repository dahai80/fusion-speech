from __future__ import annotations

import sys

# Issue #5: the daemon must be identifiable in ps/Activity Monitor as the
# product, not bare homebrew "Python". setproctitle is the only reliable way
# on macOS; a missing package must degrade to the default title, never crash.


def test_process_title_set_when_sdk_present():
    from fusion_speech.server import _set_process_title

    _set_process_title(11465)
    import setproctitle

    assert setproctitle.getproctitle() == "fusion-speech server :11465"


def test_process_title_fallback_when_sdk_missing(monkeypatch):
    # None in sys.modules makes `import setproctitle` raise ImportError —
    # the fallback path must swallow it and leave the title untouched.
    from fusion_speech.server import _set_process_title

    monkeypatch.setitem(sys.modules, "setproctitle", None)
    _set_process_title(11465)  # must not raise
