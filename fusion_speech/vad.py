from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fusion_speech.config import DEFAULT_SAMPLE_RATE, VAD_SPEECH_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_SILERO_PATH = os.path.expanduser("~/.fusion-speech/models/silero_vad.onnx")
_WINDOW_16K = 512
_DEFAULT_NEG_OFFSET = 0.15
_DEFAULT_MIN_SPEECH_MS = 250
_DEFAULT_MIN_SILENCE_MS = 500
_DEFAULT_SPEECH_PAD_MS = 30
_DEFAULT_MAX_SPEECH_S = 15.0
_ENERGY_RMS_GATE = 0.01


@dataclass
class VadSegment:
    start_sample: int
    end_sample: int

    @property
    def samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass
class _VadConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    threshold: float = VAD_SPEECH_THRESHOLD
    min_speech_ms: int = _DEFAULT_MIN_SPEECH_MS
    min_silence_ms: int = _DEFAULT_MIN_SILENCE_MS
    speech_pad_ms: int = _DEFAULT_SPEECH_PAD_MS
    max_speech_s: float = _DEFAULT_MAX_SPEECH_S


class SileroVadModel:
    # Thin onnxruntime wrapper for silero_vad_16k_op15.onnx. Holds the recurrent
    # state across chunks so it can be called one 512-sample window at a time.
    # No torch dependency: state is a plain numpy (2,1,128) float32 array, reset
    # to zeros at segment boundaries. Fail-soft: if onnxruntime or the model
    # file is missing, SileroVadEngine falls back to energy-only detection.
    def __init__(self, model_path: str = DEFAULT_SILERO_PATH):
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"silero vad model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.sample_rate = 16000
        self._context_size = 64
        self.reset_state()

    def reset_state(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        # Silero expects a rolling 64-sample context prepended to each 512-sample
        # window (input length 576). Without it the recurrent state gets no
        # signal and probabilities collapse to ~0. We mirror the official
        # utils_vad.OnnxModel.__call__ but keep numpy (no torch dependency).
        if window.size < _WINDOW_16K:
            window = np.pad(window, (0, _WINDOW_16K - window.size))
        elif window.size > _WINDOW_16K:
            window = window[:_WINDOW_16K]
        x = window.astype(np.float32, copy=False).reshape(1, -1)
        x = np.concatenate([self._context, x], axis=1)
        sr = np.array(self.sample_rate, dtype=np.int64)
        out, state = self.session.run(None, {"input": x, "state": self._state, "sr": sr})
        self._state = state
        self._context = x[:, -self._context_size :]
        return float(out[0, 0])


class SileroVadEngine:
    # Streaming two-tier VAD. Tier 1 = energy RMS gate (numpy, ~free): if a
    # window's RMS is below _ENERGY_RMS_GATE it is instantly classified silence,
    # skipping the ONNX run. Tier 2 = Silero ONNX probability vs threshold /
    # neg_threshold. Emits segment-open (speech onset) and segment-close (silence
    # after speech >= min_silence) events via callbacks; the closed segment
    # (with speech padding) is passed to on_segment_close. fail-soft: if the
    # ONNX model cannot load, degrades to energy-only with threshold on RMS.
    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        threshold: float = VAD_SPEECH_THRESHOLD,
        min_speech_ms: int = _DEFAULT_MIN_SPEECH_MS,
        min_silence_ms: int = _DEFAULT_MIN_SILENCE_MS,
        speech_pad_ms: int = _DEFAULT_SPEECH_PAD_MS,
        max_speech_s: float = _DEFAULT_MAX_SPEECH_S,
        model_path: str = DEFAULT_SILERO_PATH,
        on_open: Callable[[int], None] | None = None,
        on_close: Callable[[VadSegment], None] | None = None,
        model: SileroVadModel | None = None,
    ):
        self.cfg = _VadConfig(
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
        )
        sr = self.cfg.sample_rate
        self.window = _WINDOW_16K if sr == 16000 else 256
        self.min_speech_samples = int(sr * min_speech_ms / 1000)
        self.min_silence_samples = int(sr * min_silence_ms / 1000)
        self.speech_pad_samples = int(sr * speech_pad_ms / 1000)
        self.max_speech_samples = int(sr * max_speech_s)
        self.neg_threshold = max(threshold - _DEFAULT_NEG_OFFSET, 0.01)
        self._on_open = on_open
        self._on_close = on_close
        self._model = model
        self._energy_only = False
        if model is None:
            try:
                self._model = SileroVadModel(model_path)
                logger.info("silero VAD loaded: %s", model_path)
            except Exception as e:
                logger.warning("silero VAD unavailable, energy-only fallback: %s", e)
                self._model = None
                self._energy_only = True
        self.reset()

    @property
    def energy_only(self) -> bool:
        return self._energy_only

    def reset(self) -> None:
        # Pre-allocated scratch buffer + write cursor (_len). feed() copies new
        # samples in at _len (O(chunk), not O(buffer) like np.concatenate), and
        # consumes windows by shifting the < window tail to the front (O(window),
        # bounded) — avoids the per-feed quadratic realloc of the old concat
        # (audit P2). Capacity grows lazily; never shrinks.
        self._buf = np.empty(self.window * 4, dtype=np.float32)
        self._len = 0
        self._in_speech = False
        self._speech_start = 0
        self._temp_end = 0
        self._total_samples = 0
        if self._model is not None:
            self._model.reset_state()

    def feed(self, samples: np.ndarray) -> None:
        self._feed_sync(samples)

    async def feed_async(self, samples: np.ndarray) -> None:
        # F-VAD-LOOP: the ONNX predict (_speech_prob -> self._model(window)) is
        # a ~5-15ms synchronous call. When feed() runs on the drain task (the
        # asyncio loop), that blocks ASR-final push + WS writes during active
        # speech. Push the whole window-processing batch to the executor as one
        # sync unit so recurrent state (_state/_context) stays coherent (single
        # drain task awaits sequentially -> no two _feed_sync run concurrently)
        # and the loop stays free. on_open/on_close callbacks use
        # run_coroutine_threadsafe / ensure_future(loop=) which are thread-safe.
        if samples.size == 0:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._feed_sync, samples)

    def _feed_sync(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        n = samples.size
        need = self._len + n
        if self._buf.size < need:
            newcap = max(need, self._buf.size * 2, self.window * 4)
            grown = np.empty(newcap, dtype=np.float32)
            grown[: self._len] = self._buf[: self._len]
            self._buf = grown
        self._buf[self._len : need] = np.asarray(samples, dtype=np.float32)
        self._len = need
        while self._len >= self.window:
            window = self._buf[: self.window].copy()
            tail = self._len - self.window
            if tail > 0:
                self._buf[:tail] = self._buf[self.window : self._len]
            self._len = tail
            self._process_window(window)

    def _process_window(self, window: np.ndarray) -> None:
        self._total_samples += self.window
        prob = self._speech_prob(window)
        cur = self._total_samples - self.window

        if not self._in_speech:
            if prob >= self.cfg.threshold:
                self._in_speech = True
                self._speech_start = cur
                self._temp_end = 0
                logger.debug("vad speech onset at sample %d (prob=%.3f)", cur, prob)
                if self._on_open is not None:
                    try:
                        self._on_open(cur)
                    except Exception:
                        logger.exception("vad on_open callback failed")
        else:
            if cur - self._speech_start >= self.max_speech_samples:
                self._close_segment(cur, forced=True)
                return
            if prob < self.neg_threshold:
                if self._temp_end == 0:
                    self._temp_end = cur
                if cur - self._temp_end >= self.min_silence_samples:
                    self._close_segment(cur)
            else:
                self._temp_end = 0

    def _speech_prob(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(window * window)) + 1e-12)
        if rms < _ENERGY_RMS_GATE:
            return 0.0
        if self._energy_only or self._model is None:
            # map RMS to a 0..1-ish score so the threshold/neg_threshold logic
            # still drives onset/close in energy-only mode.
            return min(rms / max(self.cfg.threshold, 0.02), 1.0)
        return self._model(window)

    def _close_segment(self, cur_sample: int, *, forced: bool = False) -> None:
        end = self._temp_end if (self._temp_end and not forced) else cur_sample
        seg = VadSegment(start_sample=self._speech_start, end_sample=end)
        if seg.samples >= self.min_speech_samples or forced:
            padded = VadSegment(
                start_sample=max(0, seg.start_sample - self.speech_pad_samples),
                end_sample=seg.end_sample + self.speech_pad_samples,
            )
            logger.info(
                "vad segment close: %d-%d (%d samples)%s",
                padded.start_sample,
                padded.end_sample,
                padded.samples,
                " [forced max]" if forced else "",
            )
            if self._on_close is not None:
                try:
                    self._on_close(padded)
                except Exception:
                    logger.exception("vad on_close callback failed")
        self._in_speech = False
        self._speech_start = 0
        self._temp_end = 0
        if self._model is not None:
            self._model.reset_state()

    def flush(self) -> None:
        # Called on stop: close any open segment at the buffer tail.
        if self._in_speech:
            self._close_segment(self._total_samples, forced=True)

    def stats(self) -> dict:
        return {
            "energy_only": self._energy_only,
            "in_speech": self._in_speech,
            "buffered": int(self._len),
            "total_samples": self._total_samples,
            "window": self.window,
            "threshold": self.cfg.threshold,
        }
