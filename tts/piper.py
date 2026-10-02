"""Piper TTS — local CPU-based neural TTS adapter."""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from loguru import logger

from .base import TTSError, TTSProvider

# Casavita's signature "anime voice" chain. Raw Piper output (22.05 kHz) is
# pitched up and slowed slightly, then brightened with three peaking EQs,
# gently compressed and made louder. Equivalent to the 3-pass ffmpeg chain the
# Telegram bot uses, but in a single pass so latency stays acceptable for
# streaming. The trailing aresample sets the stream rate to 48000.
ANIME_VOICE_FILTER = (
    "asetrate=26200,"
    "aresample=48000,"
    "atempo=0.80,"
    "equalizer=f=1200:t=q:w=1:g=4,"
    "equalizer=f=3500:t=q:w=1:g=6,"
    "equalizer=f=7000:t=q:w=1:g=5,"
    "acompressor=threshold=0.08:ratio=2.5:attack=8:release=120,"
    "volume=1.35"
)
ANIME_VOICE_RATE = 48000

# Where the CUDA runtime libs live on this machine (Arch / CachyOS cuda package).
_CUDA_LIB_DIRS = ("/opt/cuda/lib64", "/usr/lib")


def _cuda_active(voice: Any) -> bool:
    """True when the loaded PiperVoice really placed its model on the GPU."""
    session = getattr(voice, "session", None) or getattr(voice, "voice", None)
    providers = getattr(session, "get_providers", None)
    if providers is None:
        return False
    try:
        return "CUDAExecutionProvider" in providers()
    except Exception:  # noqa: BLE001
        return False


class PiperTTS(TTSProvider):
    name = "piper"

    def __init__(
        self,
        *,
        model_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        postprocess: str = "",
        use_cuda: bool = True,
    ) -> None:
        if not model_path:
            raise TTSError("piper: model_path is required (download a .onnx voice file)")
        path = Path(model_path)
        if not path.is_file():
            raise TTSError(f"piper: model file not found: {path}")

        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise TTSError(
                "piper: 'piper-tts' is not installed. Install with: pip install piper-tts"
            ) from e

        self._voice = None
        if use_cuda:
            self._voice = self._load_cuda(PiperVoice, path)
        if self._voice is None:
            try:
                self._voice = PiperVoice.load(str(path))
                logger.info("piper: running on CPU (no CUDA execution provider)")
            except Exception as e:
                raise TTSError(f"piper: failed to load model {path.name}: {e}") from e
        self.device = "cuda" if use_cuda and _cuda_active(self._voice) else "cpu"
        logger.info(f"piper: model {path.name} loaded on {self.device.upper()}")

        # Sample rate from voice config, default if missing.
        sr = getattr(getattr(self._voice, "config", None), "sample_rate", None)
        self._model_rate = int(sr) if sr else 22050
        self.sample_rate = self._model_rate
        self.channels = 1
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w

        # Optional ffmpeg post-processing. Requires buffering the whole utterance,
        # so time-to-first-audio is slightly higher than raw Piper.
        self._postprocess = (postprocess or "").strip()
        if self._postprocess:
            self.sample_rate = ANIME_VOICE_RATE
            logger.info(f"piper: post-processing enabled -> {self._postprocess}")

    @staticmethod
    def _load_cuda(PiperVoice: Any, path: Path) -> Any:
        """Try to load the voice on the GPU. Returns None if CUDA is unavailable."""
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return None
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.info("piper: onnxruntime has no CUDA provider, using CPU")
            return None
        for cuda_dir in _CUDA_LIB_DIRS:
            if not Path(cuda_dir).is_dir():
                continue
            try:
                ort.preload_dlls(directory=cuda_dir)
            except Exception:  # noqa: BLE001 — best effort, providers may already be linked
                pass
        try:
            return PiperVoice.load(str(path), use_cuda=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"piper: CUDA load failed ({e}); falling back to CPU")
            return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._synthesize_blocking, text)
        if not self._postprocess:
            for c in chunks:
                if c:
                    yield c
            return

        raw = b"".join(chunks)
        if len(raw) & 1:
            raw = raw[:-1]
        if not raw:
            return
        processed = await loop.run_in_executor(None, self._apply_filter, raw)
        if processed:
            yield processed

    def _apply_filter(self, raw: bytes) -> bytes:
        """Run the configured ffmpeg -af chain over raw s16le mono PCM."""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.pcm"
            dst = Path(tmp) / "out.pcm"
            src.write_bytes(raw)
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "s16le", "-ar", str(self._model_rate), "-ac", "1",
                        "-i", str(src),
                        "-af", self._postprocess,
                        "-f", "s16le", "-ar", str(ANIME_VOICE_RATE), "-ac", "1",
                        str(dst),
                    ],
                    capture_output=True,
                    timeout=60,
                )
            except FileNotFoundError:
                logger.warning("piper: ffmpeg not found, skipping post-processing")
                return raw
            except subprocess.TimeoutExpired:
                logger.warning("piper: ffmpeg post-processing timed out, using raw audio")
                return raw
            if proc.returncode != 0 or not dst.is_file():
                err = proc.stderr.decode("utf-8", "replace").strip()[-300:]
                logger.warning(f"piper: ffmpeg post-processing failed ({err}); using raw audio")
                return raw
            return dst.read_bytes()

    def _synthesize_blocking(self, text: str) -> list[bytes]:
        try:
            iterator = self._iter_chunks(text)
        except Exception as e:
            raise TTSError(f"piper: synthesis failed: {e}") from e

        out: list[bytes] = []
        for chunk in iterator:
            buf: Any = getattr(chunk, "audio_int16_bytes", None)
            if buf is None:
                if isinstance(chunk, (bytes, bytearray)):
                    buf = bytes(chunk)
                else:
                    arr = getattr(chunk, "audio_float_array", None)
                    if arr is not None:
                        import numpy as np
                        buf = (arr * 32767).astype(np.int16).tobytes()
            if buf:
                out.append(bytes(buf))
        return out

    def _iter_chunks(self, text: str) -> Any:
        """Call whichever synthesize API this piper-tts build exposes.

        piper-tts >= 1.3 takes a SynthesisConfig; older builds took the
        length/noise kwargs directly, and 1.0-1.2 had synthesize_stream_raw.
        """
        try:
            from piper import SynthesisConfig  # type: ignore
        except ImportError:
            SynthesisConfig = None  # type: ignore[assignment]

        if SynthesisConfig is not None:
            cfg = SynthesisConfig(
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w,
            )
            return self._voice.synthesize(text, cfg)

        try:
            return self._voice.synthesize(
                text,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w=self._noise_w,
            )
        except TypeError:
            return self._voice.synthesize_stream_raw(
                text, length_scale=self._length_scale
            )

    async def aclose(self) -> None:
        self._voice = None  # type: ignore
