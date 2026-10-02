"""Low-latency PCM audio player with alignment-safe writes."""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger


def resolve_output_device(spec: Optional[int | str]) -> Optional[int | str]:
    """Turn a device spec into a concrete output-device index.

    A bare name like 'CABLE Input' matches the same endpoint across multiple host
    APIs (MME/DirectSound/WASAPI), which makes sounddevice raise on ambiguity. We
    resolve by name ourselves and prefer WASAPI (lowest latency, matches the rest
    of the pipeline). Indices/None/empty pass straight through.
    """
    if spec is None or spec == "":
        return None
    try:
        return int(spec)  # already an index (or numeric string)
    except (ValueError, TypeError):
        pass
    name = str(spec).lower()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:  # noqa: BLE001 — let sounddevice handle it downstream
        return spec
    wasapi = next((i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None)
    matches = [i for i, d in enumerate(devices)
               if name in d["name"].lower() and d["max_output_channels"] > 0]
    if not matches:
        return spec
    for i in matches:  # prefer the WASAPI instance when present
        if wasapi is not None and devices[i]["hostapi"] == wasapi:
            return i
    return matches[0]


class AudioPlayer:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        blocksize: int = 960,  # ~20ms at 48kHz, ~40ms at 24kHz
        device: Optional[int | str] = None,
    ) -> None:
        self._sr = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = resolve_output_device(device)

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._finished_event = asyncio.Event()
        self._finished_event.set()
        self._pending_odd: Optional[int] = None

        self._stream: Optional[sd.RawOutputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_write_ts: float = 0.0  # when audio was last queued (for hearing self-mute)
        self._underruns: int = 0
        self._underflow_rt: int = 0

    # ----- lifecycle -----
    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        # WASAPI shared mode demands the stream rate match the device's mix format
        # (e.g. VB-CABLE is fixed at 48 kHz) and rejects our 24 kHz TTS otherwise.
        # Enable auto-convert on WASAPI devices so PortAudio resamples for us.
        extra = None
        try:
            if isinstance(self._device, int):
                ha = sd.query_hostapis(sd.query_devices(self._device)["hostapi"])["name"]
                if "wasapi" in ha.lower():
                    extra = sd.WasapiSettings(auto_convert=True)
        except Exception:  # noqa: BLE001 — fall back to no extra settings
            extra = None
        self._stream = sd.RawOutputStream(
            samplerate=self._sr,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
            extra_settings=extra,
        )
        self._stream.start()
        logger.info(f"audio: playing at {self._sr} Hz, {self._channels}ch, block={self._blocksize}"
                    f"{' (WASAPI auto-convert)' if extra else ''}")
        self._prioridad_tiempo_real()

    # ----- prioridad de tiempo real -----
    def _prioridad_tiempo_real(self, prioridad: int = 20) -> None:
        """Sube a SCHED_FIFO los hilos del stream de audio.

        El callback de PortAudio corre en un hilo propio (no es un hilo de
        Python) que, con prioridad normal, se retrasa cuando la CPU se llena y
        el audio sale entrecortado. Con tiempo real el hilo se ejecuta a tiempo.
        """
        objetivo = prioridad
        subidos = 0
        for tid, comm in self._hilos_de_audio():
            try:
                if (os.sched_getscheduler(tid) == os.SCHED_FIFO
                        and os.sched_getparam(tid).sched_priority >= objetivo):
                    continue
                os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(objetivo))
                subidos += 1
                logger.info(f"audio: hilo {tid} ({comm}) a SCHED_FIFO prio {objetivo}")
            except Exception:  # noqa: BLE001 — sin permisos o hilo ya muerto
                continue
        if not subidos:
            logger.info("audio: no se pudo aplicar prioridad RT a los hilos de audio")

    @staticmethod
    def _hilos_de_audio() -> list[tuple[int, str]]:
        """Localiza los hilos nativos del stream por nombre en /proc/self/task."""
        encontrado: list[tuple[int, str]] = []
        try:
            base = Path("/proc/self/task")
            for tarea in base.iterdir():
                if not tarea.name.isdigit():
                    continue
                try:
                    comm = (tarea / "comm").read_text().strip()
                except OSError:
                    continue
                if "PortAudio" in comm or "portaudio" in comm.lower():
                    encontrado.append((int(tarea.name), comm))
        except OSError:
            return []
        return encontrado

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"audio stream close: {e}")
            self._stream = None

    # ----- writing -----
    async def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._pending_odd is not None:
            pcm = bytes((self._pending_odd,)) + pcm
            self._pending_odd = None
        if len(pcm) & 1:
            self._pending_odd = pcm[-1]
            pcm = pcm[:-1]
        if not pcm:
            return
        with self._lock:
            self._buf.extend(pcm)
        self._last_write_ts = time.time()
        self._finished_event.clear()
        await asyncio.sleep(0)

    def speaking_recently(self, window: float) -> bool:
        """True if Wallie is currently outputting audio or did within `window` seconds.
        Used by hearing to skip windows that contain Wallie's own voice (no self-echo)."""
        return self.seconds_queued() > 0.02 or (time.time() - self._last_write_ts) < window

    def boundary(self) -> None:
        if self._pending_odd is not None:
            logger.debug(f"audio: dropping leftover odd byte at sentence boundary")
            self._pending_odd = None

    def reset(self) -> None:
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._finished_event.set)
            except RuntimeError:
                pass
        logger.info(f"audio: hard reset, dropped {dropped} bytes")

    def interrupt(self) -> None:
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._finished_event.set)
        if dropped:
            logger.info(f"audio: interrupt, dropped {dropped} bytes")

    async def wait_drained(self) -> None:
        await self._finished_event.wait()

    def seconds_queued(self) -> float:
        with self._lock:
            return len(self._buf) / (self._sr * self._channels * 2)

    # ----- audio callback -----
    def _callback(self, outdata, frames: int, time_info, status) -> None:  # noqa: ARG002
        if status:
            self._underflow_rt += 1
            logger.warning(f"audio status (UNDERFLOW real #{self._underflow_rt}): {status}")
        needed = frames * self._channels * 2
        with self._lock:
            available = len(self._buf)
            take = min(needed, available)
            if take:
                chunk = bytes(self._buf[:take])
                del self._buf[:take]
            else:
                chunk = b""
            empty_after = len(self._buf) == 0

        if take < needed and (time.time() - self._last_write_ts) < 0.5:
            # Búfer vacío justo tras escribir audio = ayuno real (se oye un corte).
            # Durante silencio normal el búfer también está vacío, pero no es un fallo.
            self._underruns += 1
            logger.warning(
                f"audio: underrun #{self._underruns} (faltaron {needed - take} bytes, "
                f"quedaban {available})"
            )
        if take < needed:
            chunk = chunk + b"\x00" * (needed - take)
        outdata[: len(chunk)] = chunk

        if empty_after and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._finished_event.set)
            except RuntimeError:
                pass


def list_output_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": d.get("name", ""),
                    "hostapi": d.get("hostapi", 0),
                    "default_samplerate": d.get("default_samplerate", 0),
                }
            )
    return out
