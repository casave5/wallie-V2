"""Web Live2D avatar backend for Wallie.

Drives a browser-based Live2D model (the ``avatar/web`` viewer served on the
local network) over its ``POST /api/params`` endpoint, instead of VTube Studio.

Same lipsync envelope as the VTS client (RMS -> mouth open with asymmetric
attack/release) plus mood-linked brows/smile, `look_at`, and a status report.
Idle sway, eye darts and blinking are left to the web viewer's built-in Idle
motion + auto-blink, so this class stays deliberately slim.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from config import AvatarConfig

_RECONNECT_DELAY = 5.0
_POST_TIMEOUT = 2.0
_PING_INTERVAL = 2.0

# Semantic key -> Live2D parameter id (koharu/haruto model set).
_ID = {
    "mouth_open": "PARAM_MOUTH_OPEN_Y",
    "mouth_smile": "PARAM_MOUTH_FORM",
    "mouth_form": "PARAM_MOUTH_FORM_02",
    "face_x": "PARAM_ANGLE_X",
    "face_y": "PARAM_ANGLE_Y",
    "face_z": "PARAM_ANGLE_Z",
    "eye_x": "PARAM_EYE_BALL_X",
    "eye_y": "PARAM_EYE_BALL_Y",
    "eye_l": "PARAM_EYE_L_OPEN",
    "eye_r": "PARAM_EYE_R_OPEN",
    "brow_l": "PARAM_BROW_L_Y",
    "brow_r": "PARAM_BROW_R_Y",
    "body_x": "PARAM_BODY_ANGLE_X",
    "body_y": "PARAM_BODY_ANGLE_Y",
    "body_z": "PARAM_BODY_ANGLE_Z",
}


class WebAvatar:
    """Async HTTP client for the local Live2D viewer. Never blocks audio."""

    def __init__(self, cfg: "AvatarConfig") -> None:
        self._cfg = cfg
        url = str(getattr(cfg, "web_url", "http://127.0.0.1:8100"))
        self._base_url = url.rstrip("/")
        self._endpoint = f"{self._base_url}/api/params"
        self._status_url = f"{self._base_url}/api/estado"
        self._gesto_url = f"{self._base_url}/api/gesto"

        self._session: Any = None
        self._ready = False
        self._running = True
        self._speaking = False

        # Lipsync envelope state (same as VTS client).
        self._mouth_current: float = 0.0
        self._smile_current: float = 0.0
        self._form_current: float = 0.5
        self._brow_current: float = 0.0
        self._resting_smile: float = 0.0

        self._mood_arousal: float = 0.55
        self._mood_valence: float = 0.15
        self._mood_focus: float = 0.75

        self._connect_started_at: float = 0.0
        self._last_post: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Keep-alive loop: wait for the viewer to be reachable, reconnect later."""
        while self._running:
            try:
                await self._wait_ask_ready()
            except Exception as exc:
                logger.debug(f"avatar(web): viewer unreachable: {exc!r}")
            await asyncio.sleep(_RECONNECT_DELAY)

    async def close(self) -> None:
        self._running = False
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._ready

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._cfg.enabled,
            "connected": self._ready,
            "host": self._endpoint,
            "speaking": self._speaking,
            "model": "browser-live2d",
            "hotkey_count": 0,
            "uptime_sec": round(time.time() - self._connect_started_at, 1) if self._ready else 0.0,
            "mood_arousal": round(self._mood_arousal, 2),
            "mood_valence": round(self._mood_valence, 2),
            "mood_focus": round(self._mood_focus, 2),
        }

    # ------------------------------------------------------------------
    # Control surface (same signature as VTubeStudioAvatar)
    # ------------------------------------------------------------------

    async def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        if not speaking:
            self._mouth_current = 0.0
            self._form_current = 0.5
            params = {
                "mouth_open": 0.0,
                "mouth_smile": max(0.0, self._smile_current),
            }
            if self._cfg.enable_viseme_lipsync:
                params["mouth_form"] = 0.5
            await self._inject(params)

    async def set_volume(self, rms: float) -> None:
        if rms < self._cfg.lipsync_floor:
            target = 0.0
        else:
            target = min(1.0, rms * self._cfg.lipsync_gain) * self._cfg.lipsync_ceiling

        if target > self._mouth_current:
            self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_attack
        else:
            self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_release

        smile_target = self._cfg.speaking_smile if self._speaking else self._resting_smile
        self._smile_current += (smile_target - self._smile_current) * 0.15

        await self._inject(
            {"mouth_open": round(self._mouth_current, 3), "mouth_smile": round(self._smile_current, 3)}
        )

    async def feed_audio(self, pcm: bytes, sample_rate: int = 24000) -> None:
        if not pcm or len(pcm) < 8:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size < 8:
            return

        # Process the whole block in short windows so the mouth moves with each
        # syllable instead of opening once per Piper sentence block.
        window = max(64, int(sample_rate * 0.01))  # ~10 ms
        for start in range(0, len(samples), window):
            piece = samples[start:start + window]
            rms = float(np.sqrt(np.mean(piece ** 2))) / 32768.0
            rms = min(1.0, rms)

            if rms < self._cfg.lipsync_floor:
                target = 0.0
            else:
                target = min(1.0, rms * self._cfg.lipsync_gain) * self._cfg.lipsync_ceiling

            if target > self._mouth_current:
                self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_attack
            else:
                self._mouth_current += (target - self._mouth_current) * self._cfg.lipsync_release

            smile_target = self._cfg.speaking_smile if self._speaking else self._resting_smile
            self._smile_current += (smile_target - self._smile_current) * 0.15

            params: dict[str, float] = {
                "mouth_open": round(self._mouth_current, 3),
                "mouth_smile": round(self._smile_current, 3),
            }
            if self._cfg.enable_viseme_lipsync:
                if rms > self._cfg.lipsync_floor:
                    form = self._estimate_mouth_form(piece / 32768.0, sample_rate)
                    self._form_current += (form - self._form_current) * self._cfg.viseme_smoothing
                else:
                    self._form_current += (0.5 - self._form_current) * 0.15
                params["mouth_form"] = round(self._form_current, 3)

            await self._inject(params)
            # Real-time pacing: run slightly below realtime so audio and mouth stay
            # aligned even though Piper delivered everything up front.
            await asyncio.sleep((len(piece) / sample_rate) * 0.9)

    def _estimate_mouth_form(self, samples: np.ndarray, sr: int) -> float:
        n = len(samples)
        if n < 256:
            return self._form_current
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(samples * window))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        low_mask = (freqs >= 300) & (freqs <= 1200)
        high_mask = (freqs >= 1500) & (freqs <= 3500)
        low_energy = float(spectrum[low_mask].sum()) if low_mask.any() else 0.0
        high_energy = float(spectrum[high_mask].sum()) if high_mask.any() else 0.0
        total = low_energy + high_energy
        if total < 1e-6:
            return self._form_current
        return high_energy / total

    async def set_smile(self, amount: float) -> None:
        amount = max(0.0, min(1.0, amount))
        self._smile_current = amount
        await self._inject({"mouth_smile": round(amount, 3)})

    async def look_at(self, x: float, y: float, *, hold_sec: float = 0.6) -> None:
        x = max(-30.0, min(30.0, x))
        y = max(-30.0, min(30.0, y))
        await self._inject({"face_x": x, "face_y": y})
        if hold_sec > 0:
            await asyncio.sleep(hold_sec)

    async def trigger_expression(self, name: str) -> None:
        nombre = (name or "").strip().lower()
        if not nombre:
            return
        gesto = None
        if nombre in {"saludo", "hola", "ola", "wave", "wave_hello", "hello"}:
            gesto = "saludo"
        if gesto is None:
            logger.debug(f"avatar(web): expresión '{name}' no mapeada, ignorada")
            return
        async with self.GetSession() as s:
            try:
                async with s.post(
                    self._gesto_url, json={"gesto": gesto}, timeout=_POST_TIMEOUT
                ) as resp:
                    if resp.status != 200:
                        logger.debug(f"avatar(web): gesto '{gesto}' falló ({resp.status})")
            except Exception as exc:
                logger.debug(f"avatar(web): gesto '{gesto}' falló ({exc!r})")

    async def trigger_emotion(self, slot: str) -> None:
        logger.debug(f"avatar(web): emotion '{slot}' ignored (no expressions)")

    async def update_mood(self, arousal: float, valence: float, focus: float) -> None:
        self._mood_arousal = arousal
        self._mood_valence = valence
        self._mood_focus = focus
        if not self._cfg.enable_mood_link:
            return
        t = (valence + 1.0) / 2.0
        brow_target = self._cfg.mood_brow_min + (self._cfg.mood_brow_max - self._cfg.mood_brow_min) * t
        self._brow_current += (brow_target - self._brow_current) * 0.12
        smile_target = max(0.0, valence) * self._cfg.mood_smile_max
        self._resting_smile += (smile_target - self._resting_smile) * 0.12

        params: dict[str, float] = {
            "brow_l": round(self._brow_current, 3),
            "brow_r": round(self._brow_current, 3),
        }
        if not self._speaking:
            params["mouth_smile"] = round(max(self._resting_smile, self._smile_current), 3)
        await self._inject(params)

    # ------------------------------------------------------------------
    # Discovery stubs (dashboard compatibility)
    # ------------------------------------------------------------------

    async def query_hotkeys(self) -> list[dict[str, Any]]:
        return []

    async def query_model_info(self) -> dict[str, Any]:
        return {"modelName": "browser-live2d"}

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _wait_ask_ready(self) -> None:
        async with self.GetSession() as s:
            for _ in range(3):
                if not self._running:
                    return
                try:
                    async with s.get(self._status_url, timeout=_POST_TIMEOUT) as resp:
                        if resp.status == 200:
                            break
                except Exception:
                    pass
                await asyncio.sleep(1.0)
            else:
                return

            ok = self._ready
            self._ready = True
            if not ok:
                self._connect_started_at = time.time()
                logger.info(f"avatar(web): viewer en línea en {self._endpoint}")
            while self._running:
                await asyncio.sleep(_PING_INTERVAL)
                try:
                    async with s.get(self._status_url, timeout=_POST_TIMEOUT) as resp:
                        if resp.status == 200:
                            continue
                except Exception:
                    pass
                self._ready = False
                return

    def GetSession(self) -> Any:
        import contextlib

        @contextlib.asynccontextmanager
        async def _mgr():
            import aiohttp

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            try:
                yield self._session
            except Exception:
                await self._session.close()
                self._session = None
                raise

        return _mgr()

    async def _inject(self, params: dict[str, float]) -> None:
        if not params or not self._ready:
            return
        live: dict[str, float] = {}
        for key, val in params.items():
            pid = _ID.get(key)
            if pid and math.isfinite(val):
                live[pid] = round(float(val), 3)
        if not live:
            return

        # Throttle to ~25 Hz; the viewer smooths anyway.
        now = time.time()
        if now - self._last_post < 0.04:
            return
        self._last_post = now

        async with self.GetSession() as s:
            try:
                async with s.post(
                    self._endpoint, json={"params": live}, timeout=_POST_TIMEOUT
                ) as resp:
                    if resp.status != 200:
                        self._ready = False
            except Exception as exc:
                logger.debug(f"avatar(web): inject failed ({exc!r})")
                self._ready = False