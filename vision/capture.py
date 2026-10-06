"""Screen capture. X11 via mss; Wayland (KDE) via spectacle.

mss talks straight to X11, so on a Wayland session it happily returns a perfectly
valid frame of pure black: the XWayland root window holds none of the compositor's
output, only the few X11 apps that happen to run through it. Casavita would then
"look" at a black rectangle and comment on nothing. Spectacle is the KDE native
screenshot tool and does see the real composited desktop, so on Wayland we shell
out to it and crop the monitor we care about.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import mss
from PIL import Image


@dataclass
class Frame:
    jpeg: bytes
    width: int
    height: int
    mime: str = "image/jpeg"
    _pil_cache: Optional[Image.Image] = field(
        default=None, repr=False, compare=False, init=False,
    )

    def to_pil(self) -> Image.Image:
        if self._pil_cache is None:
            self._pil_cache = Image.open(io.BytesIO(self.jpeg))
        return self._pil_cache


def _on_wayland() -> bool:
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


class ScreenCapture:
    def __init__(
        self,
        *,
        monitor_index: int = 1,
        max_edge_px: int = 768,
        jpeg_quality: int = 80,
        backend: str = "auto",
    ) -> None:
        self._monitor_index = monitor_index
        self._max_edge = max_edge_px
        self._quality = jpeg_quality
        # mss instances are not threadsafe; one per thread.
        self._sct: Optional[mss.mss] = None
        if backend == "auto":
            backend = "wayland" if _on_wayland() else "x11"
        # spectacle may not be installed at all; fall back to mss rather than
        # failing the whole vision loop.
        if backend == "wayland" and not shutil.which("spectacle"):
            backend = "x11"
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    def _ensure(self) -> mss.mss:
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab(self) -> Frame:
        if self._backend == "wayland":
            return self._grab_wayland()
        return self._grab_x11()

    def _grab_x11(self) -> Frame:
        sct = self._ensure()
        mon = sct.monitors[self._monitor_index]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        return self._encode(img)

    def _grab_wayland(self) -> Frame:
        # spectacle has no stdout mode on this build (it writes 0 bytes), so it has
        # to go through a temp file. -b background, -n no notification, -f whole
        # desktop (we crop to the monitor afterwards so the same knob works here).
        # JPEG instead of PNG on purpose: same pixels, but spectacle spends ~35%
        # less CPU encoding it and we decode 5x less data. Measured on this laptop:
        # 0.50s vs 0.77s of CPU per capture. The frame gets re-encoded to JPEG right
        # after anyway, so nothing downstream notices.
        fd, path = tempfile.mkstemp(prefix="wallie-shot-", suffix=".jpg")
        os.close(fd)
        try:
            proc = subprocess.run(
                ["spectacle", "-b", "-n", "-f", "-o", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            if proc.returncode != 0 or os.path.getsize(path) == 0:
                # Spectacle is single-instance: a second invocation hands the job to
                # the one already running and exits 0 having written nothing. That
                # looks identical to a real failure, so say which one it was.
                if proc.returncode == 0:
                    raise RuntimeError(
                        "spectacle no escribio nada (probablemente hay otra captura "
                        "en curso: es de instancia unica)"
                    )
                raise RuntimeError(f"spectacle salio con codigo {proc.returncode}")
            with open(path, "rb") as fh:
                img = Image.open(io.BytesIO(fh.read()))
                img.load()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        img = img.convert("RGB")
        box = self._monitor_box()
        if box:
            # The virtual desktop starts at (0,0) and spectacle hands us all of it,
            # so the monitor rectangle doubles as the crop rectangle.
            left, top, right, bottom = box
            left, top = max(0, left), max(0, top)
            right, bottom = min(img.width, right), min(img.height, bottom)
            if right - left > 16 and bottom - top > 16:
                img = img.crop((left, top, right, bottom))
        return self._encode(img)

    def _monitor_box(self) -> Optional[tuple[int, int, int, int]]:
        try:
            mon = self._ensure().monitors[self._monitor_index]
        except Exception:  # noqa: BLE001 — no geometry is not worth failing a frame
            return None
        return mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"]

    def _encode(self, img: Image.Image) -> Frame:
        w, h = img.size
        scale = min(1.0, self._max_edge / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._quality)
        return Frame(jpeg=buf.getvalue(), width=img.size[0], height=img.size[1])

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None



def downscale_jpeg(jpeg_bytes: bytes, max_edge: int = 512, quality: int = 50) -> bytes:
    """Re-encode JPEG at lower resolution/quality for LLM consumption."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
