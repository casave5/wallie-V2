"""Microphone capture, continuous + lag-free.

A background thread drains the microphone non-stop into a rolling ring buffer,
so the buffer never overflows while transcription is running. The processing loop
just grabs the most RECENT `window` seconds whenever it wants — always current
audio, no "data discontinuity", no stale audio.

This is Wallie's ear: the USER'S MICROPHONE, so Casavita can hear the streamer
talk back. It does NOT record the system mix (the stream audio).

Backend, in order of preference:

1. `parec` (PipeWire) — the only reliable path on Linux desktop. Captures from
   the PipeWire graph, so it honours EasyEffects (RNNoise + DeepFilter) exactly
   like OBS/the stream do. PortAudio/ALSA is *not* used: it bypasses PipeWire and
   lands on the raw hardware, which is mute here.
2. `sounddevice` (PortAudio) — used on macOS and as a fallback.

Default source: PipeWire's default input (`@DEFAULT_SOURCE@`), so if the user
switches headset/mic in the system mixer, Casavita follows automatically.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

logger = logging.getLogger("hearing.capture")

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dep
    sd = None

_IS_WINDOWS = sys.platform.startswith("win")
_DEFAULT_SOURCE = "@DEFAULT_SOURCE@"


def _list_pipewire_sources() -> list:
    """Input sources in PipeWire, as (name, description)."""
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[1], parts[2] if len(parts) > 2 else ""))
    return rows


def _list_sounddevice_inputs() -> list:
    """Input devices via PortAudio, as (index, name)."""
    out = []
    if sd is not None:
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    out.append((idx, dev["name"]))
        except Exception:  # noqa: BLE001
            pass
    return out


def _pipewire_source_to_card(source: str) -> "str | None":
    """Map a PipeWire source name to its ALSA card number, for arecord.

    e.g. "alsa_input.usb-Razer_Razer_BlackShark_V2_X_USB_000000000001-00.analog-stereo"
    -> "3" (so arecord can open hw:3,0).

    PipeWire mangles USB device names (spaces -> underscores, a unique id and
    profile suffix get appended), so we match on the distinctive middle words
    that survive in both /proc/asound/cards and arecord -l.
    """
    if not source:
        return None
    # Reduce to the identifying words: drop the alsa_input. prefix, the usb-
    # transport prefix, the id suffix and the profile suffix.
    needle = source
    for prefix in ("alsa_input.",):
        if needle.startswith(prefix):
            needle = needle[len(prefix):]
    if needle.startswith("usb-"):
        needle = needle[len("usb-"):]
    for suffix in (".analog-stereo", ".mono-fallback", ".stereo", ".mono"):
        if needle.endswith(suffix):
            needle = needle[: -len(suffix)]
            break
    # Strip the "_Razer_<id>-00" style transport/id noise.
    needle = needle.split("_000000000001")[0]
    # Underscores are spaces in the real name.
    words = [w for w in needle.split("_") if w and w not in ("Razer",)]

    try:
        with open("/proc/asound/cards", "r") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    for line in lines:
        if "[" not in line or "]" not in line:
            continue
        num = line.strip().split()[0]
        name = line.split("[", 1)[1].split("]", 1)[0].strip()
        # Match if any meaningful word from the source appears in the card name.
        if words and any(w.lower() in name.lower() for w in words):
            return num
    return None


class MicrophoneCapture:
    """Continuous microphone capture into a rolling buffer (thread-backed)."""

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        buffer_sec: float = 10.0,
        device: "str | None" = None,
    ) -> None:
        self._sr = samplerate
        self._ch = channels
        # None/empty -> PipeWire default input (follows the system mixer).
        self._device = device or None
        self._ring = np.zeros(int(samplerate * buffer_sec), dtype="float32")
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._running = False
        self._backend = ""
        # Por que fallo el ultimo backend (arecord lo dice en stderr y antes se
        # tiraba a DEVNULL, asi que no habia forma de saberlo).
        self._last_error = ""
        self._vigilante: "threading.Thread | None" = None

    # ------------------------------------------------------------------ setup
    def open(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="audio-capture", daemon=True
        )
        self._thread.start()

    @property
    def backend(self) -> str:
        """Which capture backend actually started ('parec' or 'sounddevice')."""
        return self._backend

    def _parec_device(self) -> str:
        """PipeWire source name to record from."""
        if self._device:
            # Allow either the full name or a substring of it.
            srcs = _list_pipewire_sources()
            for name, _desc in srcs:
                if self._device in name:
                    return name
            return self._device
        return _DEFAULT_SOURCE

    # --------------------------------------------------------------- backends
    @staticmethod
    def _leer_error(proc) -> str:
        """Primeras lineas del error de un proceso (arecord explica ahi todo)."""
        try:
            if proc.stderr is None:
                return ""
            try:
                txt = proc.stderr.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                return ""
            lineas = [l.strip() for l in txt.splitlines() if l.strip()]
            return " | ".join(lineas[:2])[:200]
        except Exception:  # noqa: BLE001
            return ""

    def _run_arecord(self) -> None:
        """Read the capture device straight from ALSA hardware with arecord.

        Two things this must get right, both learned the hard way on this box:

        1. The USB headset mic only opens at its NATIVE rate (48000 Hz). Asking
           ALSA for 16000 makes it fail to open at all.
        2. With an EasyEffects gate in the input chain, `parec` can hand us a
           full buffer of zeros even though the mic is live. arecord talks to
           the device itself and bypasses the filter chain — which is what STT
           wants, since Whisper does its own voice handling and a gate only
           clips our words.

        So: record at the hardware rate, downmix to mono, then resample to the
        rate the model wants.
        """
        dev = self._arecord_device()
        rate = self._hw_rate()
        self._backend = f"arecord({dev}@{rate})"
        proc = subprocess.Popen(
            [
                "arecord",
                "-D", dev,
                "-f", "S16_LE",
                "-r", str(rate),
                "-c", "2",          # the USB headset mic is a stereo profile
                "-t", "raw",
                "-q",
            ],
            # bufsize must stay positive AND small: a raw/0 pipe lets arecord
            # overrun and drop frames, which shows up as a mic that's "mute"
            # even though the hardware is fine. Read in small chunks.
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=16384,
        )
        # Si el dispositivo esta ocupado arecord muere enseguida: mejor notarlo.
        time.sleep(0.3)
        if proc.poll() is not None:
            self._last_error = self._leer_error(proc)
            raise RuntimeError(f"arecord no arranco: {self._last_error}")
        self._last_error = ""
        self._arrancado = True
        self._arrancar_vigilante()
        try:
            while self._running:
                raw = proc.stdout.read(16384)   # small blocks: no dropped frames
                if not raw:
                    break
                usable = len(raw) - (len(raw) % 4)
                if usable <= 0:
                    continue
                data = np.frombuffer(raw[:usable], dtype="<i2").astype("float32")
                data = data.reshape(-1, 2).mean(axis=1) / 32768.0  # downmix to mono
                self._push(self._resample(data, rate, self._sr))
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._last_error = self._leer_error(proc) or self._last_error

    def _hw_rate(self) -> int:
        """Native sample rate to record at. The headset rejects anything but 48k."""
        return 48000

    @staticmethod
    def _resample(data: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
        """Linear resample — plenty for speech at 48k -> 16k, and dependency-free."""
        if src_rate == dst_rate or data.size == 0:
            return data
        n_out = max(1, int(round(data.size * (dst_rate / src_rate))))
        idx = np.linspace(0.0, data.size - 1, n_out, dtype="float64")
        return np.interp(idx, np.arange(data.size), data).astype("float32")

    def _arecord_device(self) -> str:
        """ALSA device string for arecord.

        Accepts a PipeWire source name and maps it to the raw ALSA card, because
        arecord can't read PipeWire node names.
        """
        dev = self._device or ""
        # Already an ALSA device string (e.g. "hw:3,0").
        if dev.startswith("hw:") or dev.startswith("plughw:") or dev == "default":
            return dev
        # "alsa_input.usb-Razer_...-00.analog-stereo" -> card number via /proc.
        card = _pipewire_source_to_card(dev)
        if card:
            return f"hw:{card},0"
        return "default"

    def _run_parec(self) -> None:
        dev = self._parec_device()
        frames = int(self._sr * 0.25)
        cmd = [
            "parec",
            f"--device={dev}",
            f"--rate={self._sr}",
            f"--channels={self._ch}",
            "--format=s16le",
            "--latency-msec=40",
            "--process-time=20",
        ]
        self._backend = f"parec({dev})"
        want = frames * self._ch * 2  # s16 -> 2 bytes/sample
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=want
        )
        self._arrancado = True
        self._arrancar_vigilante()
        try:
            while self._running:
                raw = proc.stdout.read(want)
                if not raw:
                    break
                usable = len(raw) - (len(raw) % (self._ch * 2))
                if usable <= 0:
                    continue
                data = np.frombuffer(raw[:usable], dtype="<i2").astype("float32")
                data /= 32768.0
                if self._ch > 1:
                    data = data.reshape(-1, self._ch)[:, 0]
                self._push(data)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._last_error = self._leer_error(proc) or self._last_error

    def _run_sounddevice(self) -> None:
        self._backend = "sounddevice"
        self._arrancado = True
        self._arrancar_vigilante()
        chunk = max(1, int(self._sr * 0.25))
        idx = None
        if self._device and sd is not None:
            try:
                idx = int(self._device)
            except (TypeError, ValueError):
                idx = None
        with sd.InputStream(
            samplerate=self._sr, channels=self._ch, dtype="float32",
            blocksize=chunk, device=idx,
        ) as stream:
            while self._running:
                try:
                    data, _ = stream.read(chunk)
                except Exception:  # noqa: BLE001
                    break
                self._push(np.asarray(data, dtype="float32").flatten())

    def _push(self, data: "np.ndarray") -> None:
        n = data.shape[0]
        if n == 0:
            return
        with self._lock:
            if n >= self._ring.shape[0]:
                self._ring[:] = data[-self._ring.shape[0]:]
            else:
                self._ring[:-n] = self._ring[n:]
                self._ring[-n:] = data

    def _capture_loop(self) -> None:
        """Arranca el primer backend de captura que ENCIENDA de verdad.

        En este equipo hay dos trampas, las dos aprendidas a la fuerza:

        1. EasyEffects (RNNoise + DeepFilter + puerta) entrega a PipeWire un
           flujo de ceros aunque el micro este vivo; por eso se prueba primero
           el hardware (arecord) y se leen despues los errores de verdad.
        2. PipeWire agarra la tarjeta ALSA (hw:3,0) y arecord falla con
           "Dispositivo o recurso ocupado". Ahora eso es NORMAL, asi que se
           reintenta un poco y si no, se cae al nodo del micro en PipeWire,
           que funciona (parec).

        Y el criterio NO es que haya senal al arrancar: este headset entrega
        ceros digitales cuando nadie habla, asi que exigir RMS al inicio
        descartaba un micro perfectamente sano. Se elige el primer backend que
        arranca, y un vigilante avisa si luego no llega nada.
        """
        backends = []
        if shutil.which("arecord"):
            backends.append(self._run_arecord)
        if shutil.which("parec"):
            backends.append(self._run_parec)
        if sd is not None:
            backends.append(self._run_sounddevice)

        for backend in backends:
            intentos = 0
            while True:
                intentos += 1
                self._arrancado = False
                try:
                    backend()
                    break                      # se acabo (micro cerrado o fallo)
                except RuntimeError as e:
                    if ("ocupado" in str(e) or "busy" in str(e).lower()) and intentos <= 2:
                        logger.warning(
                            "capture: micro ocupado (lo tiene PipeWire), "
                            f"reintento {intentos}/2"
                        )
                        time.sleep(1.0)
                        continue
                    logger.warning(f"capture: fallo -> {e}")
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"capture: fallo -> {e}")
                    break

            if not self._arrancado:
                logger.warning(
                    f"capture: {self._backend or backend.__name__} no arranco "
                    f"({self._last_error or 'sin detalle'})"
                )
                continue

            return

        raise RuntimeError("no se pudo abrir el microfono con ningun backend")

    def _arrancar_vigilante(self) -> None:
        """Se llama justo cuando un backend abre el micro de verdad.

        Aqui (y no al final de _capture_loop) porque un backend sano se queda
        grabando horas: el final de _capture_loop solo se ve si algo se rompe.
        """
        if getattr(self, "_vigilante", None) is None:
            rms = float(np.sqrt(np.mean(self.latest(1.0) ** 2))) if self.latest(1.0).size else 0.0
            logger.info(
                f"capture: microfono abierto con {self._backend} "
                f"(rms ahora {rms:.5f}"
                f"{'; nadie habla todavia' if rms <= 0.0005 else ''})"
            )
            self._vigilante = threading.Thread(
                target=self._vigilante_silencio, name="mic-watch", daemon=True)
            self._vigilante.start()

    def _vigilante_silencio(self) -> None:
        """Aviso unico si el microfono lleva rato sin entregar nada.

        Es lo que mas cuesta diagnosticar: el microfono mudo y el microfono
        callado se ven igual desde fuera, y el servicio sigue "funcionando".
        """
        avisado = False
        while self._running:
            time.sleep(30)
            if not self._running:
                return
            datos = self.latest(5.0)
            if datos.size and float(np.abs(datos).max()) < 1e-4 and not avisado:
                logger.warning(
                    f"capture: el microfono ({self._backend}) lleva 30 s sin "
                    "entregar ni una muestra de voz. Si hablar no hace nada, "
                    "revisa el mute fisico del micro o EasyEffects."
                )
                avisado = True

    # ------------------------------------------------------------------ read
    def latest(self, seconds: float) -> "np.ndarray":
        """Most recent `seconds` of audio from the ring (always current)."""
        k = int(self._sr * seconds)
        with self._lock:
            return self._ring[-k:].copy()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    @property
    def sample_rate(self) -> int:
        return self._sr


# Backwards-compatible alias: the class used to be Windows-only system-audio
# loopback. It is now a microphone capture that works on Linux/macOS too.
SystemAudioCapture = MicrophoneCapture