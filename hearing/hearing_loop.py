"""Periodic hearing loop — captures system audio windows, transcribes speech,
measures loudness, and emits HearingEvents.

Mirrors vision.vision_loop: a background task that turns a raw stream (audio)
into discrete, meaningful events the orchestrator can fuse with vision. Silence
is skipped so Wallie only "hears" when there's something to hear.
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
import site
import time
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

# Sonda de nivel del microfono: se activa con HEARING_DEBUG_RMS=1 en el
# servicio. Sin esto es imposible distinguir "micro mudo" de "hablas muy bajo".
_DEBUG_RMS = os.getenv("HEARING_DEBUG_RMS") == "1"

from .capture import SystemAudioCapture

# Phrases Whisper notoriously hallucinates over silence / noise / a TV in the
# background. When the WHOLE transcript is just one of these, it's almost certainly
# not real speech aimed at Wallie — drop it so messy rooms don't trigger replies.
_HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "please subscribe", "subscribe to my channel", "like and subscribe",
    "see you next time", "i'll see you next time", "see you in the next video",
    "bye bye", "you", "okay", "so", "the end",
})


# Fichero que se "toca" para abrir/cerrar el oido desde un atajo global del
# escritorio (KDE -> Configuracion -> Atajos -> Atajos personalizados). Evita
# depender de una libreria de teclas globales: el bucle solo mira la fecha.
ATajo_OREJA = f"/run/user/{os.getuid()}/casavita-oreja"


def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().strip(" .!?,-…")
    return t in _HALLUCINATIONS


def _register_cuda_dll_dirs() -> None:
    """Put the pip-installed NVIDIA cuBLAS/cuDNN DLLs on Windows' DLL search path so
    CTranslate2 (faster-whisper) can load them. The wheels drop their DLLs under
    site-packages/nvidia/*/bin, which isn't searched by default — without this the
    GPU path dies at first inference with 'cublas64_12.dll not found'."""
    if not hasattr(os, "add_dll_directory"):
        return
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        for bindir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bindir)
            except OSError:
                pass


def _cudnn_present() -> bool:
    """True only if the cuDNN DLLs are actually installed. CTranslate2's CUDA path
    needs them; attempting CUDA without cuDNN can HANG model load, so we gate on this."""
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        if glob.glob(os.path.join(root, "nvidia", "cudnn", "bin", "cudnn*.dll")):
            return True
    return False


@dataclass
class HearingEvent:
    transcript: str            # what was said (may be "" for non-speech sound)
    loudness: float            # RMS 0..1 of the window
    has_speech: bool
    sound_type: str = "speech"  # speech | music | sound | quiet
    descriptor: str = ""        # for music/sound: e.g. "upbeat, energetic, bright"
    # Numeric musical mood (0 when not music) — lets the MoodEngine FEEL the music,
    # not just read a word in the prompt. valence -1..1, the rest 0..1.
    is_music: bool = False
    music_valence: float = 0.0
    music_arousal: float = 0.0
    music_energy: float = 0.0
    captured_at: float = field(default_factory=time.time)


class HearingLoop:
    def __init__(self, cfg, out_queue: "asyncio.Queue[HearingEvent]",
                 is_self_speaking: Optional[Callable[[], bool]] = None,
                 is_self_echo: Optional[Callable[[str], bool]] = None) -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._capture = SystemAudioCapture(
            samplerate=16000, device=getattr(cfg, "device", "") or None
        )
        self._model = None
        self._wake_model = None
        self._task: Optional[asyncio.Task] = None
        # Returns True if Wallie is currently speaking — those windows are skipped
        # so Wallie never transcribes/reacts to its own TTS (timing-based guard).
        self._is_self_speaking = is_self_speaking
        # Returns True if a transcript matches something Wallie recently SAID — the
        # definitive self-echo guard, immune to capture lag (content, not timing).
        self._is_self_echo = is_self_echo
        # --- Push-to-talk state ---
        # Push-to-talk keeps the ear shut until the streamer calls her by name, so she
        # doesn't butt into every aside. Once open, she stays open for a short
        # conversation and closes by herself when it goes quiet.
        self._wake = False            # is the ear open right now?
        self._opened_at = 0.0         # monotonic ts when the ear opened
        self._last_voice_at = 0.0     # monotonic ts of the last speech we let through
        self._wake_was_in_text = False  # drop the "Casavita" that opened us
        # Marca de la ultima vez que se vio el atajo (para no disparar con
        # ficheros viejos al arrancar).
        try:
            self._atajo_visto = os.path.getmtime(ATajo_OREJA)
        except OSError:
            self._atajo_visto = 0.0

    # --- push-to-talk ---
    @property
    def is_listening(self) -> bool:
        """True while the ear is open (push-to-talk conversation in progress)."""
        return self._wake

    def open_ear(self) -> None:
        """Open the ear explicitly (hotkey / external trigger)."""
        if not self._wake:
            logger.info("hearing: ear opened (push-to-talk)")
        self._wake = True
        self._opened_at = time.monotonic()
        self._last_voice_at = time.monotonic()

    def close_ear(self) -> None:
        """Shut the ear (mute) until called again."""
        if self._wake:
            logger.info("hearing: ear closed")
        self._wake = False

    def _wake_word_hit(self, text: str) -> bool:
        """Does this transcript contain a wake word?

        Whisper sometimes mis-transcribes the name ("casabita" for "Casavita",
        "casavila", etc.), so match on a tolerant signature: the leading "casa"
        stem plus a few letters after it, rather than the exact string.
        """
        low = (text or "").lower()
        for w in getattr(self._cfg, "wake_words", []) or []:
            if not w:
                continue
            if w in low:
                return True
            # Fuzzy: same first 4 letters + roughly the same length.
            stem = w[:4]  # "casa"
            if stem and stem in low and abs(len(w) - len(low)) < len(low) + 8:
                # Confirm it's the name, not the Spanish word "casa"/"casarse".
                idx = low.find(stem)
                tail = low[idx:idx + len(w) + 2]
                # Accept common mishearings: casabita, casavila, casabite, casavita...
                if re.match(rf"{stem}[bvlr]i?t?a?", tail) and len(tail) >= len(stem) + 2:
                    return True
        return False

    def _strip_wake(self, text: str) -> str:
        """Remove the leading wake word so she doesn't answer to her own name."""
        for w in getattr(self._cfg, "wake_words", []) or []:
            if not w:
                continue
            stem = w[:4]
            pat = re.compile(rf"(^|[\s,.!¿?]){re.escape(stem)}[a-z]{{1,7}}([\s,.!¿?]|$)", re.I)
            if pat.search(text):
                return pat.sub(" ", text, count=1).strip()
        return text.strip()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="hearing-loop")
            logger.info(
                "hearing: loop started (window={}s, model={}, silence<{})".format(
                    self._cfg.window_sec, self._cfg.model_size, self._cfg.silence_threshold
                )
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._capture.close()

    # --- internal ---
    def _push_to_talk_pass(self, text: str, has_speech: bool) -> bool:  # noqa: D401
        """Should this window be let through? Manages the ear's open/closed state.

        Returns False for anything heard while the ear is shut (no wake word yet).
        """
        cfg = self._cfg
        now = time.monotonic()

        # El nombre se ha oido en el trozo corto del principio (ver
        # _transcribe_bloques): vale igual que no aparezca en el texto, porque
        # el push-to-talk lo busca en el TEXTO y ahi no está.
        if getattr(self, "_wake_confirmado", False):
            if not self._wake:
                self.open_ear()
                logger.info("hearing: frase aceptada por el nombre del arranque")
            self._wake_confirmado = False
            return True

        # Wake word opens the ear, even mid-sentence ("oye Casavita, una pregunta").
        if has_speech and self._wake_word_hit(text):
            if not self._wake:
                self.open_ear()
                # No se pierde la frase: se quita el nombre y se sigue con el
                # resto. Antes se descartaba la frase entera y la pregunta
                # ("Casavita, ¿que tal?") se perdia.
                self._wake_was_in_text = True
                return True
            self._wake_was_in_text = False
            return True

        if not self._wake:
            return False  # ear shut — ignore everything

        # Ear is open. Decide whether this window is real speech or a stray sound.
        answer_all = getattr(cfg, "answer_all_when_open", True)
        if not answer_all and not has_speech:
            return False

        # Fresh speech keeps the conversation alive.
        if has_speech:
            self._last_voice_at = now

        # Close by herself when the streamer goes quiet or the cap is reached.
        if self._cerrar_si_toca(now):
            return False

        return True

    def _cerrar_si_toca(self, now: float | None = None) -> bool:
        """Cierra el oido por silencio prolonged or hard cap. True si se cerro.

        Vive aparte porque el bucle de VAD nunca pasa por _push_to_talk_pass
        cuando nadie habla, y sin esta comprobacion el oido se quedaba
        abierto para siempre en cuanto se abria una vez.
        """
        if not self._wake:
            return False
        cfg = self._cfg
        if now is None:
            now = time.monotonic()
        close_after = float(getattr(cfg, "close_after_silence_sec", 8.0) or 0.0)
        if close_after > 0 and (now - self._last_voice_at) > close_after:
            self.close_ear()
            return True
        cap = float(getattr(cfg, "listen_timeout_sec", 25.0) or 0.0)
        if cap > 0 and (now - self._opened_at) > cap:
            self.close_ear()
            return True
        return False

    def _comprobar_atajo(self) -> None:
        """Abre o cierra el oido si el atajo del escritorio ha sido pulsado."""
        try:
            marca = os.path.getmtime(ATajo_OREJA)
        except OSError:
            return
        if marca <= self._atajo_visto:
            return
        self._atajo_visto = marca
        if self._wake:
            self.close_ear()
            logger.info("hearing: atajo -> oido CERRADO")
        else:
            self.open_ear()
            logger.info("hearing: atajo -> oido ABIERTO")

    async def _tick_gate(self, loop, window: float, silent: bool = False) -> None:
        self._comprobar_atajo()
        """Sleep between polls, closing the ear if the conversation timed out
        while nobody was speaking (so 'Casavita' can still re-open it)."""
        await asyncio.sleep(window)
        if self._wake and silent:
            cap = float(getattr(self._cfg, "listen_timeout_sec", 25.0) or 0.0)
            if cap > 0 and (time.monotonic() - self._opened_at) > cap:
                self.close_ear()

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            self._model = await loop.run_in_executor(None, self._load_model)
            self._wake_model = await loop.run_in_executor(None, self._load_wake_model)
            self._capture.open()
            if getattr(self._cfg, "vad_segmentation", False):
                await self._run_vad(loop)
                return
            window = self._cfg.window_sec
            silence = self._cfg.silence_threshold
            from .audio_analysis import analyze_music, analyze_window
            # Let the ring buffer fill once before the first read.
            await asyncio.sleep(window)
            while True:
                # The capture thread drains the device non-stop, so we just grab the
                # most recent window — always current, never a stale/overflowed chunk.
                audio = self._capture.latest(window)
                if self._is_self_speaking is not None and self._is_self_speaking():
                    await asyncio.sleep(window)
                    continue  # Wallie was speaking — don't hear ourselves
                rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
                if rms < silence:
                    await self._tick_gate(loop, window, silent=True)
                    continue  # nothing worth hearing
                text = await loop.run_in_executor(None, self._transcribe, audio)
                # Guard against Whisper hallucinating a stray word over music/noise.
                has_speech = bool(text) and len(text.split()) >= 2
                if has_speech and self._is_self_echo is not None and self._is_self_echo(text):
                    logger.debug(f"hearing: muted self-echo — {text[:60]!r}")
                    await self._tick_gate(loop, window)
                    continue
                # --- push-to-talk gate ---
                if getattr(self._cfg, "push_to_talk", False):
                    allowed = self._push_to_talk_pass(text, has_speech)
                    if not allowed:
                        await self._tick_gate(loop, window)
                        continue
                    texto_con_nombre = text
                    text = self._strip_wake(text) if has_speech else text
                    has_speech = bool(text) and len(text.split()) >= 2
                    if self._wake_was_in_text and not has_speech:
                        # Era solo el nombre ("Casavita") en su propia ventana:
                        # la pregunta vendra en la siguiente, con el oido ya
                        # abierto. No emitsimos un evento de voz vacio.
                        self._wake_was_in_text = False
                        logger.info(f"hearing: solo el nombre ({texto_crudo!r}), esperando la pregunta")
                        await self._tick_gate(loop, window)
                        continue
                # Analyze the MUSICAL character over a longer slice (more stable tempo/key)
                # than the STT window, pulled straight from the rolling buffer.
                music_audio = self._capture.latest(min(window * 1.8, 9.0))
                sound_type, descriptor = analyze_window(
                    music_audio, 16000, has_speech=has_speech, silence=silence
                )
                if sound_type == "quiet":
                    await asyncio.sleep(window)
                    continue
                # Pull the numeric musical mood for anything musical — a song with
                # lyrics (→ speech) OR an instrumental (→ music). This is what lets the
                # MoodEngine FEEL the track, not just read a word in the prompt.
                feats = None
                if sound_type in ("speech", "music"):
                    feats = analyze_music(music_audio, 16000)
                vibe = feats.descriptor if (sound_type == "speech" and feats) else ""
                is_music_ev = feats is not None and (sound_type == "music" or bool(vibe))
                ev = HearingEvent(
                    transcript=text if has_speech else "",
                    loudness=rms, has_speech=has_speech,
                    sound_type=sound_type,
                    descriptor=(vibe if sound_type == "speech" else descriptor),
                    is_music=is_music_ev,
                    music_valence=(feats.valence if feats else 0.0),
                    music_arousal=(feats.arousal if feats else 0.0),
                    music_energy=(feats.energy if feats else 0.0),
                )
                self._enqueue(ev)
                if sound_type == "speech":
                    tag = f" ♪({vibe})" if vibe else ""
                    logger.info(f"hear>{tag} {text[:90]}")
                elif sound_type == "music":
                    logger.info(f"hear> ♪ music: {descriptor}")
                else:
                    logger.info(f"hear> (sonido: {descriptor})")
                await asyncio.sleep(window)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"hearing: loop crashed: {e}")

    async def _run_vad(self, loop) -> None:
        """Utterance-based hearing for live two-way conversation.

        Polls voice activity frequently; when the speaker pauses (or hits the cap),
        transcribes the WHOLE utterance at once and emits it. Result: replies fire
        right after the person stops talking, and full sentences aren't truncated.
        Speech-only (skips the music-character analysis the windowed path does).
        """
        silence = self._cfg.silence_threshold
        poll = max(0.1, self._cfg.poll_interval_sec)
        end_sil = max(0.2, self._cfg.end_silence_sec)
        max_utt = max(2.0, self._cfg.max_utterance_sec)
        probe = max(0.2, poll * 1.5)  # recent slice used for the voice/silence check
        await asyncio.sleep(0.5)

        speech_active = False
        speech_start = 0.0
        last_voice = 0.0
        _ultimo_sondeo = 0.0
        while True:
            now = time.time()
            self._comprobar_atajo()      # atajo del escritorio (abre/cierra el oido)
            self._cerrar_si_toca()       # el oido se cierra solo si toca
            # While Wallie is talking, don't capture our own voice as an utterance.
            if self._is_self_speaking is not None and self._is_self_speaking():
                speech_active = False
                await asyncio.sleep(poll)
                continue

            recent = self._capture.latest(probe)
            rms = float(np.sqrt(np.mean(recent ** 2))) if recent.size else 0.0
            # Sonda de nivel (HEARING_DEBUG_RMS=1): sirve para ver si la voz
            # LLEGA al buffer o no. Con el micro mudo y con la voz demasiado
            # baja se ven igual desde fuera, y eso confunde mucho.
            if _DEBUG_RMS and now - _ultimo_sondeo >= 1.0:
                _ultimo_sondeo = now
                seg = self._capture.latest(1.0)
                logger.info(
                    f"sondeo: pico={float(np.abs(seg).max()):.4f} "
                    f"rms={float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0:.5f} "
                    f"umbral={silence:.4f} "
                    f"{'VOZ' if seg.size and float(np.abs(seg).max()) > silence else 'nada'}"
                )
            voiced = rms >= silence
            if voiced:
                if not speech_active:
                    speech_active = True
                    speech_start = now
                last_voice = now
                # Mientras suene voz el oido esta vivo: sin esto se le cierra
                # a mitad de frase (el cierre por silencio solo miraba cuando
                # se cerraba una frase completa).
                self._last_voice_at = time.monotonic()

            should_flush = False
            if speech_active:
                if now - speech_start >= max_utt:
                    should_flush = True                      # safety cap on long talkers
                elif not voiced and now - last_voice >= end_sil:
                    should_flush = True                      # natural end-of-utterance pause

            if not should_flush:
                await asyncio.sleep(poll)
                continue

            # Grab from a touch before speech onset through to now (the trailing pause
            # is harmless — Whisper's VAD trims it), capped to the ring buffer length.
            grab = min(now - speech_start + 0.5, max_utt + 0.5, 9.5)
            audio = self._capture.latest(grab)
            speech_active = False

            text = await loop.run_in_executor(None, self._transcribe, audio)
            texto_crudo = (text or "").strip()
            has_speech = len(texto_crudo.split()) >= 2
            if not has_speech:
                # Un nombre solo ("Casavita") no es una frase, pero tiene que
                # ABRIR EL OIDO. Antes se tiraba en silencio, y por eso habia
                # que repetir la palabra: la primera vez se perdia.
                if texto_crudo and self._wake_word_hit(texto_crudo):
                    if not self._wake:
                        self.open_ear()
                        self._wake_was_in_text = True
                        logger.info(f"hearing: '{texto_crudo}' -> oido abierto, "
                                    f"esperando la pregunta")
                else:
                    logger.info(f"hearing: oido cerrado, frase muy corta: {texto_crudo!r}")
                await asyncio.sleep(poll)
                continue
            if self._is_self_echo is not None and self._is_self_echo(text):
                logger.debug(f"hearing: muted self-echo — {text[:60]!r}")
                await asyncio.sleep(poll)
                continue

            # --- push-to-talk gate ---
            # Sin esto el modo VAD emitia CUALQUIER cosa que oyera, sin mirar
            # la palabra de activacion: por eso contestaba a todo.
            if getattr(self._cfg, "push_to_talk", False):
                if not self._push_to_talk_pass(text, has_speech):
                    await asyncio.sleep(poll)
                    continue
                text = self._strip_wake(text)
                has_speech = bool(text) and len(text.split()) >= 2
                if not has_speech:
                    # Era solo el nombre ("Casavita") en su propia ventana: la
                    # pregunta vendra en la siguiente, con el oido ya abierto.
                    self._wake_was_in_text = False
                    logger.info(f"hearing: solo el nombre ({texto_crudo!r}), esperando la pregunta")
                    await asyncio.sleep(poll)
                    continue

            self._enqueue(HearingEvent(
                transcript=text, loudness=rms, has_speech=True,
                sound_type="speech", descriptor="",
            ))
            nivel = float(np.abs(audio).max()) if audio.size else 0.0
            logger.info(f"hear> [nivel {nivel:.2f}] {text[:90]}")
            await asyncio.sleep(poll)

    def _load_model(self):
        """Load Whisper on the best available device — GPU (fast, lets you run a
        bigger/more accurate model) with a clean fallback to CPU int8.

        The GPU path is *probed* with a tiny inference: CTranslate2 builds the model
        handle lazily, so a missing cublas/cudnn DLL only blows up on the first real
        transcribe. We trigger that here and fall back to CPU instead of crashing the
        loop mid-session."""
        from faster_whisper import WhisperModel
        size = self._cfg.model_size
        _register_cuda_dll_dirs()

        # Only attempt CUDA when the GPU AND cuDNN are both genuinely available —
        # a half-installed CUDA stack can HANG model load and silently kill hearing.
        use_cuda = False
        if _cudnn_present():
            try:
                import ctranslate2
                use_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                use_cuda = False

        if use_cuda:
            try:
                m = WhisperModel(size, device="cuda", compute_type="float16")
                list(m.transcribe(np.zeros(16000, dtype="float32"))[0])  # force CUDA init
                logger.info(f"hearing: Whisper '{size}' on CUDA (float16)")
                return m
            except Exception as e:
                logger.warning(f"hearing: CUDA load failed ({str(e)[:80]}) — using CPU")

        m = WhisperModel(size, device="cpu", compute_type="int8")
        logger.info(f"hearing: Whisper '{size}' on CPU (int8)")
        return m

    def _load_wake_model(self):
        """Modelo 'tiny' solo para detectar el nombre activador.

        Medido sobre audio real del usuario: "base" NO oye el nombre (0 de 2) y
        "small" lo oye pero tarda 11-17 s. "tiny" lo oye 2 de 2 y tarda ~0.5 s.
        El nombre ("Casavita") no esta en el diccionario de Whisper, y los
        modelos grandes se lo comen; el pequeno, al tener menos contexto, lo
       写出 tal cual porque es lo unico que hay en el trozo.
        """
        if not (getattr(self._cfg, "push_to_talk", False)
                and (getattr(self._cfg, "wake_words", None) or [])):
            return None
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("hearing: Whisper 'tiny' para el nombre activador")
            return m
        except Exception as e:  # noqa: BLE001
            logger.warning(f"hearing: no se pudo cargar 'tiny' para el nombre: {e}")
            return None

    def _transcribe(self, audio: "np.ndarray") -> str:  # noqa: D401
        t0 = time.time()
        lang = self._cfg.language or None
        # Sung vocals sit low under the instrumental — lift the window toward full
        # scale so Whisper gets a strong signal instead of a faint mumble.
        if audio.size:
            peak = float(np.abs(audio).max())
            if 0.0 < peak < 0.9:
                audio = (audio * (0.9 / peak)).astype("float32")
        if getattr(self._cfg, "low_latency", False):
            return self._transcribe_bloques(audio, lang, t0)
        else:
            segs, _info = self._model.transcribe(
                audio, language=lang,
                beam_size=5, best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6],   # fall back to softer decoding if stuck
                # Each window is independent - don't carry prior text, which makes Whisper
                # loop/hallucinate lyrics over music. Big accuracy win for songs.
                condition_on_previous_text=False,
                # Drop hallucinated garbage that music/noise provokes.
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.4),
            )
        texto = " ".join(s.text.strip() for s in segs).strip()
        # Cuanto tarda Whisper es LA latencia de Casavita: sin este numero no
        # se sabe si un retraso viene del micro, de Whisper o del LLM.
        logger.info(f"hearing: whisper {self._cfg.model_size} tardo {time.time() - t0:.1f}s "
                    f"en {len(audio)/16000:.1f}s de audio -> {texto[:60]!r}")
        return texto

    def _volcar_audio(self, audio: "np.ndarray", sr: int) -> None:
        """Guarda la frase oida en /tmp/opencode/voz (HEARING_DUMP_WAV=1).

        Sirve para comparar modelos sobre TU voz real en vez de suponer: el
        microfono USB solo deja que lo use uno a la vez, asi que no se puede
        grabar en paralelo al servicio.
        """
        if os.getenv("HEARING_DUMP_WAV") != "1" or not audio.size:
            return
        try:
            os.makedirs("/tmp/opencode/voz", exist_ok=True)
            with wave.open(f"/tmp/opencode/voz/{time.strftime('%H%M%S')}-"
                           f"{time.time()%1000:.0f}.wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                datos = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                w.writeframes(datos)
                logger.info(f"hearing: audio guardado ({len(datos)//2} muestras)")
        except Exception as e:  # noqa: BLE001
            logger.info(f"hearing: no se pudo volcar el audio: {e}")

    def _whisper_tiny(self, audio: "np.ndarray", lang, pista: "str | None") -> str:
        segs, _info = self._wake_model.transcribe(
            audio, language=lang,
            initial_prompt=pista,
            beam_size=1, temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,          # aqui solo hay voz: sin VAD no se come el nombre
            log_prob_threshold=-2.0,
            no_speech_threshold=0.9,
        )
        return " ".join(x.text.strip() for x in segs).strip()

    def _whisper(self, audio: "np.ndarray", lang, pista: "str | None") -> str:
        segs, _info = self._model.transcribe(
            audio, language=lang,
            initial_prompt=pista,
            beam_size=1, temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-2.0,
            no_speech_threshold=0.9,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250, threshold=0.3),
        )
        return " ".join(x.text.strip() for x in segs).strip()

    def _transcribe_bloques(self, audio: "np.ndarray", lang, t0: float) -> str:
        """Transcribe la frase entera, y aparte busca el nombre en un trozo corto.

        Por que dos pasadas: "Casavita" no esta en el diccionario de Whisper y
        en una frase larga se la come (la deja fuera del texto). Con el modelo
        "small" la oia, pero tardaba 5.8 s por frase. Truco: en un trozo de 1.5 s
        el nombre es casi todo lo que hay, y ahi si sale. Ese "si" se guarda en
        _wake_confirmado para que el push-to-talk acepte la frase aunque el
        nombre no aparezca en la transcripcion completa.

        El texto SIEMPRE se saca del audio entero (trocear y quedarse con el
        primer trozo perdia la pregunta: se la comia el VAD interno).
        """
        sr = 16000
        self._wake_confirmado = False
        self._volcar_audio(audio, sr)
        pista = ", ".join(getattr(self._cfg, "wake_words", None) or ["Casavita"])
        corte = int(1.5 * sr)   # el nombre se busca en el primer segundo y medio

        if not self._wake and audio.size > corte:
            arranque = (self._whisper_tiny(audio[:corte], lang, pista)
                        if self._wake_model is not None
                        else self._whisper(audio[:corte], lang, pista))
            if arranque and self._wake_word_hit(arranque):
                self._wake_confirmado = True
                logger.info(f"hearing: nombre detectado en el arranque: {arranque!r}")
            else:
                logger.info(f"hearing: 'tiny' no oyo el nombre: {arranque[:50]!r}")

        # SIEMPRE audio entero. Se probó transcribir solo el resto (mas rapido:
        # de 3 s a 1.5 s) y fue MUCHO peor: el corte a 1.3 s parte palabras
        # ("¿qué día es hoy?" -> "muy muy") y Cascavita contestaba tonterias.
        texto, via = self._whisper(audio, lang, pista), "entero"
        logger.info(f"hearing: whisper {self._cfg.model_size} tardo "
                    f"{time.time() - t0:.1f}s en {len(audio)/sr:.1f}s de audio "
                    f"[{via}] -> {texto[:70]!r}")
        return texto

    def _enqueue(self, ev: HearingEvent) -> None:
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(ev)
            except asyncio.QueueFull:
                pass
