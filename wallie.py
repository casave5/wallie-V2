"""Entrypoint — builds all subsystems from config and starts the pipeline."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

from loguru import logger

from audio import AudioPlayer
from chat import ChatManager
from config import Runtime, get_runtime
from core import Orchestrator, Persona, MemoryStore
from config import PROFILES_DIR
from llm import LLMProvider, build_provider, build_vision_provider
from tts import build_tts


class _LoguruBridge(logging.Handler):
    """Manda los logs del modulo `logging` estándar (hearing.capture) a loguru.

    Sin esto, esos mensajes se pierden: loguru y logging son sistemas
    distintos, y el "last resort" del logging solo enseña WARNING. Por eso
    durante horas no aparecia ni un INFO del micro y costaba adivinar.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)
    root = logging.getLogger()
    root.handlers[:] = [_LoguruBridge()]
    root.setLevel(logging.DEBUG)


def build_orchestrator(runtime: Optional[Runtime] = None) -> Orchestrator:
    runtime = runtime or get_runtime()
    cfg = runtime.config
    import os
    _configure_logger(os.getenv("LOG_LEVEL", "INFO"))

    persona = Persona.from_config(cfg.persona)
    llm = build_provider(cfg.llm, runtime.secrets)
    # Turns that carry a screenshot go to a second provider when one is configured:
    # the chat model is chosen for snappy text and usually can't see anything.
    vision_llm: Optional[LLMProvider] = None
    if cfg.vision.enabled and cfg.llm.vision_provider:
        try:
            vision_llm = build_vision_provider(cfg.llm, runtime.secrets)
            logger.info(
                f"vision: turns con imagen van a {vision_llm.name}:{vision_llm.model} "
                f"(el chat sigue en {llm.name}:{llm.model})"
            )
        except Exception as e:  # noqa: BLE001 — fall back to the chat provider
            logger.error(f"vision: no se pudo crear el proveedor de vision ({e}); se usa el de texto")
    tts = build_tts(cfg.tts, runtime.secrets)
    player = AudioPlayer(sample_rate=tts.sample_rate, channels=tts.channels,
                         blocksize=int(getattr(cfg.tts, "blocksize", 0) or 4096),
                         device=(cfg.tts.output_device or None))

    chat_manager: Optional[ChatManager] = None
    
    if cfg.chat.youtube_enabled or cfg.chat.twitch_enabled or cfg.chat.kick_enabled:
        chat_manager = ChatManager(cfg.chat, runtime.secrets)

    vision_queue = None
    vision_loop = None
    if cfg.vision.enabled:
        # Either the chat model itself can see, or a dedicated vision provider was
        # configured to look at the screenshots.
        if not (cfg.llm.vision_capable or cfg.llm.vision_provider):
            logger.warning("vision enabled but no vision-capable model is configured; disabling vision")
        else:
            try:
                from vision import VisionEvent, VisionLoop
            except ModuleNotFoundError as e:
                logger.error(
                    f"vision enabled but a dep is missing: {e}. "
                    "Install: pip install mss pillow imagehash"
                )
            else:
                vision_queue = asyncio.Queue(maxsize=4)
                vision_loop = VisionLoop(cfg.vision, vision_queue)

    hearing_queue = None
    hearing_loop = None
    if cfg.hearing.enabled:
        try:
            from hearing import HearingEvent, HearingLoop
        except ModuleNotFoundError as e:
            logger.error(
                f"hearing enabled but a dep is missing: {e}. "
                "Install: pip install soundcard faster-whisper"
            )
        else:
            hearing_queue = asyncio.Queue(maxsize=8)
            # Mute hearing for the capture window PLUS a tail margin: the ear grabs
            # the last `window_sec` of audio, so to be sure none of it contains Wallie's
            # own voice we also cover the playback that's still draining after the write.
            self_mute_window = cfg.hearing.window_sec + 2.5
            hearing_loop = HearingLoop(
                cfg.hearing, hearing_queue,
                is_self_speaking=lambda: player.speaking_recently(self_mute_window),
            )
            logger.info("hearing: enabled (microphone + STT, self-muted while speaking)")

    avatar = None
    if cfg.avatar.enabled:
        try:
            if cfg.avatar.backend == "web":
                from avatar import WebAvatar
                avatar = WebAvatar(cfg.avatar)
                asyncio.create_task(avatar.connect(), name="web-avatar")
                logger.info(f"avatar: Live2D web habilitado ({cfg.avatar.web_url})")
            else:
                from avatar import VTubeStudioAvatar
                avatar = VTubeStudioAvatar(cfg.avatar)
                asyncio.create_task(avatar.connect(), name="vts-avatar")
                logger.info(f"avatar: VTube Studio enabled ({cfg.avatar.vts_host}:{cfg.avatar.vts_port})")
        except Exception as e:
            logger.error(f"avatar: failed to start client: {e}")

    profile_name = cfg.profile_name or "default"
    memory_path = PROFILES_DIR / f"{profile_name}.memory.json"
    memory_store = MemoryStore(memory_path)

    return Orchestrator(
        runtime=runtime,
        persona=persona,
        llm=llm,
        tts=tts,
        player=player,
        chat_manager=chat_manager,
        vision_loop=vision_loop,
        vision_queue=vision_queue,
        vision_llm=vision_llm,
        avatar=avatar,
        memory_store=memory_store,
        hearing_loop=hearing_loop,
        hearing_queue=hearing_queue,
    )


async def run_cli() -> None:
    orch = build_orchestrator()
    await orch.start()
    try:
        # Block forever; the orchestrator's main task handles everything.
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orch.stop()


async def run_with_dashboard() -> None:
    from dashboard.server import serve

    await serve(None)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="wallie", description="AI streamer runtime")
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard instead of the headless loop",
    )
    args = ap.parse_args()

    try:
        if args.dashboard:
            asyncio.run(run_with_dashboard())
        else:
            asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
