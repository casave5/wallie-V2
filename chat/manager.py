"""Fan-in for all enabled chat platforms into a single asyncio queue."""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from config import ChatConfig, Secrets

from .base import ChatMessage, ChatMonitor


def _normaliza_nick(nick: str) -> str:
    """Para comparar nicks de bots. Sin '@', en minusculas y sin puntos:
    Streamer.bot a veces escribe "STREAMERBOT" y si no, se cuela."""
    return nick.strip().lstrip("@").lower().replace(".", "")


class ChatManager:
    def __init__(self, cfg: ChatConfig, secrets: Secrets) -> None:
        self._cfg = cfg
        self._secrets = secrets
        self.queue: asyncio.Queue[ChatMessage] = asyncio.Queue(maxsize=200)
        self._monitors: list[ChatMonitor] = []
        # Bots de alertas y overlays: no son personas, asi que ni se loguean ni se
        # contestan (si no, Casavita responde a "StreamElements haimilato 500 bits").
        self._ignorados: set[str] = {
            _normaliza_nick(u)
            for u in (cfg.ignore_usernames or [])
            if u and u.strip()
        }

    async def start(self) -> None:
        if self._cfg.youtube_enabled:
            try:
                from .youtube import YouTubeChatMonitor
                self._monitors.append(
                    YouTubeChatMonitor(
                        client_secret_file=self._secrets.youtube_client_secret_file,
                        live_chat_id=self._secrets.youtube_live_chat_id,
                        api_key=self._secrets.youtube_api_key,
                    )
                )
            except ModuleNotFoundError as e:
                logger.warning(
                    f"youtube chat enabled but google-api-python-client is missing: {e}; "
                    "skipping. Install: pip install google-api-python-client google-auth-oauthlib"
                )

        if self._cfg.twitch_enabled:
            try:
                from .twitch import TwitchChatMonitor
                self._monitors.append(
                    TwitchChatMonitor(
                        channel=self._secrets.twitch_channel,
                        oauth_token=self._secrets.twitch_oauth_token,
                        nick=self._secrets.twitch_nick,
                    )
                )
            except ModuleNotFoundError as e:
                logger.warning(f"twitch chat enabled but websockets is missing: {e}")

        if self._cfg.kick_enabled:
            try:
                from .kick import KickChatMonitor
                self._monitors.append(KickChatMonitor(channel=self._secrets.kick_channel))
            except ModuleNotFoundError as e:
                logger.warning(f"kick chat enabled but a dep is missing: {e}")

        for m in self._monitors:
            await m.start(self.queue)

    async def stop(self) -> None:
        for m in self._monitors:
            await m.stop()
        self._monitors.clear()

    def next_nowait(self) -> Optional[ChatMessage]:
        while True:
            try:
                msg = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
            if _normaliza_nick(msg.username or "") in self._ignorados:
                logger.debug(f"chat {msg.platform}: {msg.username} ignorado (bot)")
                continue
            # Una linea por mensaje: sin esto no hay forma de saber si el chat
            # de Twitch esta llegando de verdad (a INFO, no a DEBUG, porque es lo
            # que hay que mirar cuando "no me contesta al chat").
            logger.info(f"chat {msg.platform}: {msg.username} — {msg.text}")
            return msg

    def drain(self, max_items: int = 0) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        while max_items <= 0 or len(msgs) < max_items:
            try:
                msgs.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    @property
    def pending_count(self) -> int:
        return self.queue.qsize()
