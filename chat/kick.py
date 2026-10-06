"""Kick.com chat monitor via Pusher websocket."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx
from loguru import logger

from .base import ChatMessage, ChatMonitor

_CHANNEL_API = "https://kick.com/api/v2/channels/{slug}"
_PUSHER_WSS = (
    "wss://ws-us2.pusher.com/app/eb1d5f283081a78b932c?protocol=7&client=js&version=7.6.0&flash=false"
)


class KickChatMonitor(ChatMonitor):
    platform = "kick"

    def __init__(self, *, channel: str) -> None:
        self._channel = channel.strip().lower()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self, out_queue: asyncio.Queue[ChatMessage]) -> None:
        if not self._channel:
            logger.warning("kick: no channel configured; monitor disabled")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(out_queue), name="kick-chat")
        logger.info(f"kick: chat monitor started on {self._channel}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self, out: asyncio.Queue[ChatMessage]) -> None:
        import websockets

        while not self._stop_event.is_set():
            chatroom_id: Optional[int] = None
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
                async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                    resp = await client.get(_CHANNEL_API.format(slug=self._channel))
                    resp.raise_for_status()
                    data = resp.json()
                    chatroom_id = data.get("chatroom", {}).get("id")
            except Exception as e:
                logger.warning(f"kick: channel lookup failed: {e}")

            if not chatroom_id:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                async with websockets.connect(_PUSHER_WSS, open_timeout=10) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "event": "pusher:subscribe",
                                "data": {"auth": "", "channel": f"chatrooms.{chatroom_id}.v2"},
                            }
                        )
                    )
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        msg = _parse_kick_event(str(raw))
                        if msg:
                            try:
                                out.put_nowait(msg)
                            except asyncio.QueueFull:
                                pass
            except Exception as e:
                logger.warning(f"kick: ws error: {e}, reconnecting in 5s")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass


_KICK_HIGHLIGHT_EVENTS = {
    "App\\Events\\SubscriptionEvent",
    "App\\Events\\GiftedSubscriptionsEvent",
}


def _parse_kick_event(raw: str) -> Optional[ChatMessage]:
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return None
    event = envelope.get("event", "")
    # Pusher manda "data" como texto JSON en el protocolo viejo y ya
    # deserializado en el nuevo: si se le pasa un dict a json.loads revienta
    # con "the JSON object must be str, bytes or bytearray, not dict".
    crudo = envelope.get("data", "{}")
    try:
        data = json.loads(crudo) if isinstance(crudo, (str, bytes, bytearray)) else crudo
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if event == "App\\Events\\ChatMessageEvent":
        user = data.get("sender", {}).get("username") or "viewer"
        text = data.get("content") or ""
        if not text:
            return None
        return ChatMessage(platform="kick", username=user, text=text)
    if event in _KICK_HIGHLIGHT_EVENTS:
        user = data.get("username") or data.get("gifter_username") or "viewer"
        text = data.get("message") or f"{user} just subscribed!"
        return ChatMessage(platform="kick", username=user, text=text, is_highlight=True)
    return None
