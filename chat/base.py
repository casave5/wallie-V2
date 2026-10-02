"""Shared chat monitor interface."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

Platform = Literal["youtube", "twitch", "kick", "tiktok"]


@dataclass
class ChatMessage:
    platform: Platform
    username: str
    text: str
    ts: float = field(default_factory=time.time)
    is_highlight: bool = False  # super chat / bits / donation


class ChatMonitor(Protocol):
    platform: Platform

    async def start(self, out_queue: asyncio.Queue[ChatMessage]) -> None: ...
    async def stop(self) -> None: ...
