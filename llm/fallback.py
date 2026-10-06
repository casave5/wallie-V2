"""Try several providers in order until one produces output.

Needed because providers come and go: Gemini's free quota runs out, Groq drops
vision models, OpenRouter rate-limits the `:free` tiers. A vision turn should
survive any of those instead of dying.

Rules:
- Providers are tried in the order they were given.
- A provider that fails *before* emitting any token is skipped silently (one
  warning line) and the next one is tried.
- A provider that fails *after* emitting tokens raises: the caller already got
  part of a sentence, so retrying elsewhere would make Casavita stutter or
  repeat herself.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .base import LLMProvider

logger = logging.getLogger(__name__)


class FallbackProvider:
    """Wraps a list of providers, exposing the first one that works."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        usable = [p for p in providers if p is not None]
        if not usable:
            raise ValueError("FallbackProvider necesita al menos un proveedor")
        self._providers = usable
        self.name = "+".join(p.name for p in usable)
        self.model = " | ".join(p.model for p in usable)
        self.supports_vision = any(getattr(p, "supports_vision", False) for p in usable)

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 500,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        last_error: Exception | None = None
        for provider in self._providers:
            emitted = False
            try:
                async for chunk in provider.stream(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                ):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001 - we want to try the next one
                last_error = exc
                if emitted:
                    logger.warning(
                        "%s falló tras empezar a hablar (%s): no se reintenta para no repetir",
                        provider.name,
                        exc,
                    )
                    raise
                logger.warning(
                    "%s no pudo responder (%s); pruebo %s",
                    provider.name,
                    str(exc)[:120],
                    "otro proveedor" if provider is not self._providers[-1] else "nada más",
                )
        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        for provider in self._providers:
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001 - closing must never explode
                logger.debug("fallo cerrando %s", provider.name, exc_info=True)