"""OpenAI-compatible provider — covers OpenAI, Groq, and OpenRouter."""
from __future__ import annotations

import asyncio
import base64
from typing import Any, AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from .base import LLMError, LLMProvider

# A live stream can't afford a hung LLM call: bound every request and retry a couple
# of times on transient blips (connection drops, 5xx, malformed SSE) so one hiccup
# doesn't turn into dead air. Kept short because reactions must stay snappy.
_REQUEST_TIMEOUT_SEC = 25.0
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SEC = 0.4


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        supports_vision: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: float = _REQUEST_TIMEOUT_SEC,
        reasoning_effort: str = "",
        reasoning_transport: str = "flat",
    ) -> None:
        if not api_key:
            raise LLMError(f"{name}: missing API key")
        self.name = name
        self.model = model
        self.supports_vision = supports_vision
        # OpenRouter passes `cache_control` breakpoints through to providers that
        # support prompt caching (Anthropic, Gemini). Direct OpenAI/Groq reject the
        # field, so only enable it for OpenRouter. Caching the big static system
        # prompt removes it from prefill on repeat calls → lower latency, same quality.
        self._supports_cache = name == "openrouter"
        # Reasoning models (gpt-oss) spend max_tokens on hidden reasoning before the
        # answer. "low" keeps them snappy for a live stream; "" leaves it untouched.
        self._reasoning_effort = reasoning_effort or None
        self._reasoning_transport = reasoning_transport
        self._base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or None,
            timeout=timeout,               # configurable: local models on a shared GPU can be slow
            max_retries=0,                 # we do our own (stream-aware) retry below
        )

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
        payload = [self._encode_message(m) for m in messages]
        extra_body: dict[str, Any] | None = None
        if self._reasoning_effort:
            if self._reasoning_transport == "nested":
                extra_body = {"reasoning": {"effort": self._reasoning_effort}}
            else:
                extra_body = {"reasoning_effort": self._reasoning_effort}
        last_err: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            produced = False
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    extra_body=extra_body,
                    stream=True,
                )
                async for event in stream:
                    try:
                        delta = event.choices[0].delta
                        chunk = getattr(delta, "content", None)
                    except (IndexError, AttributeError):
                        chunk = None
                    if chunk:
                        produced = True
                        yield chunk
                return  # completed cleanly
            except Exception as e:  # noqa: BLE001 — connection drop, 5xx, malformed SSE…
                last_err = e
                # If we already streamed part of the answer, restarting would duplicate
                # text — bail. Otherwise retry a couple of times before giving up.
                if produced or attempt == _MAX_ATTEMPTS - 1:
                    break
                logger.warning(
                    f"{self.name}: transient LLM error (try {attempt + 1}/{_MAX_ATTEMPTS}), "
                    f"retrying: {str(e)[:120]}"
                )
                await asyncio.sleep(_BACKOFF_BASE_SEC * (2 ** attempt))
        raise LLMError(f"{self.name} request failed: {last_err}") from last_err

    async def aclose(self) -> None:
        await self._client.close()

    # ----- helpers -----
    def _encode_message(self, m: dict[str, Any]) -> dict[str, Any]:
        content = m.get("content")
        # Mark this message's text as a prompt-cache breakpoint (OpenRouter only).
        cache = bool(m.get("cache")) and self._supports_cache
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "image":
                    b64 = base64.b64encode(b["data"]).decode("ascii")
                    mime = b.get("mime", "image/jpeg")
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "low",
                            },
                        }
                    )
            if cache and blocks and blocks[-1].get("type") == "text":
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
            return {"role": m["role"], "content": blocks}
        if cache and content:
            return {
                "role": m["role"],
                "content": [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ],
            }
        return {"role": m["role"], "content": content or ""}
