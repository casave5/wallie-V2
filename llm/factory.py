"""Provider factory — maps LLMConfig + Secrets to a concrete LLMProvider."""
from __future__ import annotations

import logging
from typing import Any

from config import LLMConfig, Secrets

logger = logging.getLogger(__name__)

from .base import LLMError, LLMProvider
from .fallback import FallbackProvider


def _missing_sdk(name: str, pkg: str) -> LLMError:
    return LLMError(
        f"{name} provider selected but '{pkg}' is not installed. "
        f"Install it with: pip install {pkg}"
    )


def build_provider(cfg: LLMConfig, secrets: Secrets) -> LLMProvider:
    p = cfg.provider

    if p in ("openai", "groq", "openrouter"):
        try:
            from .openai_compat import OpenAICompatProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk(p, "openai") from e

        if p == "openai":
            return OpenAICompatProvider(
                name="openai",
                model=cfg.model,
                api_key=secrets.openai_api_key,
                supports_vision=cfg.vision_capable,
            )
        if p == "groq":
            return OpenAICompatProvider(
                name="groq",
                model=cfg.model,
                api_key=secrets.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                supports_vision=cfg.vision_capable,
                reasoning_effort=cfg.reasoning_effort,
            )
        return OpenAICompatProvider(
            name="openrouter",
            model=cfg.model,
            api_key=secrets.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            supports_vision=cfg.vision_capable,
            reasoning_effort=cfg.reasoning_effort,
            reasoning_transport=cfg.reasoning_transport,
            extra_headers={
                "HTTP-Referer": "https://github.com/wallie-ai/wallie",
                "X-Title": "Wallie",
            },
        )

    if p == "anthropic":
        try:
            from .anthropic import AnthropicProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk("anthropic", "anthropic") from e
        return AnthropicProvider(
            model=cfg.model,
            api_key=secrets.anthropic_api_key,
            supports_vision=cfg.vision_capable,
        )

    if p == "gemini":
        try:
            from .gemini import GeminiProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk("gemini", "google-generativeai") from e
        return GeminiProvider(
            model=cfg.model,
            api_key=secrets.gemini_api_key,
            supports_vision=cfg.vision_capable,
        )

    if p == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
            keep_alive=cfg.ollama_keep_alive,
            supports_vision=cfg.vision_capable,
        )

    raise LLMError(f"Unknown LLM provider: {p}")


def build_vision_provider(cfg: LLMConfig, secrets: Secrets) -> LLMProvider:
    """Provider for turns that carry a screenshot.

    Chains `vision_provider` then `vision_fallback_provider`, so an exhausted
    free quota degrades to the next provider instead of losing the turn. Falls
    back to the text provider when no dedicated vision one is configured, so a
    setup with a multimodal chat model (or vision off) needs no extra config.
    """
    chain: list[LLMProvider] = []

    for provider_name, model_name, is_fallback in (
        (cfg.vision_provider, cfg.vision_model, False),
        (cfg.vision_fallback_provider, cfg.vision_fallback_model, True),
    ):
        if not provider_name:
            continue
        extra: dict[str, Any] = {
            "provider": provider_name,
            "model": model_name or cfg.model,
            # Whatever the text model is, the vision model is by definition
            # asked to look at pictures.
            "vision_capable": True,
        }
        if is_fallback and cfg.vision_fallback_model:
            # The OpenRouter free tier only has heavy reasoning models left for
            # vision. Asking for "low" keeps them from burning the whole token
            # budget thinking and answering with nothing.
            extra["reasoning_effort"] = cfg.vision_fallback_reasoning_effort
            extra["reasoning_transport"] = "nested"
        vis = cfg.model_copy(update=extra)
        try:
            chain.append(build_provider(vis, secrets))
        except Exception as exc:  # noqa: BLE001 - e missing key must not kill vision
            logger.warning(
                "no se pudo preparar el proveedor de vision %s (%s)", provider_name, exc
            )

    if not chain:
        return build_provider(cfg, secrets)
    if len(chain) == 1:
        return chain[0]
    return FallbackProvider(chain)
