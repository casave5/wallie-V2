"""Cross-session persistent memory: streamer notes + viewer interaction log."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

_MAX_LOG = 500


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.notes: str = ""
        self.viewer_log: list[dict[str, Any]] = []
        # "platform:username" de los que ya recibieron bienvenida, para no
        # repetirla en el siguiente stream. Distingue plataforma: un "chui54" de
        # TikTok no es el mismo "chui54" que uno de YouTube.
        self.greeted: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._path.exists():
            logger.debug(f"memory: no store at {self._path.name} — starting fresh")
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.notes = (data.get("notes") or "").strip()
            raw_log = data.get("viewer_log") or []
            self.viewer_log = raw_log[-_MAX_LOG:]
            self.greeted = set(data.get("greeted") or [])
            logger.info(
                f"memory: loaded {self._path.name} — "
                f"{len(self.notes)} chars notes, {len(self.viewer_log)} viewer entries, "
                f"{len(self.greeted)} ya saludados"
            )
        except Exception as e:
            logger.warning(f"memory: load failed (non-fatal): {e}")

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "notes": self.notes,
                "viewer_log": self.viewer_log[-_MAX_LOG:],
                "greeted": sorted(self.greeted)[-2000:],
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                f"memory: saved {self._path.name} — "
                f"{len(self.notes)} chars notes, {len(self.viewer_log)} viewer entries"
            )
        except Exception as e:
            logger.warning(f"memory: save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def update_notes(self, new_notes: str) -> None:
        self.notes = (new_notes or "").strip()

    def saludado_ya(self, platform: str, username: str) -> bool:
        """True si a este usuario de esta plataforma ya se ledio la bienvenida."""
        return f"{platform}:{username.strip().lstrip('@').lower()}" in self.greeted

    def marcar_saludado(self, platform: str, username: str) -> None:
        self.greeted.add(f"{platform}:{username.strip().lstrip('@').lower()}")
        if len(self.greeted) > 2000:
            self.greeted = set(sorted(self.greeted)[-1000:])

    def log_viewer(self, *, username: str, platform: str, text: str) -> None:
        self.viewer_log.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "username": username,
                "platform": platform,
                "text": text[:300],
            }
        )
        if len(self.viewer_log) > _MAX_LOG * 2:
            self.viewer_log = self.viewer_log[-_MAX_LOG:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def recent_viewers(self, n: int = 50) -> list[dict[str, Any]]:
        return self.viewer_log[-n:]

    def summary_for_prompt(self, max_chars: int = 800) -> str:
        if not self.notes:
            return ""
        return self.notes[:max_chars]
