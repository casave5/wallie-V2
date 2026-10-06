"""Single pipeline — intent selection, LLM streaming, TTS, and audio playback."""
from __future__ import annotations

import asyncio
import random
import re
import struct
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Literal, Optional

from loguru import logger

from audio import AudioPlayer
from chat import ChatManager, ChatMessage
from config import AppConfig, Runtime
from core.context import Conversation, ImageBlock, pick_topic
from core.highlights import HighlightTracker
from core.persona import Persona
from llm import LLMProvider
from tts import TTSProvider
from utils.sentences import SentenceStreamer
from vision.capture import downscale_jpeg

if TYPE_CHECKING:
    from avatar import VTubeStudioAvatar
    from vision import VisionEvent, VisionLoop
    from hearing import HearingLoop, HearingEvent

from vision.scene_classifier import ChangeType, ScreenActivity
from vision.vision_memory import SceneMemory, UserBehaviorTracker
from core.attention import AttentionEngine, VisionDirective, VisionReaction
from core.mood import MoodEngine
from core.memory_store import MemoryStore
from core.live_activity import get_activity as _live_activity

IntentKind = Literal["chat", "vision", "monologue", "outro", "hearing"]


@dataclass
class Intent:
    kind: IntentKind
    chat: Optional[ChatMessage] = None
    vision: Optional[VisionEvent] = None
    urgent: bool = False
    vision_directive: Optional[VisionDirective] = None
    heard: str = ""  # for kind="hearing": the audio context to react to


class Orchestrator:
    def __init__(
        self,
        runtime: Runtime,
        persona: Persona,
        llm: LLMProvider,
        tts: TTSProvider,
        player: AudioPlayer,
        chat_manager: Optional[ChatManager] = None,
        vision_loop: Optional[VisionLoop] = None,
        vision_queue: Optional[asyncio.Queue[VisionEvent]] = None,
        vision_llm: Optional["LLMProvider"] = None,
        avatar: Optional["VTubeStudioAvatar"] = None,
        memory_store: Optional[MemoryStore] = None,
        hearing_loop: Optional["HearingLoop"] = None,
        hearing_queue: Optional["asyncio.Queue[HearingEvent]"] = None,
    ) -> None:
        self._runtime = runtime
        self._cfg: AppConfig = runtime.config
        self._persona = persona
        self._llm = llm
        self._tts = tts
        self._player = player
        self._chat = chat_manager
        self._vision_loop = vision_loop
        self._vision_queue = vision_queue
        # Dedicated provider for turns that carry a screenshot; None = reuse self._llm.
        self._vision_llm = vision_llm
        self._vision_ready_at: float = 0.0
        self._avatar = avatar
        self._memory = memory_store
        self._highlights = HighlightTracker(
            self._cfg.profile_name or "default",
            enabled=self._cfg.orchestrator.auto_highlight,
            threshold=self._cfg.orchestrator.highlight_threshold,
        )
        self._hearing_loop = hearing_loop
        self._hearing_queue = hearing_queue
        self._latest_heard: str = ""       # most recent transcript Wallie "hears"
        self._latest_heard_ts: float = 0.0
        self._last_reacted_heard: str = ""   # last audio context Wallie reacted to
        self._last_hearing_turn_ts: float = 0.0
        # Rolling log of (ts, normalized_text) Wallie recently SPOKE — used to reject
        # self-echo in the hearing transcript (our own TTS bleeding through loopback).
        self._recent_spoken: "deque[tuple[float, str]]" = deque(maxlen=24)
        # Wire the content-based self-echo guard into the ear we were handed.
        if self._hearing_loop is not None:
            self._hearing_loop._is_self_echo = self._is_self_echo

        oc = self._cfg.orchestrator
        self._conv = Conversation(
            max_messages=oc.max_messages,
            max_chars=oc.max_chars,
            recent_verbatim_turns=oc.recent_verbatim_turns,
        )
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._segment_task: Optional[asyncio.Task] = None
        self._summarizer_task: Optional[asyncio.Task] = None
        self._recent_topics: list[str] = []
        self._last_chat_reply_ts = 0.0
        self._ultimo_saludo_chat = 0.0
        # ultimo texto que escribimos en el chat de Twitch (para no spamear)
        self._ultimo_eco_chat = 0.0
        self._current_topic: Optional[str] = None
        self._last_spoken: str = ""
        self._last_monologue_spoken: str = ""
        self._last_intent_kind: str = ""
        self._session_start_ts: float = 0.0
        self._outro_done: bool = False
        self._segments_since_summary: int = 0
        self._open_threads: list[str] = []
        # Narrative spine: a rotating session goal the character pursues (feels alive).
        self._current_goal: Optional[str] = None
        self._goal_index: int = 0
        self._goal_started_ts: float = 0.0
        self._recent_themes: list[str] = []
        self._phrase_uses: dict[str, int] = {}
        # usernames ya vistos en el chat (para dar la bienvenida solo la primera vez)
        self._chat_vistos: set[str] = set()
        # últimas cosas que dijo casave, para mantener coherencia entre turnos
        self._nota_dueno: list[str] = []
        self._segments_ended_with_question: int = 0
        # Only consumed for vision-intent segments, never monologue.
        self._latest_frame: Optional["VisionEvent"] = None
        self._latest_frame_ts: float = 0.0
        self._scene_memory: SceneMemory = SceneMemory()
        self._behavior: UserBehaviorTracker = UserBehaviorTracker()
        self._mood = MoodEngine(base_energy=self._cfg.persona.energy)
        self._attention = AttentionEngine(
            organicity=self._cfg.vision.organicity,
            min_vision_react_interval=self._cfg.vision.min_vision_react_interval_sec,
            deep_base=self._cfg.vision.react_deep_base,
            glance_base=self._cfg.vision.react_glance_base,
            tangent_base=self._cfg.vision.react_tangent_base,
            ignore_base=self._cfg.vision.react_ignore_base,
            silence_base=self._cfg.vision.react_silence_base,
            active_content_boost=self._cfg.vision.active_content_boost,
        )
        self._last_vision_turn_ts: float = 0.0
        self._consecutive_skips: int = 0
        self._pending_directive: Optional[VisionDirective] = None
        self._last_segment_spoken_ts: float = time.time()
        self._target_silence_sec: float = random.uniform(5.0, 12.0)
        self._on_break = False
        self._next_break_at = float("inf")
        self._enrich_this_turn = False
        self._break_event: asyncio.Event = asyncio.Event()

    # --- lifecycle ---
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._session_start_ts = time.time()
        self._outro_done = False
        self._segments_since_summary = 0
        self._on_break = False
        self._break_event = asyncio.Event()
        self._schedule_next_break()
        if self._memory:
            self._memory.load()
        self._player.start()
        if self._chat:
            await self._chat.start()
        if self._vision_loop:
            delay = self._cfg.vision.startup_delay_sec
            if delay > 0:
                self._vision_ready_at = time.time() + delay
                async def _delayed_vision_start() -> None:
                    logger.info(f"orchestrator: vision starts in {delay:.0f}s — switch to your content now")
                    await asyncio.sleep(delay)
                    if self._running and self._vision_loop:
                        self._vision_loop.start()
                        logger.info("orchestrator: vision started")
                    self._vision_ready_at = 0.0
                asyncio.create_task(_delayed_vision_start(), name="vision-delay")
            else:
                self._vision_loop.start()
        if self._hearing_loop:
            self._hearing_loop.start()
            logger.info("orchestrator: hearing started")
        self._highlights.start()
        self._main_task = asyncio.create_task(self._run(), name="orchestrator")
        d = self._cfg.orchestrator.session_duration_min
        if d > 0:
            logger.info(f"orchestrator: started, session length {d:.1f} min")
        else:
            logger.info("orchestrator: started, unlimited session")

    async def stop(self) -> None:
        self._running = False
        for t in (self._segment_task, self._summarizer_task):
            if t and not t.done():
                t.cancel()
        if self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._main_task.cancel()
        if self._vision_loop:
            await self._vision_loop.stop()
        if self._hearing_loop:
            await self._hearing_loop.stop()
        if self._chat:
            await self._chat.stop()
        self._highlights.stop()
        self._player.close()
        await self._llm.aclose()
        if self._vision_llm is not None and self._vision_llm is not self._llm:
            await self._vision_llm.aclose()

        await self._tts.aclose()
        if self._memory:
            self._memory.save()
        logger.info("orchestrator: stopped")

    # --- status ---
    def status(self) -> dict[str, Any]:
        elapsed = time.time() - self._session_start_ts if self._session_start_ts else 0.0
        d = self._cfg.orchestrator.session_duration_min
        remaining_sec = max(0.0, d * 60.0 - elapsed) if d > 0 else None
        return {
            "running": self._running,
            "current_topic": self._current_topic,
            "last_spoken": self._last_spoken[-300:] if self._last_spoken else "",
            "audio_queue_sec": round(self._player.seconds_queued(), 2),
            "llm": f"{self._llm.name}:{self._llm.model}",
            "vision_llm": (
                f"{self._vision_llm.name}:{self._vision_llm.model}"
                if self._vision_llm is not None and self._vision_llm is not self._llm
                else None
            ),
            "tts": self._tts.name,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": round(remaining_sec, 1) if remaining_sec is not None else None,
            "segments_spoken": self._conv.total_segments,
            "session_notes_chars": len(self._conv.session_notes),
            "session_notes_preview": (self._conv.session_notes[:300] + "…")
                if len(self._conv.session_notes) > 300
                else self._conv.session_notes,
            "verbatim_msgs": len(self._conv.messages()),
            "open_threads": list(self._open_threads),
            "recent_themes": list(self._recent_themes),
            "user_behavior": self._behavior.current_pattern,
            "user_settled": self._behavior.is_settled,
            "browsing_pace": round(self._behavior.browsing_pace, 2),
            "vision_countdown": round(max(0.0, self._vision_ready_at - time.time()), 1) if self._vision_ready_at > 0 else 0,
            "on_break": self._on_break,
            "next_break_in_sec": (
                round(max(0.0, self._next_break_at - time.time()), 1)
                if self._next_break_at != float("inf") else None
            ),
        }

    # --- main loop ---
    async def _run(self) -> None:
        try:
            self._current_topic = self._pick_next_topic()

            if self._cfg.vision.enabled and self._vision_queue is not None:
                for _ in range(20):  # up to 2 seconds
                    if not self._vision_queue.empty():
                        break
                    await asyncio.sleep(0.1)
                if not self._vision_queue.empty():
                    logger.info("orchestrator: first vision frame ready")
                else:
                    logger.debug("orchestrator: no vision frame after 2s, starting anyway")

            while self._running:
                if self._should_outro() and not self._outro_done:
                    intent: Intent = Intent(kind="outro")
                    self._outro_done = True
                elif self._should_stop_immediately():
                    logger.info("orchestrator: session time up, stopping")
                    break
                else:
                    intent = await self._choose_intent()

                if intent.urgent and self._segment_task and not self._segment_task.done():
                    self._player.interrupt()
                    self._segment_task.cancel()
                    try:
                        await self._segment_task
                    except asyncio.CancelledError:
                        pass

                await self._cue_intent_expression(intent)

                self._segment_task = asyncio.create_task(self._run_segment(intent), name="segment")
                try:
                    await self._segment_task
                except asyncio.CancelledError:
                    pass

                if intent.kind == "outro":
                    await self._player.wait_drained()
                    logger.info("orchestrator: outro played, stopping")
                    break

                if (intent.kind == "vision" and intent.vision_directive
                        and intent.vision_directive.reaction not in (
                            VisionReaction.SILENCE, VisionReaction.IGNORE)):
                    await asyncio.sleep(random.uniform(1.5, 3.0))

                self._maybe_kick_summarizer()
                self._mood.tick()
                await self._sync_mood_to_avatar()

                lookahead = self._cfg.orchestrator.max_audio_lookahead_sec
                if not self._has_urgent_pending():
                    if lookahead > 0:
                        while (self._player.seconds_queued() > lookahead
                               and self._running
                               and not self._has_urgent_pending()):
                            await asyncio.sleep(0.1)
                        if self._player.seconds_queued() < 0.5:
                            gap = self._compute_breathing_gap()
                            if gap > 0:
                                await asyncio.sleep(gap)
                    else:
                        await self._player.wait_drained()

                if self._should_take_break():
                    await self._player.wait_drained()
                    await self._take_break()
            self._running = False
        except Exception as e:
            logger.exception(f"orchestrator: fatal error: {e}")

    # --- session timing ---
    def _should_outro(self) -> bool:
        d = self._cfg.orchestrator.session_duration_min
        if d <= 0 or not self._session_start_ts:
            return False
        remaining = d * 60.0 - (time.time() - self._session_start_ts)
        return remaining <= max(1.0, self._cfg.orchestrator.outro_seconds)

    def _should_stop_immediately(self) -> bool:
        d = self._cfg.orchestrator.session_duration_min
        if d <= 0 or not self._session_start_ts:
            return False
        return (time.time() - self._session_start_ts) >= d * 60.0 and self._outro_done

    # --- intent ---
    async def _choose_intent(self) -> Intent:
        self._drain_hearing()
        highlight = self._pop_highlight_chat()
        if highlight:
            self._mood.on_highlight_chat()
            return Intent(kind="chat", chat=highlight, urgent=True)

        # Conversational mode (e.g. VRChat): the person talking to you IS the priority.
        # Reply fast, ahead of any vision reaction. Only on speech (not music/sound events,
        # which _drain_hearing wraps in parentheses).
        if self._cfg.persona.conversational and self._cfg.hearing.enabled:
            heard = self._fresh_heard()
            now_h = time.time()
            is_speech = bool(heard) and not heard.startswith("(")
            if (is_speech and heard != self._last_reacted_heard
                    and now_h - self._last_hearing_turn_ts >= self._cfg.hearing.reply_gate_sec):
                self._last_reacted_heard = heard
                self._last_hearing_turn_ts = now_h
                self._mood.on_scene_change()
                return Intent(kind="hearing", heard=heard, urgent=True)

        # PLAY MODE: when Wallie is playing (e.g. Minecraft), the commentary is grounded in the
        # agent's REAL actions and runs on its own cadence — fully independent of the Vision toggle.
        # So Play works even with Vision OFF, and Vision/Hearing/Play each switch on/off separately.
        play = getattr(self._cfg, "play", None)
        if play is not None and play.enabled and play.talk_from_agent and _live_activity():
            if self._vision_loop is not None:
                self._pop_latest_vision()            # drain so the loop can't back up; frame unused
            now_t = time.time()
            since_last_spoken = now_t - self._last_segment_spoken_ts
            since_last_vision = now_t - self._last_vision_turn_ts if self._last_vision_turn_ts > 0 else 9999.0
            hard_floor = self._cfg.vision.min_vision_react_interval_sec * 0.65
            if (self._cfg.vision.spontaneous_monologue
                    and since_last_spoken >= self._target_silence_sec
                    and since_last_vision >= hard_floor
                    and self._mood.wants_vision_engagement() > 0.25):
                roll = random.random()
                if roll < 0.30:
                    reaction, sentences = VisionReaction.DEEP, random.choice([2, 3, 3])
                elif roll < 0.45:
                    reaction, sentences = VisionReaction.TANGENT, random.choice([2, 2, 3])
                else:
                    reaction, sentences = VisionReaction.GLANCE, random.choice([1, 2, 2])
                directive = VisionDirective(reaction=reaction, target_sentences=sentences,
                                            rationale="play: grounded agent commentary", is_fallback=True)
                self._pending_directive = directive
                logger.info(f"play: commentary ({reaction.value}) after {since_last_spoken:.0f}s")
                return Intent(kind="vision", vision=self._grab_play_event(), urgent=False,
                              vision_directive=directive)
            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0, rationale="play: quiet beat"))

        vision = self._pop_latest_vision()
        if vision is not None:
            vcfg = self._cfg.vision
            is_active_content = vision.activity.value in ("media", "app_switch")
            hard_floor = (
                vcfg.min_vision_react_interval_sec * 0.40 if is_active_content
                else vcfg.min_vision_react_interval_sec * 0.65
            )
            since_last_vision = time.time() - self._last_vision_turn_ts
            if self._last_vision_turn_ts > 0 and since_last_vision < hard_floor:
                logger.debug(
                    f"vision: hard cooldown ({since_last_vision:.1f}s < "
                    f"{hard_floor:.1f}s, active_content={is_active_content}), skipping"
                )
                vision = None

            if vision is not None:
                interest = self._estimate_interest(vision)
                threshold = self._cfg.vision.min_engagement_for_react
                if interest < threshold:
                    logger.debug(
                        f"vision: interest {interest:.2f} < {threshold} "
                        f"(activity={vision.activity.value}), auto-IGNORE"
                    )
                    self._attention.on_vision_ignored()
                    vision = None
                    directive = None
                else:
                    directive = self._decide_vision_reaction(vision)
                    self._pending_directive = directive
                    logger.info(
                        f"vision step2: {directive.reaction.value} "
                        f"(interest={interest:.2f}, activity={vision.activity.value}, "
                        f"change={vision.change_type.value}) — {directive.rationale}"
                    )
            else:
                directive = None

            if directive is not None and directive.reaction == VisionReaction.SILENCE:
                self._mood.on_silence_beat()
                self._attention.on_silence_beat()
                return Intent(kind="monologue", vision=None,
                              vision_directive=VisionDirective(
                                  reaction=VisionReaction.SILENCE,
                                  target_sentences=0,
                                  rationale=directive.rationale,
                              ))

            if directive is not None and directive.reaction == VisionReaction.IGNORE:
                self._attention.on_vision_ignored()
            elif directive is not None and directive.reaction in (
                VisionReaction.DEEP, VisionReaction.GLANCE, VisionReaction.TANGENT,
            ):
                self._mood.on_scene_change()
                self._attention.on_scene_change()
                return Intent(kind="vision", vision=vision, urgent=False,
                              vision_directive=directive)

        ordinary = self._pop_ordinary_chat()
        if ordinary:
            self._mood.on_ordinary_chat()
            return Intent(kind="chat", chat=ordinary, urgent=False)

        # HEARING reaction — when there's fresh, NEW audio context and nothing more
        # urgent took priority, react to what Wallie hears (just like vision reacts to
        # what it sees). This is what drives pure-hearing mode instead of blank monologue.
        if self._cfg.hearing.enabled:
            heard = self._fresh_heard()
            now_h = time.time()
            if (heard and heard != self._last_reacted_heard
                    and now_h - self._last_hearing_turn_ts >= self._cfg.hearing.reply_gate_sec):
                self._last_reacted_heard = heard
                self._last_hearing_turn_ts = now_h
                self._mood.on_scene_change()
                return Intent(kind="hearing", heard=heard, urgent=False)

        if self._cfg.vision.organic_vision and self._attention.should_hold_silence(
            mood_silence_probability=self._mood.silence_probability(),
            in_monologue_flow=self._segments_in_monologue_flow() >= 2,
        ):
            self._mood.on_silence_beat()
            self._attention.on_silence_beat()
            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0,
                rationale="silence-beat: monologue chain too long",
            ))

        self._enrich_this_turn = False
        vcfg = self._cfg.vision
        if vcfg.enabled and self._vision_loop is not None:
            now_t = time.time()
            since_last_spoken = now_t - self._last_segment_spoken_ts
            since_last_vision = now_t - self._last_vision_turn_ts if self._last_vision_turn_ts > 0 else 9999.0
            hard_floor = vcfg.min_vision_react_interval_sec * 0.65
            if (
                vcfg.spontaneous_monologue
                and since_last_spoken >= self._target_silence_sec
                and since_last_vision >= hard_floor
                and self._latest_frame is not None
                and self._mood.wants_vision_engagement() > 0.3
            ):
                evt = self._latest_frame
                age = time.time() - self._latest_frame_ts if self._latest_frame_ts else 9999.0
                if age > vcfg.max_frame_age_sec:
                    fresh = await self._vision_loop.grab_now()
                    if fresh is not None:
                        evt = fresh
                        self._latest_frame = fresh
                        self._latest_frame_ts = time.time()
                fb_energy = (self._mood.arousal + self._mood.talkativity) / 2.0
                fb_roll = random.random()
                if fb_roll < 0.25 and fb_energy > 0.45:
                    fb_reaction = VisionReaction.DEEP
                    fb_sentences = random.choice([3, 4, 4, 5])
                elif fb_roll < 0.40:
                    fb_reaction = VisionReaction.TANGENT
                    fb_sentences = random.choice([2, 3, 3])
                else:
                    fb_reaction = VisionReaction.GLANCE
                    fb_sentences = random.choice([1, 2, 2, 3])
                directive = VisionDirective(
                    reaction=fb_reaction,
                    target_sentences=fb_sentences,
                    rationale=f"fallback: silence too long → {fb_reaction.value}",
                    is_fallback=True,
                )
                self._pending_directive = directive
                logger.info(
                    f"vision: fallback {fb_reaction.value} after {since_last_spoken:.0f}s "
                    f"(target was {self._target_silence_sec:.0f}s)"
                )
                return Intent(kind="vision", vision=evt, urgent=False,
                              vision_directive=directive)

            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0,
                rationale="vision-mode: quiet between reactions",
            ))

        # Pure-hearing mode (hearing on, no vision): do NOT fall back to a blank
        # monologue — that makes Wallie talk non-stop, which keeps self-mute engaged
        # and the mic never actually hears anything. Instead stay quiet and LISTEN, so
        # self-mute clears, the ear catches audio, and the next loop reacts to it.
        if self._cfg.hearing.enabled:
            self._mood.on_silence_beat()
            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0,
                rationale="hearing-mode: quiet, listening between reactions",
            ))

        if not self._cfg.vision.spontaneous_monologue:
            # Aun muda por iniciativa propia, sí puede preguntarle a casave cada
            # tanto ("¿qué haces ahora?") si el chat lleva rato callado.
            ccfg = self._cfg.chat
            if (
                ccfg.check_in_probability > 0.0
                and self._chat is not None
                and time.time() - self._last_segment_spoken_ts >= ccfg.check_in_min_idle_sec
                and random.random() < ccfg.check_in_probability
            ):
                return Intent(kind="monologue", vision_directive=VisionDirective(
                    reaction=VisionReaction.TANGENT, target_sentences=1,
                    rationale="check-in: preguntar a casave que esta haciendo",
                ))
            self._mood.on_silence_beat()
            self._attention.on_silence_beat()
            return Intent(kind="monologue", vision_directive=VisionDirective(
                reaction=VisionReaction.SILENCE, target_sentences=0,
                rationale="spontaneous_monologue off: callada hasta que le hablen",
            ))

        return Intent(kind="monologue")

    def _grab_play_event(self) -> "VisionEvent":
        """Grab a FRESH game screenshot for Play-mode commentary, with its own capture so it works
        even when the Vision toggle is OFF. This is what lets Wallie actually SEE what it's playing
        (grounding the line in the real screen, not just the agent's text note)."""
        try:
            if getattr(self, "_play_cap", None) is None:
                from vision.capture import ScreenCapture
                self._play_cap = ScreenCapture(
                    monitor_index=self._cfg.vision.monitor_index, max_edge_px=768, jpeg_quality=60)
            frame = self._play_cap.grab()
            from vision.vision_loop import VisionEvent
            return VisionEvent(frame=frame, changed=True,
                               change_type=ChangeType.DELTA, activity=ScreenActivity.MEDIA)
        except Exception as e:
            logger.debug(f"play: screen grab failed ({e}); commenting from note only")
            return self._play_vision_event()

    def _play_vision_event(self) -> "VisionEvent":
        """Fallback stand-in event when no real frame can be grabbed (blank frame -> note-only line)."""
        if self._latest_frame is not None:
            return self._latest_frame
        from vision.vision_loop import VisionEvent
        from vision.capture import Frame
        return VisionEvent(frame=Frame(jpeg=b"", width=1, height=1), changed=True,
                           change_type=ChangeType.DELTA, activity=ScreenActivity.STATIC)

    # --- interest pre-filter ---
    def _estimate_interest(self, vision: "VisionEvent") -> float:
        _ACTIVITY_SCORES: dict[str, float] = {
            "static": 0.10,
            "micro": 0.05,
            "scroll": 0.20,
            "typing": 0.00,
            "navigation": 0.50,
            "app_switch": 0.70,
            "media": 0.60,
        }
        if vision.change_type == ChangeType.SCENE_CHANGE:
            return 1.0
        score = _ACTIVITY_SCORES.get(vision.activity.value, 0.15)
        if vision.user_pattern == "settled":
            score += 0.15
        if self._behavior.is_rapid_browsing:
            score *= 0.30
        return min(1.0, score)

    # --- vision policy ---
    def _decide_vision_reaction(self, vision: "VisionEvent") -> VisionDirective:
        change_kind = "scene" if vision.change_type == ChangeType.SCENE_CHANGE else "delta"
        # Mid-thread: downgrade DELTA so open thread can finish.
        force_glance = bool(self._open_threads) and change_kind == "delta"

        if not self._cfg.vision.organic_vision:
            return VisionDirective(
                reaction=VisionReaction.DEEP, target_sentences=3,
                rationale="organic_vision off → DEEP",
            )

        directive = self._attention.decide_on_vision(
            change_kind=change_kind,
            scene_age_sec=self._scene_memory.scene_age_sec(),
            mood_arousal=self._mood.arousal,
            mood_focus=self._mood.focus,
            mood_talkativity=self._mood.talkativity,
            vision_engagement=self._mood.wants_vision_engagement(),
            has_topic=bool(self._current_topic),
            in_monologue_flow=self._segments_in_monologue_flow() >= 2,
            segments_total=self._conv.total_segments,
            tangent_seeds=list(self._cfg.persona.running_gags or []),
            screen_activity=vision.activity.value,
            user_pattern=vision.user_pattern or self._behavior.current_pattern,
            user_settled=self._behavior.is_settled,
            rapid_browsing=self._behavior.is_rapid_browsing,
        )
        if force_glance and directive.reaction == VisionReaction.DEEP:
            directive = VisionDirective(
                reaction=VisionReaction.GLANCE,
                target_sentences=1,
                glance_style=directive.glance_style,
                rationale="mid-thread defer: DEEP→GLANCE so open thread can close",
            )
        return directive

    def _segments_in_monologue_flow(self) -> int:
        count = 0
        for m in reversed(self._conv.messages()):
            if m.role != "assistant":
                continue
            if m.source.startswith("monologue"):
                count += 1
            else:
                break
        return count

    def _has_urgent_pending(self) -> bool:
        return self._pop_highlight_chat_peek() is not None

    # --- silence & breathing ---
    def _compute_next_silence_target(self) -> float:
        energy = (self._mood.arousal + self._mood.talkativity) / 2.0
        if energy > 0.65:
            lo, hi = 3.0, 8.0
        elif energy > 0.45:
            lo, hi = 6.0, 14.0
        elif energy > 0.3:
            lo, hi = 10.0, 20.0
        else:
            lo, hi = 15.0, 28.0
        base = random.uniform(lo, hi)
        jitter = base * random.uniform(-0.25, 0.25)
        target = max(2.0, base + jitter) * self._cfg.vision.silence_fallback_scale
        logger.debug(f"silence target: {target:.1f}s (energy={energy:.2f})")
        return target

    def _compute_breathing_gap(self) -> float:
        oc = self._cfg.orchestrator
        lo = oc.min_inter_segment_gap_sec
        hi = oc.breathing_gap_max_sec
        if hi <= lo:
            return lo
        t = 1.0 - self._mood.talkativity
        gap = lo + (hi - lo) * t
        gap *= 0.8 + random.random() * 0.4
        return max(lo, min(hi, gap))

    def _schedule_next_break(self) -> None:
        oc = self._cfg.orchestrator
        if not oc.enable_breaks:
            self._next_break_at = float("inf")
            return
        base = oc.break_every_min * 60.0
        jitter = base * oc.break_every_jitter
        self._next_break_at = time.time() + base + random.uniform(-jitter, jitter)
        logger.debug(f"orchestrator: next break in {self._next_break_at - time.time():.0f}s")

    def _should_take_break(self) -> bool:
        if not self._cfg.orchestrator.enable_breaks or self._on_break:
            return False
        return time.time() >= self._next_break_at

    async def _take_break(self) -> None:
        oc = self._cfg.orchestrator
        duration = random.uniform(oc.break_min_sec, oc.break_max_sec)
        duration *= 0.7 + 0.6 * (1.0 - self._mood.arousal)
        self._on_break = True
        self._break_event.clear()
        logger.info(f"orchestrator: taking a break ({duration:.1f}s)")

        deadline = time.time() + duration
        while time.time() < deadline and self._running:
            if self._has_urgent_pending():
                logger.info("orchestrator: break interrupted by urgent event")
                break
            try:
                await asyncio.wait_for(self._break_event.wait(), timeout=0.5)
                logger.info("orchestrator: break ended (manual resume)")
                break
            except asyncio.TimeoutError:
                pass

        self._on_break = False
        self._mood.on_silence_beat()
        self._schedule_next_break()
        logger.info("orchestrator: break over, resuming")

    def trigger_break(self) -> None:
        """Force an immediate break (called from dashboard)."""
        self._next_break_at = 0.0

    def resume_from_break(self) -> None:
        """End the current break early (called from dashboard)."""
        if self._on_break:
            self._break_event.set()

    def _pop_highlight_chat_peek(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        drained: list[ChatMessage] = []
        while True:
            msg = self._chat.next_nowait()
            if msg is None:
                break
            drained.append(msg)
        highlight = next((m for m in drained if m.is_highlight), None)
        for m in drained:
            self._avisar_saludo_chat(m)
            try:
                self._chat.queue.put_nowait(m)
            except asyncio.QueueFull:
                break
        return highlight

    def _avisar_saludo_chat(self, msg: ChatMessage) -> None:
        """Saluda con la mano si alguien escribe un saludo en el chat.

        Se llama desde el peek, que ve todos los mensajes sin decidir si
        Casavita contesta, asi que saluda tambien a los que no responden.
        """
        texto = (msg.text or "").strip().lower()
        if not texto or len(texto) > 40:
            return
        limpio = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
        if not _SALUDO_CHAT.search(limpio):
            return
        ahora = time.monotonic()
        if ahora - self._ultimo_saludo_chat < 4.0:
            return
        self._ultimo_saludo_chat = ahora
        logger.info("saludo en el chat de %s: %s", msg.platform, msg.username)
        try:
            asyncio.get_running_loop().create_task(
                self._avatar_safe_call("trigger_expression", "saludo")
            )
        except RuntimeError:
            pass

    def _pop_highlight_chat(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        keep: list[ChatMessage] = []
        found: Optional[ChatMessage] = None
        while True:
            msg = self._chat.next_nowait()
            if msg is None:
                break
            if found is None and msg.is_highlight:
                found = msg
            else:
                keep.append(msg)
        for m in keep:
            try:
                self._chat.queue.put_nowait(m)
            except asyncio.QueueFull:
                break
        return found

    def _usuarios_favoritos(self) -> set[str]:
        """Nombres (en minúsculas, sin '@') que siempre reciben respuesta."""
        return {
            u.strip().lstrip("@").lower()
            for u in (self._cfg.chat.always_reply_usernames or [])
            if u and u.strip()
        }

    def _pop_ordinary_chat(self) -> Optional[ChatMessage]:
        if not self._chat:
            return None
        ahora = time.monotonic()
        favoritos = self._usuarios_favoritos()
        # El dueño del canal siempre entra primero y sin dados ni enfriamiento.
        if favoritos and (ahora - self._last_chat_reply_ts) < self._cfg.chat.min_reply_interval_sec:
            msg = self._chat.next_nowait()
            if msg is None:
                return None
            if (msg.username or "").lower().lstrip("@") in favoritos:
                logger.info(f"chat: {msg.username} (favorito) — respuesta siempre")
                return msg
            self._chat.queue.put_nowait(msg)

        now = time.time()
        if now - self._last_chat_reply_ts < self._cfg.chat.min_reply_interval_sec:
            return None
        max_age = self._cfg.chat.max_message_age_sec
        while True:
            msg = self._chat.next_nowait()
            if msg is None:
                return None
            if now - msg.ts <= max_age:
                break
            logger.debug(f"chat: dropping stale message from {msg.username} ({now - msg.ts:.0f}s old)")
        if (msg.username or "").lower().lstrip("@") in favoritos:
            logger.info(f"chat: {msg.username} (favorito) — respuesta siempre")
            return msg
        if random.random() >= self._cfg.chat.reply_probability:
            return None
        return msg

    # --- hearing (multimodal fusion) ---
    def _drain_hearing(self) -> None:
        """Pull all pending HearingEvents and keep the freshest transcript as the
        'what Wallie is hearing right now' context. This is fused into vision and
        monologue turns so sight and sound are organized into one coherent reaction
        — never two competing ones."""
        if self._hearing_queue is None:
            return
        latest = None
        while True:
            try:
                latest = self._hearing_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if latest is None:
            return
        # Music FEELS its way into the mood, even when Wallie stays quiet: a fresh
        # musical window nudges arousal/valence toward the track's emotional pole.
        if getattr(latest, "is_music", False):
            self._mood.on_music(latest.music_valence, latest.music_arousal,
                                latest.music_energy)
        heard = ""
        if latest.sound_type == "speech" and latest.transcript:
            if self._is_self_echo(latest.transcript):
                return  # our own voice came back through the loopback — ignore
            heard = latest.transcript
            # If the speech rode on top of music, tag the vibe so Wallie knows
            # it's a *song* (lyrics + mood), not just someone talking.
            if latest.descriptor:
                heard = f'{latest.transcript} [{latest.descriptor} music]'
        elif latest.sound_type == "music":
            heard = f"({latest.descriptor or 'music'} music playing)"
        elif latest.sound_type == "sound":
            heard = f"({latest.descriptor or 'a sound'})"
        if heard:
            self._latest_heard = heard
            self._latest_heard_ts = time.time()

    def _fresh_heard(self) -> str:
        """The recent heard transcript, only if still timely; else ''."""
        if not self._latest_heard:
            return ""
        age = time.time() - self._latest_heard_ts
        if age > self._cfg.hearing.max_context_age_sec:
            return ""
        return self._latest_heard

    @staticmethod
    def _norm_for_echo(text: str) -> str:
        return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())

    def _note_spoken(self, text: str) -> None:
        """Remember a line Wallie just said, so the ear can reject hearing it back."""
        t = self._norm_for_echo(text)
        if len(t) >= 4:
            self._recent_spoken.append((time.time(), t))

    def _is_self_echo(self, heard: str) -> bool:
        """True if `heard` closely matches something Wallie said in the last ~20s.

        Content-based (not timing) so it catches self-echo even when the loopback
        capture lags several seconds behind Wallie's actual speech.
        """
        h = self._norm_for_echo(heard)
        if len(h) < 4:
            return False
        h_tokens = set(h.split())
        now = time.time()
        for ts, said in self._recent_spoken:
            if now - ts > 20.0:
                continue
            if h in said or said in h:
                return True
            if SequenceMatcher(None, h, said).ratio() >= 0.6:
                return True
            # Heavy token overlap (heard is mostly a subset of a recent line).
            if h_tokens:
                said_tokens = set(said.split())
                overlap = len(h_tokens & said_tokens) / len(h_tokens)
                if overlap >= 0.7:
                    return True
        return False

    def _pop_latest_vision(self) -> Optional["VisionEvent"]:
        """Drain the vision queue. Returns the most recent change event (if any)
        and updates the always-on `_latest_frame` cache so monologue turns can
        still see the screen even when there's no change to react to.

        Also updates SceneMemory so the AI knows whether this is a brand-new scene
        or a small delta within the same scene.
        """
        if self._vision_queue is None:
            return None
        latest: Optional["VisionEvent"] = None
        while True:
            try:
                latest = self._vision_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if latest is not None:
            self._latest_frame = latest
            self._latest_frame_ts = time.time()
            # Update scene memory based on change type.
            if latest.change_type == ChangeType.SCENE_CHANGE:
                self._scene_memory.record_scene_change(None)  # hash not accessible here
            elif latest.change_type == ChangeType.DELTA:
                self._scene_memory.record_delta()
            # Feed activity info into behavior tracker for organic adaptation.
            scroll_dir = ""
            if latest.activity_detail is not None:
                scroll_dir = latest.activity_detail.scroll_direction
            self._behavior.record_activity(
                activity=latest.activity,
                pattern=latest.user_pattern,
                scroll_direction=scroll_dir,
            )
        return latest

    # ----- segment execution -----
    async def _ensure_fresh_screen_frame(self) -> None:
        """Make sure ``_latest_frame`` is no older than ``max_frame_age_sec`` before a
        monologue or vision segment runs. If it's stale, grab a fresh one synchronously.
        This guarantees that vision-mode segments always have something visual to react to
        even when the screen is static."""
        if not self._cfg.vision.enabled or self._vision_loop is None:
            return
        max_age = self._cfg.vision.max_frame_age_sec
        age = time.time() - self._latest_frame_ts if self._latest_frame_ts else 9999.0
        if self._latest_frame is not None and age <= max_age:
            return
        evt = await self._vision_loop.grab_now()
        if evt is not None:
            self._latest_frame = evt
            self._latest_frame_ts = time.time()
            logger.info(f"vision: fresh on-demand frame grabbed ({len(evt.frame.jpeg)} bytes)")

    async def _run_segment(self, intent: Intent) -> None:
        """Run one segment as a producer/consumer pipeline.

        Producer: pulls LLM tokens, splits into sentences, applies dedupe + scrub,
        and queues final spoken sentences.
        Consumer: plays the first sentence chunk-by-chunk for low TTFA, then
        pre-fires the next sentence's TTS while the previous one is still on
        the player. This eliminates the inter-sentence silence gaps that the
        previous serial implementation produced.
        """
        if intent.vision_directive and intent.vision_directive.reaction == VisionReaction.SILENCE:
            remaining = self._target_silence_sec - (time.time() - self._last_segment_spoken_ts)
            pause = min(random.uniform(1.0, 2.0), max(0.5, remaining))
            await asyncio.sleep(pause)
            return

        if intent.kind == "vision":
            await self._ensure_fresh_screen_frame()

        user_msg, images, source_tag = await self._build_user_turn(intent)
        if user_msg:
            self._conv.add_user(user_msg, source=source_tag, images=images)
        if intent.kind == "chat" and intent.chat is not None:
            self._last_chat_reply_ts = time.time()

        system_prompt = self._persona.system_prompt(
            topic=self._current_topic if intent.kind == "monologue" else None,
            vision_enabled=self._cfg.vision.enabled,
            session_notes=self._conv.session_notes or None,
            persistent_notes=self._memory.summary_for_prompt() if self._memory else None,
            topic_drift_style=self._cfg.topics.drift_style,
            allow_vision_skip=self._cfg.llm.allow_vision_skip,
            current_goal=self._current_session_goal(),
            enable_callbacks=self._cfg.persona.enable_callbacks,
        )
        provider_msgs = self._conv.to_provider_messages(
            system_prompt,
            max_history_messages=self._cfg.orchestrator.llm_history_messages,
        )

        max_tok = self._cfg.llm.max_tokens
        if intent.kind == "outro":
            max_tok = min(max_tok, 220)
        elif intent.kind == "vision" and intent.vision_directive:
            _vision_tok_caps = {
                # A vision model needs room to look before it answers, and 200 tokens
                # was cutting her off mid-sentence on the deep reactions.
                # These are ceilings, not targets: Gemini stops when it's done, but the
                # OpenRouter fallback is a reasoning model that burns 400-600 tokens
                # thinking before the first word, and anything under ~1000 came back
                # empty (measured 0/3 at 500, 2/2 at 1000).
                VisionReaction.GLANCE: 1000,
                VisionReaction.DEEP: 1200,
                VisionReaction.TANGENT: 1000,
            }
            max_tok = min(max_tok, _vision_tok_caps.get(
                intent.vision_directive.reaction, 100))
        elif intent.kind == "hearing":
            max_tok = min(max_tok, 140)
        allow_repeat = intent.kind == "outro"

        streamer = SentenceStreamer()
        spoken_parts: list[str] = []
        sentence_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=8)

        skip_eligible = (
            intent.kind == "vision"
            and self._cfg.llm.allow_vision_skip
            and self._consecutive_skips < 3
        )
        skipped = False

        vision_sentence_cap = 0
        if intent.kind == "vision" and intent.vision_directive:
            energy = (self._mood.arousal + self._mood.talkativity) / 2.0
            if intent.vision_directive.reaction == VisionReaction.GLANCE:
                if energy > 0.65:
                    vision_sentence_cap = random.choice([2, 3, 3, 4])
                elif energy > 0.4:
                    vision_sentence_cap = random.choice([2, 2, 3, 3])
                else:
                    vision_sentence_cap = random.choice([1, 2, 2])
            elif intent.vision_directive.reaction == VisionReaction.TANGENT:
                vision_sentence_cap = random.choice([3, 3, 4, 5])
            else:  # DEEP
                if energy > 0.6:
                    vision_sentence_cap = random.choice([4, 5, 5, 6])
                else:
                    vision_sentence_cap = random.choice([3, 4, 4, 5])
        produced_sentence_count = 0

        async def producer() -> None:
            nonlocal skipped, produced_sentence_count
            first_seen = False
            capped = False
            try:
                async for token in self._llm_for(provider_msgs).stream(
                    provider_msgs,
                    temperature=self._cfg.llm.temperature,
                    top_p=self._cfg.llm.top_p,
                    max_tokens=max_tok,
                    presence_penalty=self._cfg.llm.presence_penalty,
                    frequency_penalty=self._cfg.llm.frequency_penalty,
                ):
                    if skipped or capped:
                        continue
                    for sent in streamer.feed(token):
                        if not first_seen and _is_skip_signal(sent):
                            if skip_eligible:
                                skipped = True
                                logger.info("vision: SKIP — frame too boring, no audio for this turn")
                                break
                            else:
                                # Model tried to SKIP but not allowed — force react, drop the token.
                                logger.info("vision: SKIP blocked (consecutive=%d), forcing reaction", self._consecutive_skips)
                                continue
                        first_seen = True
                        pieces = self._prepare_sentence(sent, allow_repeat=allow_repeat)
                        for piece in pieces:
                            await sentence_q.put(piece)
                        if pieces:
                            produced_sentence_count += 1
                        if vision_sentence_cap > 0 and produced_sentence_count >= vision_sentence_cap:
                            capped = True
                            logger.info(f"vision: sentence cap reached ({produced_sentence_count}/{vision_sentence_cap})")
                            break
                    if skipped or capped:
                        continue
                if not skipped and not capped:
                    for sent in streamer.flush():
                        if not first_seen and _is_skip_signal(sent):
                            if skip_eligible:
                                skipped = True
                                logger.info("vision: SKIP — frame too boring, no audio for this turn")
                                break
                            else:
                                logger.info("vision: SKIP blocked on flush (consecutive=%d)", self._consecutive_skips)
                                continue
                        first_seen = True
                        pieces = self._prepare_sentence(sent, allow_repeat=allow_repeat)
                        for piece in pieces:
                            await sentence_q.put(piece)
                        if pieces:
                            produced_sentence_count += 1
                        if vision_sentence_cap > 0 and produced_sentence_count >= vision_sentence_cap:
                            logger.info(f"vision: sentence cap reached on flush ({produced_sentence_count}/{vision_sentence_cap})")
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"segment: generation error: {e}")
                # Backoff on errors (rate limit, safety, etc.) to prevent spam.
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.info("segment: rate-limited, backing off 30s")
                    await asyncio.sleep(30.0)
                elif "safety" in err_str or "finish_reason" in err_str:
                    await asyncio.sleep(2.0)
                else:
                    await asyncio.sleep(5.0)
            finally:
                await sentence_q.put(None)

        async def consumer() -> None:
            in_flight: list[asyncio.Task[bytes]] = []  # Pre-fired buffered TTS.
            first = True
            while True:
                sent = await sentence_q.get()
                if sent is None:
                    break
                if first:
                    first = False
                    await self._stream_sentence_direct(sent, spoken_parts)
                else:
                    if in_flight:
                        audio = await in_flight.pop(0)
                        await self._play_buffered(spoken_parts[-1] if spoken_parts else sent, audio)
                    in_flight.append(asyncio.create_task(self._buffer_tts(sent)))
                    spoken_parts.append(sent)
                    self._note_spoken(sent)
                    logger.info(f"say> {sent}")
            for task in in_flight:
                try:
                    audio = await task
                    await self._play_buffered("", audio)
                except Exception as e:
                    logger.warning(f"tts buffered task failed: {e}")

        try:
            await asyncio.gather(producer(), consumer())
        except asyncio.CancelledError:
            raise

        full = " ".join(spoken_parts).strip()
        if intent.kind == "vision" and not full:
            # SKIP or empty output: update vision timestamp to prevent rapid re-firing.
            self._last_vision_turn_ts = time.time()
            self._consecutive_skips += 1
            logger.debug(f"vision: no output (skipped={skipped}), consecutive skips = {self._consecutive_skips}")
        elif full:
            if intent.kind == "vision":
                self._consecutive_skips = 0  # Reset on successful speech.
            self._conv.add_assistant(full, source=intent.kind)
            self._last_spoken = full
            self._last_intent_kind = intent.kind
            # Auto-highlight: flag this segment if it's clip-worthy (opt-in).
            _react = intent.kind
            if getattr(intent, "vision_directive", None) is not None:
                try:
                    _react = intent.vision_directive.reaction.value
                except Exception:
                    pass
            self._highlights.note(
                text=full, arousal=getattr(self._mood, "arousal", 0.0), reaction=_react
            )
            if intent.kind != "vision":
                self._last_monologue_spoken = full
            # Update continuity trackers AFTER each spoken segment.
            self._update_open_threads(full, intent.kind)
            self._update_recent_themes(full)
            self._record_phrase_uses(full)
            self._record_question_ending(full)
            is_fallback_vision = (
                intent.kind == "vision"
                and intent.vision_directive is not None
                and intent.vision_directive.is_fallback
            )
            if intent.kind == "vision":
                self._scene_memory.record_spoken(full)
                self._last_vision_turn_ts = time.time()
            if intent.kind == "chat" and intent.chat is not None and self._memory:
                self._memory.log_viewer(
                    username=intent.chat.username,
                    platform=intent.chat.platform,
                    text=intent.chat.text,
                )
                # Si es el dueño, su frase también es contexto del stream: queda
                # guardada para que los próximos turnos conserven la coherencia.
                if (intent.chat.username or "").lower().lstrip("@") in self._usuarios_favoritos():
                    del self._nota_dueno[:-12]      # solo los 12 ultimos
                    # Lo que dijo CASAVE (intent.chat.text), no `full`, que es
                    # lo que Casavita esta diciendo: el prompt del check-in lo
                    # lee como "Lo ultimo que dijo casave fue: ...".
                    self._nota_dueno.append(intent.chat.text)
            self._mood.on_segment_spoken(length_chars=len(full))
            if is_fallback_vision:
                self._attention.on_segment_spoken(intent_kind="monologue")
            else:
                self._attention.on_segment_spoken(intent_kind=intent.kind)
            self._last_segment_spoken_ts = time.time()
            self._target_silence_sec = self._compute_next_silence_target()

            # Eco en el chat de Twitch: una vez por turno, al terminar, con el
            # texto entero que acaba de decir en voz alta (a veces no se oye bien).
            # Solo cuando esta contestando a alguien: si lo suelta en monologo
            # visionado seria ruido en el canal.
            if full and self._chat and intent.kind in ("chat", "hearing"):
                # Para el chat solo va la respuesta: el "fulano dice que..." que
                # dice por voz alta no lo quiere ahi, solo el aviso por si no se oye.
                # Se manda a la MISMA plataforma de la que le hablaron: si le
                # escriben en Kick, responde en Kick, no se cuela en Twitch.
                _plataforma = intent.chat.platform if intent.kind == "chat" and intent.chat else ""
                _quien = intent.chat.username if (intent.kind == "chat" and intent.chat) else ""
                await self._eco_al_chat(full, _plataforma, _quien)

        if intent.kind == "monologue":
            if self._cfg.topics.mode == "list" and random.random() < self._cfg.topics.switch_chance:
                if self._conv.session_seconds() >= self._cfg.topics.switch_min_sec:
                    self._current_topic = self._pick_next_topic()

    async def _eco_al_chat(self, text: str, plataforma: str, username: str = "") -> None:
        """Escribe en el chat lo que Casavita acaba de decir en voz alta.

        Va a la MISMA plataforma de la que le hablaron: si alguien escribe en
        Kick, la respuesta aparece en Kick y no se cuela en Twitch.

        Se le quita el parafraseo inicial ("fulano dice que..."): en voz alta sirve
        para que se entienda de quien habla la cosa, pero escrito en el chat es ruido.

        Twitch corta los mensajes a 500 caracteres, y mandar medio dozen de lineas
        seguidas dispara el rate limit de spam, asi que hay que podar y espaciar.
        """
        destino = (plataforma or "twitch").lower()
        msg = _quitar_parafraseo(" ".join(text.split()), username)
        if not msg:
            return
        if len(msg) > 480:
            msg = msg[:477].rstrip() + "…"

        now = time.monotonic()
        if now - self._ultimo_eco_chat < 3.0:
            logger.debug(f"{destino}: eco saltado, demasiado seguido")
            return

        for m in getattr(self._chat, "_monitors", []):
            if getattr(m, "platform", "") != destino:
                continue
            # Sin permiso de escritura (p.ej. el monitor de Kick solo lee):
            # se avisa en vez de fallar en silencio cada turno.
            if not hasattr(m, "send_message"):
                logger.warning(f"{destino}: este monitor no puede escribir (solo lectura)")
                return
            try:
                await m.send_message(msg)
            except Exception as e:
                logger.warning(f"{destino}: no se pudo escribir en el chat: {e}")
                return
            # El reloj solo avanza si el mensaje se fue de verdad: si falla, no
            # queremos que el reintento se coma el hueco del rate limit.
            self._ultimo_eco_chat = time.monotonic()
            return
        logger.debug(f"{destino}: sin monitor activo, no se escribe")

    # ----- sentence preparation -----
    def _prepare_sentence(self, raw: str, *, allow_repeat: bool) -> list[str]:
        """Run dedupe + scrub + long-sentence-splitting. May yield 0..N pieces."""
        if not raw or not raw.strip():
            return []
        if not allow_repeat and self._conv.is_repeat(
            raw,
            window=self._cfg.orchestrator.dedupe_window,
            threshold=self._cfg.orchestrator.dedupe_threshold,
        ):
            logger.debug(f"dedupe: skipped near-duplicate: {raw[:60]}")
            return []
        scrubbed = _scrub_unspeakable(raw)
        if not scrubbed:
            return []
        return _split_run_on(scrubbed, max_words=self._cfg.orchestrator.max_words_per_sentence)

    # ----- speech execution -----
    async def _stream_sentence_direct(self, sentence: str, spoken_parts: list[str]) -> None:
        """First sentence of a segment: stream chunk-by-chunk for the lowest TTFA."""
        spoken_parts.append(sentence)
        self._note_spoken(sentence)
        logger.info(f"say> {sentence}")
        await self._avatar_safe_call("trigger_emotion_from_text", sentence)
        await self._avatar_safe_call("set_speaking", True)
        first_chunk = True
        try:
            async for pcm in self._tts.synthesize(sentence):
                if first_chunk:
                    first_chunk = False
                    if _looks_non_pcm(pcm):
                        logger.error(
                            f"tts: non-PCM data detected (header={pcm[:6]!r}); aborting sentence"
                        )
                        break
                await self._player.write(pcm)
                if self._avatar:
                    await self._avatar_safe_call("feed_audio", pcm, self._tts.sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"tts (direct) failed: {e}")
        finally:
            self._player.boundary()
            await self._avatar_safe_call("set_speaking", False)

    async def _buffer_tts(self, sentence: str) -> bytes:
        """Used by the consumer to pre-fire a sentence's TTS in parallel."""
        await self._avatar_safe_call("trigger_emotion_from_text", sentence)
        chunks: list[bytes] = []
        first_chunk = True
        try:
            async for pcm in self._tts.synthesize(sentence):
                if first_chunk:
                    first_chunk = False
                    if _looks_non_pcm(pcm):
                        logger.error(
                            f"tts: non-PCM data detected (header={pcm[:6]!r}); aborting sentence"
                        )
                        break
                chunks.append(pcm)
        except Exception as e:
            logger.warning(f"tts (buffered) failed: {e}")
        audio = b"".join(chunks)
        if len(audio) & 1:
            audio = audio[:-1]
        return audio

    async def _play_buffered(self, sentence_for_log: str, audio: bytes) -> None:
        if not audio:
            return
        await self._avatar_safe_call("set_speaking", True)
        try:
            await self._player.write(audio)
            if self._avatar:
                await self._avatar_safe_call("feed_audio", audio, self._tts.sample_rate)
        finally:
            self._player.boundary()
            await self._avatar_safe_call("set_speaking", False)

    # ----- avatar safety wrapper -----
    async def _avatar_safe_call(self, method: str, *args: Any) -> None:
        """Avatar errors must never abort audio. Swallow everything.

        Two special-case methods:
          * ``"trigger_emotion_from_text"`` — keyword classifier on a sentence.
          * ``"trigger_emotion"`` — direct slot trigger (e.g. "hype", "thinking").

        Everything else is a passthrough to the avatar instance.
        """
        if self._avatar is None:
            return
        try:
            if method == "trigger_emotion_from_text":
                await _trigger_emotion_from_text(args[0], self._avatar, self._cfg.avatar)
                return
            fn = getattr(self._avatar, method, None)
            if fn is not None:
                await fn(*args)
        except Exception as e:
            logger.debug(f"avatar.{method} failed: {e}")

    # ----- image helpers -----
    def _llm_for(self, messages: list[dict[str, Any]]) -> "LLMProvider":
        """Pick the provider that can actually handle this turn.

        The chat model is tuned for fast text; most of them reject a screenshot
        outright ("content must be a string"). So if any message carries an image
        block and a vision provider was configured, this turn goes there.
        """
        if self._vision_llm is None:
            return self._llm
        for m in messages:
            content = m.get("content")
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "image" for b in content
            ):
                return self._vision_llm
        return self._llm

    async def _downscale_for_llm(self, jpeg_bytes: bytes) -> bytes:
        """Shrink frame for LLM to reduce upload size and processing time.

        JPEG decode + resize + re-encode is pure-CPU work that would otherwise
        block the event loop (stalling audio playback and the vision loop for the
        duration). Offload it to the default thread-pool executor so the pipeline
        keeps breathing while a frame is being prepared.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: downscale_jpeg(
                jpeg_bytes,
                max_edge=self._cfg.vision.llm_max_edge_px,
                quality=self._cfg.vision.llm_jpeg_quality,
            ),
        )

    # ----- turn assembly -----
    async def _build_user_turn(self, intent: Intent) -> tuple[str, list[ImageBlock], str]:
        if intent.kind == "chat" and intent.chat is not None:
            m = intent.chat
            key = (m.username or "").strip().lstrip("@").lower()
            es_dueno = key in self._usuarios_favoritos()
            # La bienvenida se recuerda por plataforma: un "chui54" de TikTok y
            # otro de YouTube son personas distintas. Sobrevive a los reinicios.
            primera_vez = bool(key) and key not in self._chat_vistos
            if self._memory is not None:
                primera_vez = not self._memory.saludado_ya(m.platform, m.username)
            if primera_vez:
                if self._memory is not None:
                    self._memory.marcar_saludado(m.platform, m.username)
                if key:
                    self._chat_vistos.add(key)
                logger.info(
                    f"chat: {m.username} en {m.platform}, primera vez (dueno={es_dueno})"
                )
            return (
                self._persona.chat_turn(
                    username=m.username,
                    platform=m.platform,
                    text=m.text,
                    is_highlight=m.is_highlight,
                    is_first_time=primera_vez,
                    is_owner=es_dueno,
                    chat_banter=self._cfg.chat.banter,
                ),
                [],
                f"chat:{m.platform}:{m.username}",
            )
        if intent.kind == "vision" and intent.vision is not None:
            play = getattr(self._cfg, "play", None)
            play_note = _live_activity() if (play is not None and play.enabled and play.talk_from_agent) else ""
            directive = intent.vision_directive
            if directive is None:
                directive = VisionDirective(reaction=VisionReaction.DEEP, target_sentences=3)
            if directive.reaction == VisionReaction.GLANCE:
                mode = "glance"
            elif directive.reaction == VisionReaction.TANGENT:
                mode = "tangent"
            else:
                mode = intent.vision.change_type.value  # "scene" | "delta"
            has_img = bool(intent.vision.frame.jpeg)
            return (
                self._persona.vision_turn(
                    change_type=mode,
                    last_description=self._scene_memory.recent_context or self._scene_memory.last_description,
                    current_topic=self._current_topic,
                    scene_age_sec=self._scene_memory.scene_age_sec(),
                    target_sentences=directive.target_sentences,
                    glance_style=directive.glance_style,
                    tangent_seed=directive.tangent_seed,
                    mood_label=self._mood.label,
                    adaptation_hint=self._behavior.adaptation_hint(),
                    screen_activity=intent.vision.activity.value,
                    allow_vision_skip=self._cfg.llm.allow_vision_skip,
                    heard=self._fresh_heard(),
                    activity_note=play_note,
                    play_has_screen=bool(play_note) and has_img,
                ),
                ([ImageBlock(data=await self._downscale_for_llm(intent.vision.frame.jpeg), mime="image/jpeg")]
                 if has_img else []),
                "vision",
            )
        if intent.kind == "hearing":
            energy = (self._mood.arousal + self._mood.talkativity) / 2.0
            target = random.choice([1, 2, 2]) if energy < 0.6 else random.choice([2, 2, 3])
            return (
                self._persona.hearing_turn(
                    heard=intent.heard or self._fresh_heard(),
                    mood_label=self._mood.label,
                    target_sentences=target,
                ),
                [],
                "hearing",
            )

        if intent.kind == "outro":
            mins = (time.time() - self._session_start_ts) / 60.0 if self._session_start_ts else 0.0
            return (self._persona.outro_turn(minutes_streamed=mins), [], "outro")

        
        images: list[ImageBlock] = []
        screen_attached = False
        if self._enrich_this_turn:
            self._enrich_this_turn = False
            logger.debug("vision: enrich_monologue — flag reset, image intentionally not attached")

        forbidden_phrases = self._phrases_to_forbid()
        suppress_question = self._segments_ended_with_question >= 1

        oc = self._cfg.orchestrator
        continuity_segment = self._last_monologue_spoken or None
        return (
            self._persona.monologue_turn(
                topic=self._current_topic,
                last_segment=continuity_segment,
                open_threads=list(self._open_threads) or None,
                recent_themes=list(self._recent_themes) or None,
                forbidden_phrases=forbidden_phrases or None,
                suppress_question=suppress_question,
                screen_attached=screen_attached,
                enrich_last_description=self._scene_memory.last_description if screen_attached else "",
                adaptation_hint=self._behavior.adaptation_hint() if screen_attached else "",
                sentences_min=oc.segment_sentences_min,
                sentences_max=oc.segment_sentences_max,
                topic_drift_style=self._cfg.topics.drift_style,
                after_vision=self._last_intent_kind == "vision",
                heard=self._fresh_heard(),
                check_in=self._intento_es_checkin(intent),
                owner_recent=list(self._nota_dueno) or None,
                chat_banter=self._cfg.chat.banter,
            ),
            images,
            "monologue",
        )

    def _intento_es_checkin(self, intent: Intent) -> bool:
        """True si este turno es una pregunta a casave, no monologueo."""
        return (
            not self._cfg.vision.spontaneous_monologue
            and self._cfg.chat.check_in_probability > 0.0
            and intent.vision_directive is not None
            and intent.vision_directive.reaction == VisionReaction.TANGENT
            and intent.vision_directive.target_sentences > 0
        )

    def _pick_next_topic(self) -> Optional[str]:
        if self._cfg.topics.mode == "ai_picks":
            return None
        chosen = pick_topic(self._cfg.topics.topics, self._recent_topics)
        if chosen:
            self._recent_topics.append(chosen)
            if len(self._recent_topics) > 5:
                self._recent_topics.pop(0)
        return chosen

    # ----- avatar event cues -----
    async def _cue_intent_expression(self, intent: Intent) -> None:
        """Show an avatar reaction the moment we PICK an intent — before any
        TTS audio is produced. This gives the viewer instant feedback that the
        streamer is reacting (super chat → hype face, vision change → looking)
        instead of staring blankly while the LLM warms up.
        """
        if self._avatar is None:
            return
        try:
            if intent.kind == "chat" and intent.chat is not None:
                if intent.chat.is_highlight:
                    await self._avatar_safe_call("trigger_emotion", "hype")
                # Glance at "the chat" — slight head turn left.
                await self._avatar_safe_call("look_at", -10.0, 0.0)
            elif intent.kind == "vision":
                # Look up at the "screen" briefly + a small surprised tick.
                await self._avatar_safe_call("trigger_emotion", "surprised")
                await self._avatar_safe_call("look_at", 0.0, -8.0)
            elif intent.kind == "monologue":
                # Brief thinking face right before speaking starts.
                await self._avatar_safe_call("trigger_emotion", "thinking")
        except Exception as e:
            logger.debug(f"avatar cue failed: {e}")

    # ----- avatar mood sync -----
    async def _sync_mood_to_avatar(self) -> None:
        """Push current mood state to the avatar for reactive idle behaviour."""
        if self._avatar is None:
            return
        try:
            await self._avatar.update_mood(
                self._mood.arousal,
                self._mood.valence,
                self._mood.focus,
            )
        except Exception as e:
            logger.debug(f"avatar mood sync failed: {e}")

    # ----- continuity trackers -----
    def _phrases_to_forbid(self) -> list[str]:
        """Catchphrases + running gags used within the last 5 segments. The
        next monologue turn passes these as 'do not say' to break the loop
        where the model latches onto one signature line and uses it every
        segment."""
        if not self._phrase_uses:
            return []
        cutoff = self._conv.total_segments - 5
        recent = [phrase for phrase, idx in self._phrase_uses.items() if idx > cutoff]
        return recent

    def _record_phrase_uses(self, spoken: str) -> None:
        """Scan the segment for any catchphrase or gag and stamp it with the
        current segment index. Comparison is case-insensitive substring."""
        low = spoken.lower()
        persona = self._cfg.persona
        all_phrases = list(persona.catchphrases) + list(persona.running_gags)
        for phrase in all_phrases:
            stub = phrase.lower().strip()
            if not stub:
                continue
            stub_key = " ".join(stub.split()[:4])
            if stub_key and stub_key in low:
                self._phrase_uses[phrase] = self._conv.total_segments

    def _record_question_ending(self, spoken: str) -> None:
        """Increment if the segment ended on a question mark in its last
        sentence; reset otherwise. Used to throttle the question-loop pattern."""
        tail = spoken.strip()[-80:].strip()
        if "?" in tail:
            self._segments_ended_with_question += 1
        else:
            self._segments_ended_with_question = 0

    def _current_session_goal(self) -> Optional[str]:
        """The character's current through-line goal, rotating into chapters over time."""
        goals = self._cfg.persona.session_goals
        if not goals:
            return None
        now = time.time()
        rotate = self._cfg.persona.goal_rotate_sec
        if self._current_goal is None:
            self._goal_index = random.randrange(len(goals))
            self._current_goal = goals[self._goal_index]
            self._goal_started_ts = now
            logger.info(f"goal: this session -> {self._current_goal}")
        elif rotate > 0 and (now - self._goal_started_ts) >= rotate and len(goals) > 1:
            self._goal_index = (self._goal_index + 1) % len(goals)
            self._current_goal = goals[self._goal_index]
            self._goal_started_ts = now
            logger.info(f"goal: advanced -> {self._current_goal}")
        return self._current_goal

    def _update_open_threads(self, spoken: str, intent_kind: str) -> None:
        """Detect questions / teases at the end of the last segment so the next
        one can be told to pay them off."""
        if intent_kind != "monologue" and self._open_threads:
            self._open_threads = self._open_threads[-2:]

        tail = spoken.strip()[-220:].strip()
        opened: list[str] = []
        last_sentence = re.split(r"[\.!?]\s+", tail)
        last_sentence = last_sentence[-1] if last_sentence else tail
        if "?" in tail[-80:]:
            q_match = re.findall(r"([^\.!?]*\?)", tail)
            if q_match:
                opened.append(q_match[-1].strip())
        # Tease patterns.
        TEASE = (
            "wait until you hear", "let me tell you", "here's the kicker",
            "here is the kicker", "the funny part is", "you'll see why",
            "i'll tell you", "i will tell you", "coming back to this",
            "we'll come back", "we will come back", "spoiler",
            "watch this", "trust me", "it gets better", "it gets worse",
        )
        low = last_sentence.lower()
        if any(t in low for t in TEASE):
            opened.append(last_sentence.strip())
        for o in opened:
            if o and o not in self._open_threads:
                self._open_threads.append(o)
        # Cap.
        if len(self._open_threads) > 4:
            self._open_threads = self._open_threads[-4:]

        if intent_kind == "monologue" and not opened and self._open_threads:
            self._open_threads.pop(0)

    def _update_recent_themes(self, spoken: str) -> None:
        """Tag this segment with a short theme line so the next segment knows
        what's already been covered."""
        first = re.split(r"[\.!?]\s+", spoken.strip())[0]
        words = first.split()
        if len(words) > 12:
            first = " ".join(words[:12]) + "…"
        self._recent_themes.append(first.strip())
        if len(self._recent_themes) > 8:
            self._recent_themes.pop(0)

    # ----- rolling summarizer -----
    def _maybe_kick_summarizer(self) -> None:
        self._segments_since_summary += 1
        if self._segments_since_summary < self._cfg.orchestrator.summarize_every_n:
            return
        if self._summarizer_task and not self._summarizer_task.done():
            return  # one in flight is enough
        eligible = self._conv.messages_eligible_for_summary()
        if not eligible:
            self._segments_since_summary = 0
            return
        transcript = "\n".join(
            f"[{m.source}] {m.content}" for m in eligible if m.role == "assistant"
        )
        prior = self._conv.session_notes
        self._segments_since_summary = 0
        logger.info(
            f"summary: kicking (folding {len(eligible)} old turns into running notes)"
        )
        self._summarizer_task = asyncio.create_task(
            self._run_summarizer(transcript, prior),
            name="summarizer",
        )

    async def _run_summarizer(self, transcript: str, prior_notes: str) -> None:
        try:
            prompt = self._persona.summarizer_prompt(transcript=transcript, prior_notes=prior_notes)
            tokens: list[str] = []
            async for tok in self._llm.stream(
                [
                    {"role": "system", "content": "You compress streamer transcripts into tight bullet notes. Match the language of the transcript."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=400,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            ):
                tokens.append(tok)
            new_notes = "".join(tokens).strip()
            if new_notes:
                self._conv.compact_history(new_notes)
                if self._memory:
                    self._memory.update_notes(new_notes)
                preview = new_notes[:160].replace("\n", " ⤷ ")
                logger.info(
                    f"summary: notes now {len(new_notes)} chars, "
                    f"history compacted to {len(self._conv.messages())} msgs. "
                    f"preview> {preview}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"summary: failed: {e}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_STAGE_DIR = re.compile(
    r"\*[^*\n]{1,60}\*"
    r"|\[[^\]\n]{1,60}\]"
)
_BOLD = re.compile(r"\*\*([^*\n]{1,80})\*\*")
# Leftover formatting characters that shouldn't be voiced.
_MD_CHARS = re.compile(r"[\*_`#>]")
# Unicode that TTS engines mangle: emoji (and their ZWJ/variation-selector glue),
# dingbats, arrows, box drawing and exotic spaces. Chat models emit these constantly
# and Piper literally tries to pronounce them, which bloats a 2.5s line into 5.9s of
# gibberish. Spanish punctuation (¡ ¿ ° …) and accented letters are preserved.
_UNSPEAKABLE = re.compile(
    "["
    "ᄀ-ᅟ"                       # Hangul Jamo (LLaVA-style placeholder tokens)
    "←-⇿"                  # arrows
    "⌀-⏿"                  # misc technical + control pictures
    "①-⓿"                  # circled alphanumerics
    "■-◿"                  # geometric shapes
    "☀-➿"                  # misc symbols + dingbats
    "⬀-⯿"                  # misc symbols and arrows
    "──-╿"                  # box drawing
    "‌‍"                 # zero-width space / ZWNJ / ZWJ
    "⁠-⁯"                  # other zero-width + word-joiner glue
    "︎️"                     # variation selectors
    "﻿"                               # BOM
    "­"                               # soft hyphen
    "\U0001f000-\U0001faff"    # emoji, pictographs, flags
    "   -   　"    # exotic spaces
    "]"
)
_PUNCT_GAP = re.compile(r"\s+([,.;:!?…])")

# Saludo escrito en el chat: dispara el gesto de la mano. Solo palabras
# sueltas, para no saludar por cualquier mensaje que las contenga.
_SALUDO_PALABRAS = (
    r"hola|holis|holaa|ola|buenas|buenos dias|buenas tardes|buenas noches"
    r"|hey|hi|hello|casavita|todas|todos|amigos|everyone"
    r"|saluda|saludo|saludos|adios|chao|chula|guapa|que tal|que onda|waves?|a"
)
_SALUDO_CHAT = re.compile(rf"^(?:{_SALUDO_PALABRAS})(?:\s+(?:{_SALUDO_PALABRAS}))*$", re.I)

_EMOTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Laughter — strongest, fire first.
    (re.compile(r"\b(laugh|haha+|hehe+|lmao+|rofl|giggle|chuckle|cackle|kjjk|ahaha|yha|yhaha)\b", re.I), "laughing"),
    # Hype: shouted excitement, donations, big wins.
    (re.compile(r"\b(hype|let's go|lets go|let's gooo|insane|unreal|huge|massive|on fire|absolute|crazy good|hell yes|hell yeah)\b", re.I), "hype"),
    # Surprise / shock.
    (re.compile(r"\b(wow|whoa+|oh no|omg|what the|holy|no way|surprised|wait what|are you serious|seriously\?|excuse me)\b", re.I), "surprised"),
    # Anger / frustration.
    (re.compile(r"\b(furious|pissed|annoyed|infuriating|hate this|so bad|disaster|trash|garbage|broken|terrible|awful)\b", re.I), "angry"),
    # Sadness / disappointment.
    (re.compile(r"\b(sad|depressing|tragic|breaks my heart|so bleak|gut-wrenching|devastat|sucks)\b", re.I), "sad"),
    # Eyeroll / smug — read sarcasm and dismissal.
    (re.compile(r"\b(of course|naturally|obviously|sure thing|cool cool|riiight|whatever|classic|imagine that)\b", re.I), "eyeroll"),
    # Confused / pondering.
    (re.compile(r"\b(confused|wait|hold on|let me|is that|how does|how did|why is|why are|why would|i don't get)\b", re.I), "confused"),
    # Smug — confident takes.
    (re.compile(r"\b(told you|called it|exactly|knew it|of course it|i was right)\b", re.I), "smug"),
    # Deadpan — flat reactions.
    (re.compile(r"\b(\.\.\.|huh|okay then|alright then|well that|moving on|anyway)\b", re.I), "deadpan"),
    # Generic happy — broad fallback BEFORE we give up.
    (re.compile(r"\b(love it|nice|awesome|great|amazing|beautiful|stunning|incredible|so good|that's the move)\b", re.I), "happy"),
]


# Verbos con los que Casavita parafrasea lo que acaba de leer en el chat
# ("fulano dice que...", "alguien agradece el animo..."). Lista amplia a
# proposito: si un verbo falta, el parafraseo se queda (molesto pero inofensivo);
# si sobra uno, se come el principio de su respuesta (peor).
_PARAFRASEO = re.compile(
    r"\b(?:dice|dijo|dice que|dijo que|dicen|diría|pregunta|preguntó|preguntan|"
    r"pregunta que|preguntó que|pregunta si|preguntó si|pregunta por|"
    r"quiere|quiso|quiere saber|quiere preguntar|quiere contar|quiere decir|"
    r"agradece|agradeció|agradecido|agradezco|saluda|saludó|se presenta|"
    r"está|esta|están|estaba|acaba de|acaba de escribir|acaba de preguntar|"
    r"acaba de decir|acaba de saludar|ha dicho|ha preguntado|ha escrito|ha mandado|"
    r"ha comentado|han dicho|menciona|comenta|comentó|cuenta|expone|indica|señala|"
    r"avisa|pide|pidió|anuncia|comparte|confiesa|aclara|felicita|felicitó|"
    r"nos dice|nos pregunta|nos cuenta)\b",
    re.I,
)
# "fulano, ¿qué tal?" / "fulano te contesto" NO son parafraseo: le estan
# hablando a el. Si el nombre va seguido de puntuacion o de un tratamiento, se
# respeta la frase intacta.
_AL_PARCELO = re.compile(r"[,;:!?¡¿]|\b(?:y|te|te|nos|va|vas|puedes|tu|tú|queremos)\b", re.I)


def _quitar_parafraseo(texto: str, username: str) -> str:
    """Quita la primera frase si solo repite lo que escribió el usuario.

    El prompt le obliga a empezar reconociendo ("fulano dice que quiere saber X"), y en
    voz alta eso ayuda, pero escrito en el chat es ruido: lo acaba de leer todo el
    mundo. Solo se toca la PRIMERA frase y solo si empieza por el nombre del que
    escribió con un verbo de parafraseo, asi que si lo que dice es su respuesta no
    se toca nada.
    """
    nombre = (username or "").strip().lstrip("@")
    if not nombre:
        return texto
    partes = re.split(r"(?<=[.!?…])\s+", texto.strip(), maxsplit=1)
    if len(partes) < 2:
        return texto  # una sola frase: mejor mandarla que dejar el chat mudo
    primera, resto = partes
    # Hasta dos palabras de relleno delante ("oye casave dice que..."). Los "_"
    # del final del nick se permiten opcionales: ella escribe "casave" y el
    # usuario de Twitch es "tu_canal_twitch".
    nombre = re.escape(nombre.rstrip("_")) + r"_*"
    m = re.match(rf"^((?:[\w'’]+\s+){{0,2}}@?{nombre}\b\s*)(\S+)?", primera, re.I)
    if not m:
        return texto
    # Lo que va justo detras del nombre decide: si es puntuacion o un
    # tratamiento, le esta hablando a el y la frase se queda entera.
    if _AL_PARCELO.match(m.group(2) or ""):
        return texto
    if not _PARAFRASEO.search(primera):
        return texto
    return resto.strip()



def _scrub_unspeakable(text: str) -> str:
    text = _BOLD.sub(r"\1", text)
    text = _STAGE_DIR.sub("", text)
    text = _MD_CHARS.sub("", text)
    text = _UNSPEAKABLE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = _PUNCT_GAP.sub(r"\1", text)
    return text.strip()


_BREAK_AFTER = {"and", "but", "which", "so", "because", "or", "though", "while", "until"}

def _split_run_on(sentence: str, max_words: int = 22) -> list[str]:
    """Safety net for run-on sentences the model spits out as one giant
    comma-spliced thought. Splits at conjunctions only when the original
    sentence would be too long to deliver naturally in one breath.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    out: list[str] = []
    cur: list[str] = []
    for i, w in enumerate(words):
        cur.append(w)
        prev_had_comma = i > 0 and words[i - 1].endswith(",")
        is_break_word = w.lower().rstrip(",.!?") in _BREAK_AFTER
        long_enough = len(cur) >= 12
        if long_enough and (
            (prev_had_comma and is_break_word) or (cur[-1].endswith(",") and len(cur) >= 18)
        ):
            piece = " ".join(cur).rstrip(", ")
            if not piece.endswith((".", "!", "?")):
                piece += "."
            out.append(piece)
            cur = []
    if cur:
        rest = " ".join(cur).strip()
        if rest:
            if not rest.endswith((".", "!", "?")):
                rest += "."
            out.append(rest)
    out = [p for p in out if p.strip()]
    return out or [sentence]


def _is_skip_signal(text: str) -> bool:
    """True if the model returned the SKIP escape-hatch instead of a real
    sentence. Tolerant of trailing punctuation, capitalization, and a small
    leading whitespace because LLMs are sloppy about exact format compliance.
    """
    if not text:
        return False
    t = text.strip().rstrip(".!?,").strip()
    if not t:
        return False
    return t.upper() in {"SKIP", "SKIP."}


def _looks_non_pcm(chunk: bytes) -> bool:
    """Heuristic: detect when a TTS provider returned MP3/WAV/text instead of
    the PCM we asked for. Wrong format played as PCM = ear-melting static.

    We err on the side of false negatives (let real PCM through) because raw
    PCM can start with literally any byte values.
    """
    if not chunk:
        return False
    head = chunk[:6]
    # WAV containers
    if head[:4] == b"RIFF" or head[:4] == b"OggS":
        return True
    # MP3 signatures: ID3 tag or sync frame.
    if head[:3] == b"ID3":
        return True
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 and (head[1] & 0x06) != 0:
        # MP3 sync word with valid layer bits — far more likely than the same
        # bytes appearing as legitimate PCM samples.
        return True
    if head[:1] in (b"{", b"<"):
        return True
    return False


def _pcm_rms(pcm: bytes) -> float:
    """Return RMS amplitude (0-1) for a 16-bit little-endian PCM chunk."""
    if not pcm or len(pcm) < 2:
        return 0.0
    try:
        import numpy as np  # type: ignore
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return min(1.0, rms / 32768.0)
    except Exception:
        n = len(pcm) // 2
        if n == 0:
            return 0.0
        total = sum(
            abs(struct.unpack_from("<h", pcm, i * 2)[0])
            for i in range(n)
        )
        return min(1.0, (total / n) / 32768.0)


async def _trigger_emotion_from_text(
    sentence: str,
    avatar: Any,
    avatar_cfg: Any,
) -> None:
    """Classify a sentence with the keyword table and fire the matching VTS slot.

    The classifier scans both the visible text AND any stage-direction markers
    (e.g. ``[laughs]``) so authors can hint at expressions explicitly without
    leaking them to TTS.
    """
    stage_matches = _STAGE_DIR.findall(sentence)
    combined = " ".join(stage_matches) + " " + sentence

    for pattern, slot in _EMOTION_PATTERNS:
        if pattern.search(combined):
            try:
                fn = getattr(avatar, "trigger_emotion", None)
                if callable(fn):
                    await fn(slot)
                else:
                    hotkey_name = getattr(avatar_cfg, f"expr_{slot}", "")
                    if hotkey_name:
                        await avatar.trigger_expression(hotkey_name)
            except Exception:
                pass
            return  # one expression per sentence
