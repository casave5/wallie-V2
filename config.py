"""Runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
STATE_FILE = BASE_DIR / ".wallie_state.json"


# -------------------------------------------------------------------
# Secrets
# -------------------------------------------------------------------
class Secrets(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    fish_api_key: str = Field(default_factory=lambda: os.getenv("FISH_API_KEY", ""))
    elevenlabs_api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))

    youtube_api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    youtube_client_secret_file: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET_FILE", "scripts/client_secret.json")
    )
    youtube_live_chat_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_LIVE_CHAT_ID", ""))

    twitch_oauth_token: str = Field(default_factory=lambda: os.getenv("TWITCH_OAUTH_TOKEN", ""))
    twitch_channel: str = Field(default_factory=lambda: os.getenv("TWITCH_CHANNEL", ""))
    twitch_nick: str = Field(default_factory=lambda: os.getenv("TWITCH_NICK", ""))

    kick_channel: str = Field(default_factory=lambda: os.getenv("KICK_CHANNEL", ""))


# -------------------------------------------------------------------
# Persona
# -------------------------------------------------------------------
Profanity = Literal["none", "mild", "heavy"]
Formality = Literal["street", "casual", "formal"]
SentenceLength = Literal["short", "medium", "mixed"]
HumorStyle = Literal[
    "ironic", "deadpan", "absurd", "observational",
    "self_deprecating", "roast", "wholesome", "chaotic",
]
Energy = Literal["chill", "warm", "hyped", "unhinged"]


class PersonaConfig(BaseModel):
    name: str = "Wallie"
    handle: str = "@wallie"
    language: str = "en"
    pronouns: str = "they/them"
    age_range: str = "early 20s"
    origin: str = "somewhere online"
    archetype: str = "variety streamer"
    backstory: str = (
        "A chronically online streamer who has seen every weird corner of the internet "
        "and is mildly amused by all of it."
    )

    energy: Energy = "warm"
    humor_style: list[HumorStyle] = Field(default_factory=lambda: ["ironic", "observational"])
    profanity: Profanity = "mild"
    formality: Formality = "casual"
    sentence_length: SentenceLength = "short"
    catchphrases: list[str] = Field(default_factory=list)
    running_gags: list[str] = Field(default_factory=list)
    banned_words: list[str] = Field(default_factory=list)
    extra_style_notes: str = ""

    strong_opinions: bool = True
    admit_uncertainty: bool = True
    break_fourth_wall: bool = False  # Can reference being on a stream
    favorite_topics: list[str] = Field(default_factory=list)
    taboo_topics: list[str] = Field(default_factory=list)

    address_style: Literal["by_name", "generic", "crowd"] = "by_name"
    reply_length: Literal["snappy", "medium", "longer"] = "snappy"
    react_to_highlights_hype: bool = True

    vision_first_person: bool = True
    vision_commentary_density: Literal["sparse", "balanced", "dense"] = "balanced"
    vision_interests: list[str] = Field(default_factory=list)
    vision_boring_signals: list[str] = Field(default_factory=lambda: [
        "google homepage", "new tab", "loading", "blank page",
        "desktop", "login screen", "settings",
    ])

    entertainer_mode: bool = True
    audience_hook_rate: float = 0.30
    anecdote_seeds: list[str] = Field(default_factory=list)
    personal_beat_rate: float = 0.35

    # --- Conversational / companion mode (e.g. VRChat) ---
    # When true, Wallie talks WITH people in a real-time back-and-forth instead of
    # hosting a show: short turns, replies to what the other person actually said,
    # no audience/monologue framing.
    conversational: bool = False
    reveal_ai: bool = False          # openly an AI; owns it, can joke about it
    plug_url: str = ""               # soft self-plug (e.g. github) when someone's curious
    plug_rate: float = 0.15          # 0 = never bring it up unprompted; higher = more readily

    # --- Feeling alive: intent + memory ---
    # An ongoing goal/objective the character pursues this session (a "through-line").
    # Gives the stream a spine instead of pure moment-to-moment reactions. Rotates so
    # the session has chapters. Empty = no explicit goal.
    session_goals: list[str] = Field(default_factory=list)
    goal_rotate_sec: float = 600.0   # advance to the next goal every N sec (0 = never rotate)
    # Encourage natural callbacks to earlier moments/takes from this session.
    enable_callbacks: bool = True
    # When conversational + vision is on: Wallie may START conversations on its own —
    # if it sees someone it could talk to, it opens with a greeting/line instead of
    # only ever waiting to be spoken to.
    initiates: bool = False


# -------------------------------------------------------------------
# Other subsystems
# -------------------------------------------------------------------
class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "openrouter", "anthropic", "gemini", "ollama"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 350
    presence_penalty: float = 0.3
    frequency_penalty: float = 0.4
    vision_capable: bool = False
    # Most text models can't see, so a turn carrying a screenshot is routed to a
    # SECOND provider instead of the chat one. Empty = use the same provider/model
    # as the text turns (works when that model happens to be multimodal).
    vision_provider: str = ""
    vision_model: str = ""
    # Tried when the vision provider above fails (usually: free quota exhausted).
    # Same shape as vision_provider/vision_model, so several can be chained.
    vision_fallback_provider: str = ""
    vision_fallback_model: str = ""
    # Reasoning budget for the fallback; "low" = answer instead of overthinking.
    vision_fallback_reasoning_effort: str = "low"
    # Reasoning models (gpt-oss on Groq) hide a reasoning pass that eats max_tokens
    # and latency. "low" keeps a live stream snappy; "" sends nothing.
    reasoning_effort: str = ""
    # "flat" sends reasoning_effort=<value> (OpenAI/Groq style); "nested" sends
    # reasoning={"effort": <value>}, which OpenRouter's reasoning models need.
    reasoning_transport: Literal["flat", "nested"] = "flat"
    allow_vision_skip: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "5m"


class TTSConfig(BaseModel):
    provider: Literal["fish", "elevenlabs", "piper", "kokoro"] = "fish"
    voice_id: str = ""
    sample_rate: int = 24000
    # Output device for Wallie's voice. "" = system default. Accepts a device index
    # or a name substring, e.g. "CABLE Input" to route TTS into VRChat via VB-CABLE.
    output_device: str = ""
    # Buffer del reproductor en muestras. Mas grande = mas margen para que el
    # hilo de audio llegue a tiempo (suena sin cortes) a cambio de latencia.
    # 4096 a 48 kHz son ~85 ms. 0 = usar el valor por defecto del reproductor.
    blocksize: int = 4096
    el_model_id: str = "eleven_turbo_v2_5"
    el_stability: float = 0.45
    el_similarity_boost: float = 0.75
    el_style: float = 0.0
    fish_latency_mode: Literal["normal", "balanced"] = "balanced"
    # Server-side text buffer before audio generation. Lower = faster time-to-first-audio
    # (snappier reactions), higher = smoother prosody. Range 100-300; 100 favors latency.
    fish_chunk_length: int = 100
    piper_model_path: str = ""
    piper_length_scale: float = 1.0
    # Optional ffmpeg -af chain applied to Piper output (Casavita's anime voice).
    # Empty = raw Piper. See tts/piper.py for the preset used by the profile.
    piper_postprocess: str = ""
    # Run Piper on the GPU (needs onnxruntime-gpu + CUDA libs). ~15x faster than
    # CPU on this laptop; automatically falls back to CPU if CUDA is missing.
    piper_use_cuda: bool = True
    # Kokoro — local, high-quality neural TTS (free, runs on CPU/GPU). voice e.g.
    # af_heart / am_adam / bf_emma; lang_code 'a'=US English, 'b'=UK. speed 0.5-2.0.
    kokoro_voice: str = "af_heart"
    kokoro_lang_code: str = "a"
    kokoro_speed: float = 1.0


class VisionConfig(BaseModel):
    enabled: bool = False
    source: Literal["monitor"] = "monitor"
    monitor_index: int = 1
    interval_sec: float = 3.0
    min_change_threshold: int = 8
    max_edge_px: int = 768
    llm_max_edge_px: int = 512
    llm_jpeg_quality: int = 50
    scene_change_threshold: int = 20
    min_emit_interval_sec: float = 8.0
    max_frame_age_sec: float = 5.0
    idle_variance_threshold: float = 15.0
    enrich_monologue: bool = False
    enrich_probability: float = 0.08

    organic_vision: bool = False
    organicity: float = 0.75
    never_interrupt_speech: bool = True
    min_vision_react_interval_sec: float = 20.0
    micro_change_threshold: int = 4
    idle_check_interval_sec: float = 45.0
    min_engagement_for_react: float = 0.35
    startup_delay_sec: float = 5.0
    # AttentionEngine reaction weights — how often a vision event becomes each kind of
    # reaction. Lower deep/glance + higher ignore/silence = talks LESS (good for games
    # where the whole screen changes constantly). Defaults match the engine's baseline.
    react_deep_base: float = 0.22
    react_glance_base: float = 0.28
    react_tangent_base: float = 0.05
    react_ignore_base: float = 0.27
    react_silence_base: float = 0.18
    # Scales the "fill the silence" fallback timer. >1 waits longer before narrating
    # into quiet stretches (less ambient chatter); <1 fills dead air sooner.
    silence_fallback_scale: float = 1.0
    # When False, Wallie NEVER starts a turn on her own. She only speaks when the
    # viewer writes in chat, the owner talks to her, or she is explicitly addressed.
    # Vision still runs (for reactions) but the "silence got too long → narrate"
    # fallbacks stay parked.
    spontaneous_monologue: bool = True
    # How strongly the app_switch/media "active content" reaction boost applies.
    # 1.0 = full boost (right for browsing, where switching apps is a real event).
    # Lower it for FULLSCREEN GAMES, where every camera move looks like an app_switch
    # and the boost makes the streamer over-talk. 0.0 = treat it like normal navigation.
    active_content_boost: float = 1.0


class HearingConfig(BaseModel):
    """Wallie's ears — captures system audio (game/video/music/voice) via loopback,
    transcribes speech, and feeds it to the orchestrator to fuse with vision."""
    enabled: bool = False
    window_sec: float = 5.0          # length of each capture+transcribe window
    model_size: str = "small"        # faster-whisper model (tiny/base/small/medium)
    language: str = ""               # "" = auto-detect; or "en", "tr", ...
    silence_threshold: float = 0.006  # RMS below this = silence, skipped entirely
    sound_event_threshold: float = 0.06  # loud non-speech still emits a "sound" event
    max_context_age_sec: float = 12.0    # how long a heard line stays relevant for fusion
    reply_gate_sec: float = 6.0          # min seconds between spoken reactions to heard audio
                                         # (lower it for snappy two-way conversation, e.g. 1.5)
    # --- Conversation latency / accuracy (two-way mode) ---
    # VAD segmentation: instead of fixed `window_sec` slices, detect when the person
    # starts and STOPS talking and transcribe the whole utterance the moment they pause.
    # Replies land right after they finish (low latency) and full sentences aren't cut.
    vad_segmentation: bool = False
    poll_interval_sec: float = 0.3       # how often to check for voice activity
    end_silence_sec: float = 0.6         # trailing silence that marks end-of-utterance
    max_utterance_sec: float = 12.0      # safety cap on a single utterance
    low_latency: bool = False            # fewer decode retries — faster, slightly less robust
    # --- Robustness (accents, background TV, messy mics) ---
    speech_only: bool = False            # dialogue only — never react to music / non-speech sound
    beam_size: int = 5                   # Whisper beam width (higher = more accurate on accents/noise)
    denoise: bool = False                # spectral noise reduction before STT (needs `noisereduce`)

    # --- Microphone source ---
    # PipeWire source name to record. "" = follow the system default input, so if the
    # streamer switches headset/mic in the mixer, Casavita follows automatically.
    # Set an explicit name (e.g. the Razer headset's source) to pin it.
    device: str = ""
    # --- Push-to-talk: only listen when asked ---
    # Always-on hearing makes her butt into every aside. With this on she keeps her ear
    # shut until she is called, then listens until the conversation goes quiet.
    push_to_talk: bool = True
    # Wake words that open her ear (lowercase, matched anywhere in the heard text).
    wake_words: list[str] = ["casavita", "casa vita"]
    # How long the ear stays open after a wake word before it closes again (seconds).
    listen_timeout_sec: float = 25.0
    # Close the ear this many seconds after the streamer stops talking (0 = never).
    close_after_silence_sec: float = 8.0
    # Ignore the beginning of the utterance so she doesn't answer her own wake word.
    wake_guard_sec: float = 1.2
    # While the ear is open, also react to plain speech (not just the wake word).
    answer_all_when_open: bool = True


class ChatConfig(BaseModel):
    youtube_enabled: bool = False
    twitch_enabled: bool = False
    kick_enabled: bool = False
    reply_probability: float = 0.35
    min_reply_interval_sec: float = 8.0
    max_message_age_sec: float = 45.0
    # Users who ALWAYS get an answer, no dice roll and no cooldown. Case-insensitive,
    # without the leading '@'. This is the streamer talking to her in their own chat.
    always_reply_usernames: list[str] = []
    # Users she must NEVER answer: alert/overlay bots (StreamElements, Streamer.bot,
    # Nightbot...), and any account whose messages are not a real person talking.
    # Case-insensitive, without the leading '@'.
    ignore_usernames: list[str] = []
    # Light roasting is on/off. When on, she is allowed one friendly jab per reply,
    # aimed at chat AND at casave himself.
    banter: bool = True
    # How often she pings casave with a question ("¿qué haces ahora?") when the chat
    # is quiet. 0.0 = never ask on her own. Independent of spontaneous_monologue.
    check_in_probability: float = 0.0
    check_in_min_idle_sec: float = 300.0


class TopicConfig(BaseModel):
    mode: Literal["list", "ai_picks"] = "ai_picks"
    topics: list[str] = Field(default_factory=lambda: [
        "Artificial intelligence and the future",
        "Strange decisions from tech companies",
        "Absurd observations from everyday life",
    ])
    switch_min_sec: float = 90.0
    switch_chance: float = 0.15
    drift_style: Literal["rigid", "natural", "freeform"] = "natural"


class OrchestratorConfig(BaseModel):
    segment_target_sec: float = 12.0
    dedupe_window: int = 8
    dedupe_threshold: float = 0.65
    prebuffer: bool = True
    max_words_per_sentence: int = 22

    segment_sentences_min: int = 3
    segment_sentences_max: int = 6
    max_audio_lookahead_sec: float = 8.0

    session_duration_min: float = 0.0
    outro_seconds: float = 30.0

    recent_verbatim_turns: int = 24
    summarize_every_n: int = 14
    max_messages: int = 200
    max_chars: int = 60000
    # Cap how many recent turns are actually SENT to the LLM each call (0 = no cap).
    # The rolling summary already carries older context, so trimming the verbatim
    # tail cuts prompt prefill → lower latency, with minimal continuity loss.
    llm_history_messages: int = 0
    # Auto-highlight: while streaming, flag Wallie's most clip-worthy moments to a
    # per-session JSONL in highlights/ so you can cut Shorts without scrubbing.
    auto_highlight: bool = False
    highlight_threshold: float = 0.55

    organic_enabled: bool = True
    silence_beat_min_sec: float = 2.0
    silence_beat_max_sec: float = 5.5
    silence_beat_ceiling: float = 0.35
    min_inter_segment_gap_sec: float = 0.35
    breathing_gap_max_sec: float = 2.5
    post_vision_silence_sec: float = 3.0

    enable_breaks: bool = True
    break_every_min: float = 8.0
    break_every_jitter: float = 0.35
    break_min_sec: float = 4.0
    break_max_sec: float = 12.0


class AvatarConfig(BaseModel):
    enabled: bool = False
    backend: str = "vts"              # "vts" or "web"
    web_url: str = "http://127.0.0.1:8100"
    vts_host: str = "127.0.0.1"
    vts_port: int = 8001

    param_mouth_open:  str = "MouthOpen"
    param_mouth_smile: str = "MouthSmile"
    param_mouth_form:  str = "ParamMouthForm"
    param_face_x:      str = "FaceAngleX"
    param_face_y:      str = "FaceAngleY"
    param_face_z:      str = "FaceAngleZ"
    param_eye_x:       str = "EyeLeftX"
    param_eye_y:       str = "EyeLeftY"
    param_brows:       str = "Brows"

    lipsync_gain:    float = 4.0
    lipsync_ceiling: float = 0.85
    lipsync_floor:   float = 0.02
    lipsync_attack:  float = 0.65
    lipsync_release: float = 0.30
    speaking_smile:  float = 0.15

    enable_viseme_lipsync: bool = True
    viseme_smoothing:      float = 0.35

    enable_idle_motion: bool = True
    idle_sway_amplitude: float = 4.0
    idle_sway_period_sec: float = 6.0
    enable_eye_darts: bool = True
    eye_dart_interval_sec: float = 4.5

    expr_happy:      str = ""
    expr_surprised:  str = ""
    expr_laughing:   str = ""
    expr_angry:      str = ""
    expr_sad:        str = ""
    expr_thinking:   str = ""
    expr_smug:       str = ""
    expr_eyeroll:    str = ""
    expr_confused:   str = ""
    expr_hype:       str = ""
    expr_deadpan:    str = ""

    enable_blink:          bool = True
    param_eye_open_left:   str = "EyeOpenLeft"
    param_eye_open_right:  str = "EyeOpenRight"
    blink_interval_sec:    float = 3.8
    blink_hold_sec:        float = 0.045
    double_blink_chance:   float = 0.15

    enable_body_motion:    bool = True
    param_body_x:          str = "BodyAngleX"
    param_body_y:          str = "BodyAngleY"
    param_body_z:          str = "BodyAngleZ"
    body_sway_amplitude:   float = 2.5
    body_sway_period_sec:  float = 9.0

    enable_mood_link:      bool = True
    mood_idle_min_scale:   float = 0.5
    mood_idle_max_scale:   float = 1.6
    mood_brow_min:         float = -0.4
    mood_brow_max:         float = 0.3
    mood_smile_max:        float = 0.20

    auto_map_expressions:  bool = True


class PlayConfig(BaseModel):
    """Minecraft Play mode — Wallie actually PLAYS the game (agent brain), and the streamer
    commentary is grounded in what the agent is really doing instead of guessing from the screen.
    When disabled, Wallie runs in standard vision mode (reacts to whatever is on screen)."""
    enabled: bool = False
    game: str = "minecraft"
    goal: str = (
        "Build a thriving Minecraft empire LIVE for an audience: gather and stockpile resources, "
        "craft full armour and tool sets, build varied good-looking structures, fight, explore and "
        "take on varied adventures. Progress toward the Ender Dragon over time, but keep the JOURNEY "
        "entertaining — this is a show, NOT a speedrun."
    )
    talk_from_agent: bool = True      # commentary uses the agent's real actions/state, not the frame
    hide_chat: bool = True            # hide in-game chat + Baritone commands on stream
    avoid_water: bool = True          # keep Baritone out of water (open-ground play)


class AppConfig(BaseModel):
    profile_name: str = "default"
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    hearing: HearingConfig = Field(default_factory=HearingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    play: PlayConfig = Field(default_factory=PlayConfig)


# -------------------------------------------------------------------
# Profiles (multi-persona support)
# -------------------------------------------------------------------
@dataclass
class Runtime:
    config: AppConfig
    secrets: Secrets
    base_dir: Path = BASE_DIR


def _ensure_dirs() -> None:
    PROFILES_DIR.mkdir(exist_ok=True)


def _active_profile_name() -> str:
    _ensure_dirs()
    if STATE_FILE.exists():
        try:
            import json
            return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("active", "default")
        except Exception:
            pass
    return "default"


def _set_active_profile_name(name: str) -> None:
    import json
    STATE_FILE.write_text(json.dumps({"active": name}), encoding="utf-8")


def _profile_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "default"
    return PROFILES_DIR / f"{safe}.yaml"


def list_profiles() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: Optional[str] = None) -> AppConfig:
    name = name or _active_profile_name()
    path = _profile_path(name)
    if not path.exists():
        cfg = AppConfig(profile_name=name)
        save_profile(cfg, name)
        _set_active_profile_name(name)
        return cfg
    try:
        import yaml  # type: ignore
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["profile_name"] = name
        return AppConfig(**data)
    except ModuleNotFoundError:
        import json
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data["profile_name"] = name
        return AppConfig(**data)


def save_profile(cfg: AppConfig, name: Optional[str] = None) -> None:
    _ensure_dirs()
    name = name or cfg.profile_name or "default"
    cfg = cfg.model_copy(update={"profile_name": name})
    path = _profile_path(name)
    data = cfg.model_dump()
    try:
        import yaml  # type: ignore
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ModuleNotFoundError:
        import json
        path.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def activate_profile(name: str) -> AppConfig:
    cfg = load_profile(name)
    _set_active_profile_name(name)
    return cfg


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        if _active_profile_name() == name:
            remaining = list_profiles()
            _set_active_profile_name(remaining[0] if remaining else "default")
        return True
    return False


def clone_profile(src: str, dst: str) -> AppConfig:
    cfg = load_profile(src)
    save_profile(cfg, dst)
    return load_profile(dst)


def get_runtime() -> Runtime:
    return Runtime(config=load_profile(), secrets=Secrets())


def load_config(*_args, **_kwargs) -> AppConfig:
    return load_profile()


def save_config(cfg: AppConfig, *_args, **_kwargs) -> None:
    save_profile(cfg)
