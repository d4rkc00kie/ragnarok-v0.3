"""Configuration loader for Ollama TUI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    main_model: str = field(default_factory=lambda: os.getenv("MAIN_MODEL", "llama3.2"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))

    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"))
    elevenlabs_model_id: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    )
    tts_optimize_latency: int = field(
        default_factory=lambda: int(os.getenv("TTS_OPTIMIZE_LATENCY", "4"))
    )
    tts_output_format: str = field(
        default_factory=lambda: os.getenv("TTS_OUTPUT_FORMAT", "mp3_22050_32")
    )
    tts_fast_mode: bool = field(
        default_factory=lambda: os.getenv("TTS_FAST_MODE", "true").lower() in ("1", "true", "yes")
    )

    rag_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DIR", "./rag_docs")))
    chroma_path: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PATH", "./chroma_db")))
    max_history: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY", "20")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))

    tts_enabled: bool = field(default_factory=lambda: os.getenv("TTS_ENABLED", "false").lower() in ("1", "true", "yes"))
    tts_auto_play: bool = field(default_factory=lambda: os.getenv("TTS_AUTO_PLAY", "true").lower() in ("1", "true", "yes"))

    tools_enabled: bool = field(default_factory=lambda: os.getenv("TOOLS_ENABLED", "true").lower() in ("1", "true", "yes"))
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_ROUNDS", "5")))

    avatar_enabled: bool = field(default_factory=lambda: os.getenv("AVATAR_ENABLED", "true").lower() in ("1", "true", "yes"))
    avatar_backend: str = field(default_factory=lambda: os.getenv("AVATAR_BACKEND", "3d"))
    avatar_path: str = field(default_factory=lambda: os.getenv("AVATAR_PATH", "./avatar.webm"))
    avatar_width: int = field(default_factory=lambda: int(os.getenv("AVATAR_WIDTH", "380")))
    avatar_height: int = field(default_factory=lambda: int(os.getenv("AVATAR_HEIGHT", "420")))
    avatar_title: str = field(default_factory=lambda: os.getenv("AVATAR_TITLE", "Assistant"))
    avatar_skin: str = field(default_factory=lambda: os.getenv("AVATAR_SKIN", "90,170,210"))
    avatar_without_tts: bool = field(
        default_factory=lambda: os.getenv("AVATAR_WITHOUT_TTS", "true").lower() in ("1", "true", "yes")
    )

    unity_host: str = field(default_factory=lambda: os.getenv("UNITY_HOST", "127.0.0.1"))
    unity_port: int = field(default_factory=lambda: int(os.getenv("UNITY_PORT", "8765")))
    unity_auto_launch: bool = field(
        default_factory=lambda: os.getenv("UNITY_AUTO_LAUNCH", "false").lower() in ("1", "true", "yes")
    )
    unity_player_path: str = field(default_factory=lambda: os.getenv("UNITY_PLAYER_PATH", ""))
    unity_audio_dir: str = field(default_factory=lambda: os.getenv("UNITY_AUDIO_DIR", "/tmp/ollama-tui-unity-audio"))

    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.9")))
    top_k_gen: int = field(default_factory=lambda: int(os.getenv("TOP_K_GEN", "40")))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("NUM_CTX", "8192")))
    repeat_penalty: float = field(default_factory=lambda: float(os.getenv("REPEAT_PENALTY", "1.1")))

    system_prompt: str = (
        "You are a helpful local AI assistant running via Ollama. "
        "You have access to a RAG knowledge base of user files when relevant context is provided. "
        "You can call tools when needed: list/read/write files, run shell commands, calculate, "
        "get the current time, and search the RAG store. "
        "Prefer tools over guessing for file system or computation questions. "
        "Be concise, accurate, and cite file sources when you use retrieved context."
    )


def get_config() -> Config:
    return Config()
