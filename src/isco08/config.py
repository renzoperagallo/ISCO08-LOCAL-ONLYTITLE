from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    input: Path = PROJECT_ROOT / "data" / "input"
    isco: Path = PROJECT_ROOT / "data" / "isco08"
    embeddings: Path = PROJECT_ROOT / "data" / "isco08" / "embeddings"
    outputs: Path = PROJECT_ROOT / "outputs"
    checkpoints: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    cache: Path = PROJECT_ROOT / "data" / "cache"

    def ensure(self) -> "Paths":
        for path in (
            self.data,
            self.input,
            self.isco,
            self.embeddings,
            self.outputs,
            self.checkpoints,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Settings:
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).strip()
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL", "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S"
        ).strip()
    )
    ollama_num_ctx: int = field(
        default_factory=lambda: _env_int("OLLAMA_NUM_CTX", 8192)
    )
    ollama_num_gpu: int = field(
        default_factory=lambda: _env_int("OLLAMA_NUM_GPU", -1)
    )
    ollama_thinking: bool = field(
        default_factory=lambda: _env_bool("OLLAMA_THINKING", True)
    )
    ollama_keep_alive: str = field(
        default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "ibm-granite/granite-embedding-311m-multilingual-r2",
        ).strip()
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64)
    )
    top_k_retrieve: int = field(default_factory=lambda: _env_int("TOP_K_RETRIEVE", 20))
    top_k_prompt: int = field(default_factory=lambda: _env_int("TOP_K_PROMPT", 10))
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", 1))
    description_chars: int = field(
        default_factory=lambda: _env_int("DESCRIPTION_CHARS", 900)
    )
    examples_per_code: int = field(
        default_factory=lambda: _env_int("EXAMPLES_PER_CODE", 10)
    )
    classification_max_tokens: int = field(
        default_factory=lambda: _env_int("CLASSIFICATION_MAX_TOKENS", 0)
    )
    request_timeout: float = field(
        default_factory=lambda: _env_float("OLLAMA_TIMEOUT", 900.0)
    )
    max_retries: int = field(default_factory=lambda: _env_int("OLLAMA_MAX_RETRIES", 2))
    cache_enabled: bool = field(
        default_factory=lambda: _env_bool("OLLAMA_CACHE_ENABLED", True)
    )
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    def __post_init__(self) -> None:
        if self.classification_max_tokens <= 0:
            self.classification_max_tokens = 8192 if self.ollama_thinking else 256
        self.embedding_device = str(self.embedding_device).strip().lower()
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS debe ser >= 1")
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE debe ser >= 1")

    @property
    def model_slug(self) -> str:
        return self.embedding_model.replace("/", "__").replace(".", "_")

    def paths(self) -> Paths:
        return Paths().ensure()


def get_settings() -> Settings:
    settings = Settings()
    if not settings.ollama_model:
        raise ValueError("OLLAMA_MODEL no puede estar vacio")
    return settings
