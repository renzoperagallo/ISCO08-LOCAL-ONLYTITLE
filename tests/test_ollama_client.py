from __future__ import annotations

import pytest

from isco08.config import Settings
from isco08.llm import ResponseCache
from isco08.ollama_client import OllamaClient


def test_settings_defaults_to_title_only_local_pipeline(monkeypatch):
    for name in (
        "OLLAMA_MODEL",
        "OLLAMA_THINKING",
        "TOP_K_RETRIEVE",
        "TOP_K_PROMPT",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "MAX_WORKERS",
        "CLASSIFICATION_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()
    assert settings.ollama_model.startswith("hf.co/unsloth/gemma-4-E4B-it-GGUF")
    assert settings.ollama_thinking is True
    assert settings.top_k_retrieve == 20
    assert settings.top_k_prompt == 10
    assert settings.embedding_device == "cpu"
    assert settings.max_workers == 1
    assert settings.classification_max_tokens == 8192


def test_settings_reads_max_workers_from_env(monkeypatch):
    monkeypatch.setenv("MAX_WORKERS", "4")
    assert Settings().max_workers == 4


def test_settings_keeps_auto_embedding_device(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "auto")
    assert Settings().embedding_device == "auto"


def test_settings_normalizes_embedding_device(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", " CUDA ")
    assert Settings().embedding_device == "cuda"


def test_settings_rejects_zero_workers(monkeypatch):
    monkeypatch.setenv("MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="MAX_WORKERS"):
        Settings()


def test_settings_without_thinking_has_smaller_budget():
    settings = Settings(
        ollama_thinking=False,
        classification_max_tokens=0,
    )
    assert settings.classification_max_tokens == 256


def test_thinking_token_is_added_only_when_enabled():
    enabled = OllamaClient(thinking=True, dry_run=True)
    messages = enabled._prepare_messages(
        [{"role": "system", "content": "Clasifica."}]
    )
    assert messages[0]["content"].startswith("<|think|>\n")

    disabled = OllamaClient(thinking=False, dry_run=True)
    messages = disabled._prepare_messages(
        [{"role": "system", "content": "Clasifica."}]
    )
    assert not messages[0]["content"].startswith("<|think|>")


def test_response_cache(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.set("key", "4226", {"prompt_tokens": 10})
    assert cache.get("key") == ("4226", {"prompt_tokens": 10})
    assert cache.count() == 1
