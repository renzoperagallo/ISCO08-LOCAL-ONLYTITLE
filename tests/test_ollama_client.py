from __future__ import annotations

from isco08.config import Settings
from isco08.llm import ResponseCache
from isco08.ollama_client import OllamaClient


def test_settings_defaults_to_title_only_local_pipeline():
    settings = Settings()
    assert settings.ollama_model.startswith("hf.co/unsloth/gemma-4-E4B-it-GGUF")
    assert settings.ollama_thinking is True
    assert settings.top_k_retrieve == 20
    assert settings.top_k_prompt == 10
    assert settings.embedding_device == "cpu"
    assert settings.classification_max_tokens == 8192


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
