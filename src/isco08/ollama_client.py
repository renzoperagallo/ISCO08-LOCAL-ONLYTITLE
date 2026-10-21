from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from .llm import ChatResult, ResponseCache, UsageTracker


class OllamaClient:
    """Cliente para la API nativa de Ollama, con thinking y metricas."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_S",
        num_ctx: int = 8192,
        num_gpu: int = -1,
        thinking: bool = True,
        keep_alive: str = "30m",
        cache_path: Path | None = None,
        cache_enabled: bool = True,
        dry_run: bool = False,
        timeout: float = 900.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = int(num_ctx)
        self.num_gpu = int(num_gpu)
        self.thinking = bool(thinking)
        self.keep_alive = keep_alive
        self.dry_run = bool(dry_run)
        self.max_retries = max(1, int(max_retries))
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.cache = ResponseCache(cache_path) if cache_path else None
        self.cache_enabled = bool(cache_enabled)
        self.usage = UsageTracker()
        self._lock = threading.Lock()
        self._timings = {
            "total_duration_s": 0.0,
            "load_duration_s": 0.0,
            "prompt_eval_duration_s": 0.0,
            "eval_duration_s": 0.0,
        }
        self._prompt_eval_tokens = 0
        self._generated_tokens = 0

    @property
    def available(self) -> bool:
        if self.dry_run:
            return True
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        except Exception:
            return False
        names = {str(item.get("name", "")) for item in models}
        if self.model in names:
            return True
        return any(name.split(":")[0] == self.model.split(":")[0] for name in names)

    def _cache_key(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "provider": "ollama",
            "base_url": self.base_url,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "thinking": self.thinking,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _prepare_messages(self, messages: Sequence[dict]) -> list[dict]:
        prepared = [dict(item) for item in messages]
        if not self.thinking:
            return prepared
        token = "<|think|>"
        if prepared and prepared[0].get("role") == "system":
            content = str(prepared[0].get("content", ""))
            if not content.lstrip().startswith(token):
                prepared[0]["content"] = f"{token}\n{content}"
        else:
            prepared.insert(0, {"role": "system", "content": token})
        return prepared

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> ChatResult:
        prepared_messages = self._prepare_messages(messages)
        key = self._cache_key(prepared_messages, max_tokens, temperature)

        if self.cache is not None and self.cache_enabled and not self.dry_run:
            cached = self.cache.get(key)
            if cached is not None:
                text, usage = cached
                self.usage.add(usage, cached=True)
                return ChatResult(
                    text=text,
                    usage=usage,
                    cached=True,
                    dry_run=False,
                    model=self.model,
                    timings={},
                )

        if self.dry_run:
            return ChatResult(
                text="",
                usage={},
                cached=False,
                dry_run=True,
                model=self.model,
                timings={},
            )

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
        if self.num_gpu >= 0:
            options["num_gpu"] = self.num_gpu
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": prepared_messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if not self.thinking:
            payload["think"] = False

        started = time.monotonic()
        response = self._post_with_retry(payload)
        wall_latency = time.monotonic() - started
        data = response.json()
        message = data.get("message", {}) or {}
        text = message.get("content", "") or ""
        reasoning_content = message.get("thinking", "") or ""

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        timings = {
            "total_duration_s": round(float(data.get("total_duration", 0) or 0) / 1e9, 6),
            "load_duration_s": round(float(data.get("load_duration", 0) or 0) / 1e9, 6),
            "prompt_eval_duration_s": round(
                float(data.get("prompt_eval_duration", 0) or 0) / 1e9, 6
            ),
            "eval_duration_s": round(float(data.get("eval_duration", 0) or 0) / 1e9, 6),
            "wall_latency_s": round(wall_latency, 6),
        }
        if timings["eval_duration_s"] > 0:
            timings["generation_tokens_per_s"] = round(
                completion_tokens / timings["eval_duration_s"], 3
            )
        if timings["prompt_eval_duration_s"] > 0:
            timings["prompt_tokens_per_s"] = round(
                prompt_tokens / timings["prompt_eval_duration_s"], 3
            )

        if self.cache is not None and self.cache_enabled and text.strip():
            self.cache.set(key, text, usage)
        self.usage.add(usage, cached=False, latency_s=wall_latency)
        self._record_timings(timings, prompt_tokens, completion_tokens)
        return ChatResult(
            text=text,
            usage=usage,
            cached=False,
            dry_run=False,
            model=self.model,
            latency_s=wall_latency,
            reasoning_content=reasoning_content,
            timings=timings,
        )

    def _record_timings(
        self,
        timings: dict[str, float],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with self._lock:
            for key in (
                "total_duration_s",
                "load_duration_s",
                "prompt_eval_duration_s",
                "eval_duration_s",
            ):
                self._timings[key] += float(timings.get(key, 0.0) or 0.0)
            self._prompt_eval_tokens += prompt_tokens
            self._generated_tokens += completion_tokens

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post("/api/chat", json=payload)
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {response.text[:300]}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(20.0, 2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Fallo inesperado en Ollama.")

    def summary(self) -> dict:
        base = self.usage.summary(model=self.model, dry_run=self.dry_run)
        with self._lock:
            timings = dict(self._timings)
            generated = self._generated_tokens
            prompt_tokens = self._prompt_eval_tokens
        timings["generated_tokens"] = generated
        timings["prompt_eval_tokens"] = prompt_tokens
        if timings["eval_duration_s"] > 0:
            timings["generation_tokens_per_s"] = round(
                generated / timings["eval_duration_s"], 3
            )
        if timings["prompt_eval_duration_s"] > 0:
            timings["prompt_tokens_per_s"] = round(
                prompt_tokens / timings["prompt_eval_duration_s"], 3
            )
        return {
            "provider": "ollama",
            "base_url": self.base_url,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "thinking": self.thinking,
            **base,
            "timings": timings,
        }
