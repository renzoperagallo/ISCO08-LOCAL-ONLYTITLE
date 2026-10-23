from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from isco08.config import Settings
from isco08.llm import ChatResult
from isco08.ollama_client import OllamaClient
from isco08.pipeline import JsonlCheckpoint, _call_many, run_pipeline


class FakeEmbedder:
    model_name = "fake-embedder"

    def encode(self, texts, batch_size=None, show_progress_bar=True):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "vendedor" in lowered:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_title_only_pipeline_dry_run(tmp_path):
    labels = pd.DataFrame(
        {
            "code": ["1111", "2222"],
            "label": ["Vendedores", "Cocineros"],
        }
    )
    candidates = pd.DataFrame(
        {
            "code": ["1111", "2222"],
            "label": ["Vendedores", "Cocineros"],
            "description": ["Venden productos", "Preparan alimentos"],
            "examples": [["Vendedor de tienda"], ["Cocinero de restaurante"]],
        }
    )
    candidate_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    ads = pd.DataFrame(
        {
            "id": ["a", "b"],
            "title": ["Vendedor de repuestos", "Cocinero"],
        }
    )
    settings = Settings(
        ollama_thinking=False,
        top_k_retrieve=2,
        top_k_prompt=2,
    )
    client = OllamaClient(thinking=False, dry_run=True)

    frame, summary = run_pipeline(
        ads=ads,
        labels=labels,
        candidates=candidates,
        candidate_embeddings=candidate_embeddings,
        embedder=FakeEmbedder(),
        client=client,
        settings=settings,
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        output_dir=tmp_path,
        run_name="test",
    )

    assert len(frame) == 2
    assert set(frame["code"]) == {"1111", "2222"}
    assert set(frame["fallback_used"]) == {True}
    assert set(frame["distance"]) == {0.0}
    assert summary["predictions"] == 2
    assert (tmp_path / "predictions_test.csv").exists()
    assert (tmp_path / "predictions_test.xlsx").exists()


def test_checkpoint_keeps_latest_record(tmp_path):
    checkpoint = JsonlCheckpoint(tmp_path / "run.jsonl")
    checkpoint.append({"id": "a", "code": "1111"})
    checkpoint.append({"id": "b", "code": "2222"})
    checkpoint.append({"id": "a", "code": "1112"})
    records = checkpoint.load()
    assert set(records) == {"a", "b"}
    assert records["a"]["code"] == "1112"


class _ConcurrentClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def chat(self, messages, max_tokens=1, temperature=0.0):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return ChatResult(text="1111")


def test_call_many_uses_configured_workers():
    client = _ConcurrentClient()
    requests = [
        {"messages": [], "max_tokens": 8, "temperature": 0.0}
        for _ in range(4)
    ]

    results = _call_many(client, requests, max_workers=4, description="test")

    assert len(results) == 4
    assert all(isinstance(result, ChatResult) for result in results)
    assert client.max_active >= 2
