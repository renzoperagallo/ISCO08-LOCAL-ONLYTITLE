from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Settings
from .embeddings import GraniteEmbedder, encode_queries_cached
from .generator import (
    build_classification_messages,
    correct_code,
    format_candidate_block,
    parse_code,
    top_candidates_payload,
)
from .llm import ChatResult
from .retriever import cosine_distance


class JsonlCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        records: dict[str, dict] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in record:
                    records[str(record["id"])] = record
        return records

    def append(self, record: dict) -> None:
        record = dict(record)
        record["id"] = str(record["id"])
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                handle.flush()


def _call_many(
    client: Any,
    requests: Sequence[dict],
    max_workers: int,
    description: str,
) -> list[ChatResult | Exception]:
    if not requests:
        return []
    results: list[ChatResult | Exception | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, request in enumerate(requests):
            future = executor.submit(
                client.chat,
                request["messages"],
                max_tokens=int(request.get("max_tokens", 8192)),
                temperature=float(request.get("temperature", 0.0)),
            )
            futures[future] = index
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = exc
    return [item if item is not None else RuntimeError("Sin resultado") for item in results]


def _usage_from_result(result: ChatResult | Exception) -> tuple[int, int, bool]:
    if isinstance(result, Exception):
        return 0, 0, False
    usage = result.usage or {}
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        bool(result.cached),
    )


def _text_from_result(result: ChatResult | Exception) -> str:
    if isinstance(result, Exception):
        return ""
    return result.text or ""


def _error_from_result(result: ChatResult | Exception) -> str | None:
    if isinstance(result, Exception):
        return f"{type(result).__name__}: {result}"
    if result.finish_reason == "length":
        return "finish_reason=length: la respuesta se trunco"
    return None


def _timing_from_result(result: ChatResult | Exception) -> dict:
    if isinstance(result, Exception):
        return {}
    timings = dict(getattr(result, "timings", {}) or {})
    if "wall_latency_s" not in timings and getattr(result, "latency_s", 0):
        timings["wall_latency_s"] = float(result.latency_s)
    return timings


def _record_base(
    *,
    row: pd.Series,
    code: str | None,
    label_map: dict[str, str],
    payload: list[dict],
    distance: float,
    corrected: bool,
    fallback_used: bool,
    raw_response: str,
    usage_prompt: int,
    usage_completion: int,
    cached: bool,
    error: str | None,
    timings: dict | None = None,
) -> dict:
    timings = timings or {}
    latency_value = float(
        timings.get("wall_latency_s", timings.get("total_duration_s", 0.0)) or 0.0
    )
    generation_tps = timings.get("generation_tokens_per_s")
    prompt_tps = timings.get("prompt_tokens_per_s")
    if generation_tps is None and latency_value > 0 and usage_completion > 0:
        generation_tps = round(usage_completion / latency_value, 3)
    if prompt_tps is None and latency_value > 0 and usage_prompt > 0:
        prompt_tps = round(usage_prompt / latency_value, 3)
    return {
        "id": str(row.get("id", "")),
        "title": str(row.get("title", "")),
        "code": code or "",
        "label": label_map.get(code, "") if code else "",
        "distance": round(float(distance), 6),
        "similarity": round(1.0 - float(distance), 6),
        "corrected": bool(corrected),
        "fallback_used": bool(fallback_used),
        "llm_response": raw_response,
        "top_candidates": payload,
        "prompt_tokens": int(usage_prompt),
        "completion_tokens": int(usage_completion),
        "cached": bool(cached),
        "error": error,
        "latency_s": round(latency_value, 6),
        "prompt_eval_s": round(
            float(timings.get("prompt_eval_duration_s", 0.0) or 0.0), 6
        ),
        "eval_s": round(float(timings.get("eval_duration_s", 0.0) or 0.0), 6),
        "generation_tokens_per_s": generation_tps,
        "prompt_tokens_per_s": prompt_tps,
    }


def _resolve_code(
    raw_text: str,
    valid_codes: Sequence[str],
    fallback_code: str,
) -> tuple[str, bool, bool]:
    parsed = parse_code(raw_text, valid_codes)
    code, corrected = correct_code(parsed, valid_codes)
    fallback_used = False
    if code is None:
        code = fallback_code
        fallback_used = True
    return code, corrected, fallback_used


def _output_dataframe(records: Sequence[dict], ads: pd.DataFrame) -> pd.DataFrame:
    order = {str(value): index for index, value in enumerate(ads["id"].astype(str))}
    ordered = sorted(records, key=lambda rec: order.get(str(rec["id"]), 10**9))
    frame = pd.DataFrame(ordered)
    if frame.empty:
        return frame
    frame["top_candidates"] = frame["top_candidates"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
        if not isinstance(value, str)
        else value
    )
    return frame


def save_predictions(
    records: Sequence[dict],
    ads: pd.DataFrame,
    output_dir: Path,
    run_name: str,
) -> tuple[pd.DataFrame, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _output_dataframe(records, ads)
    csv_path = output_dir / f"predictions_{run_name}.csv"
    xlsx_path = output_dir / f"predictions_{run_name}.xlsx"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    excel_columns = [
        "id",
        "title",
        "code",
        "label",
        "distance",
        "similarity",
        "corrected",
        "fallback_used",
        "latency_s",
        "prompt_eval_s",
        "eval_s",
        "generation_tokens_per_s",
        "prompt_tokens_per_s",
        "top_candidates",
        "llm_response",
    ]
    existing = [column for column in excel_columns if column in frame.columns]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame[existing].to_excel(writer, sheet_name="predictions", index=False)
    return frame, csv_path, xlsx_path


def _distance_summary(frame: pd.DataFrame) -> dict:
    if frame.empty or "distance" not in frame.columns:
        return {}
    distance = pd.to_numeric(frame.get("distance"), errors="coerce").dropna()
    similarity = pd.to_numeric(frame.get("similarity"), errors="coerce").dropna()
    return {
        "mean_distance": round(float(distance.mean()), 6)
        if not distance.empty
        else None,
        "min_distance": round(float(distance.min()), 6)
        if not distance.empty
        else None,
        "mean_similarity": round(float(similarity.mean()), 6)
        if not similarity.empty
        else None,
    }


def _timing_summary(frame: pd.DataFrame) -> dict:
    if frame.empty or "latency_s" not in frame.columns:
        return {}
    latency = pd.to_numeric(frame.get("latency_s"), errors="coerce").dropna()
    generation = pd.to_numeric(
        frame.get("generation_tokens_per_s"), errors="coerce"
    ).dropna()
    prompt_tps = pd.to_numeric(
        frame.get("prompt_tokens_per_s"), errors="coerce"
    ).dropna()
    result = {
        "mean_latency_s": round(float(latency.mean()), 6) if not latency.empty else None,
        "median_latency_s": round(float(latency.median()), 6)
        if not latency.empty
        else None,
        "p95_latency_s": round(float(latency.quantile(0.95)), 6)
        if not latency.empty
        else None,
        "max_latency_s": round(float(latency.max()), 6) if not latency.empty else None,
        "mean_generation_tokens_per_s": round(float(generation.mean()), 3)
        if not generation.empty
        else None,
        "mean_prompt_tokens_per_s": round(float(prompt_tps.mean()), 3)
        if not prompt_tps.empty
        else None,
    }
    result.update(_distance_summary(frame))
    return result


def run_pipeline(
    ads: pd.DataFrame,
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    embedder: GraniteEmbedder,
    client: Any,
    settings: Settings,
    checkpoint_path: Path,
    output_dir: Path,
    run_name: str,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    started = time.monotonic()
    ads = ads.copy()
    ads["id"] = ads["id"].astype(str)
    if "title" not in ads.columns:
        raise ValueError("La entrada debe tener una columna title.")

    label_map = {
        str(row["code"]).zfill(4): str(row["label"])
        for _, row in labels.iterrows()
    }
    valid_codes = sorted(label_map)
    checkpoint = JsonlCheckpoint(checkpoint_path)
    done = checkpoint.load()

    todo = ads[~ads["id"].isin(done.keys())].reset_index(drop=True)
    if limit is not None:
        todo = todo.head(int(limit)).reset_index(drop=True)
    if todo.empty:
        print("[pipeline] No hay anuncios nuevos por procesar; se reutiliza el checkpoint.")
        records = list(checkpoint.load().values())
        frame, csv_path, xlsx_path = save_predictions(records, ads, output_dir, run_name)
        summary = {
            "run_name": run_name,
            "ads_total": int(len(ads)),
            "predictions": int(len(frame)),
            "note": "checkpoint ya completo",
            "provider": "ollama",
            "llm_model": settings.ollama_model,
            "llm": client.summary(),
            "timing": _timing_summary(frame),
            "elapsed_s": round(time.monotonic() - started, 2),
            "outputs": {"csv": str(csv_path), "xlsx": str(xlsx_path)},
        }
        return frame, summary

    print(f"[pipeline] Anuncios a procesar en esta corrida: {len(todo)}")
    titles = todo["title"].astype(str).str.strip().tolist()
    title_embeddings = encode_queries_cached(
        titles,
        embedder,
        cache_path=Path(checkpoint_path).parent / f"{run_name}_title_embeddings.npy",
        force=False,
        show_progress_bar=True,
    )
    top_indices, top_distances = cosine_distance(
        title_embeddings,
        candidate_embeddings,
        settings.top_k_retrieve,
    )

    stage1_requests = []
    for index, row in todo.iterrows():
        prompt_indices = top_indices[index][: settings.top_k_prompt]
        block = format_candidate_block(
            candidates,
            prompt_indices,
            max_examples=settings.examples_per_code,
            description_chars=settings.description_chars,
        )
        stage1_requests.append(
            {
                "messages": build_classification_messages(
                    title=str(row["title"]),
                    candidate_block=block,
                ),
                "max_tokens": settings.classification_max_tokens,
                "temperature": 0.0,
            }
        )

    stage1_results = _call_many(
        client,
        stage1_requests,
        max_workers=settings.max_workers,
        description="Clasificando titulos",
    )

    for index, row in todo.iterrows():
        result = stage1_results[index]
        payload = top_candidates_payload(
            candidates,
            top_indices[index][:10],
            top_distances[index][:10],
            description_chars=300,
        )
        fallback_code = payload[0]["code"] if payload else ""
        raw = _text_from_result(result)
        code, corrected, fallback_used = _resolve_code(raw, valid_codes, fallback_code)
        usage_prompt, usage_completion, cached = _usage_from_result(result)
        record = _record_base(
            row=row,
            code=code,
            label_map=label_map,
            payload=payload,
            distance=float(top_distances[index][0]),
            corrected=corrected,
            fallback_used=fallback_used,
            raw_response=raw,
            usage_prompt=usage_prompt,
            usage_completion=usage_completion,
            cached=cached,
            error=_error_from_result(result),
            timings=_timing_from_result(result),
        )
        checkpoint.append(record)

    records = list(checkpoint.load().values())
    known_ids = set(ads["id"].astype(str))
    records = [record for record in records if str(record.get("id")) in known_ids]
    frame, csv_path, xlsx_path = save_predictions(records, ads, output_dir, run_name)

    elapsed = time.monotonic() - started
    summary = {
        "run_name": run_name,
        "ads_total": int(len(ads)),
        "ads_processed_this_run": int(len(todo)),
        "predictions": int(len(frame)),
        "corrected": int(frame["corrected"].sum()) if not frame.empty else 0,
        "fallback_used": int(frame["fallback_used"].sum()) if not frame.empty else 0,
        "elapsed_s": round(elapsed, 2),
        "provider": "ollama",
        "llm_model": settings.ollama_model,
        "llm": client.summary(),
        "timing": _timing_summary(frame),
        "settings": {
            "model": settings.ollama_model,
            "provider": "ollama",
            "embedding_model": settings.embedding_model,
            "distance": "cosine (1 - similarity)",
            "top_k_retrieve": settings.top_k_retrieve,
            "top_k_prompt": settings.top_k_prompt,
            "max_workers": settings.max_workers,
        },
        "outputs": {"csv": str(csv_path), "xlsx": str(xlsx_path)},
    }
    summary_path = Path(output_dir) / f"run_summary_{run_name}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[pipeline] Predicciones: {len(frame)}")
    print(f"[pipeline] CSV: {csv_path}")
    print(f"[pipeline] Resumen: {summary_path}")
    return frame, summary
