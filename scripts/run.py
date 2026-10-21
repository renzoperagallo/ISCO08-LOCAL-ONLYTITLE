#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isco08.config import get_settings  # noqa: E402
from isco08.data_prep import load_labels  # noqa: E402
from isco08.embeddings import GraniteEmbedder, load_candidate_embeddings  # noqa: E402
from isco08.ollama_client import OllamaClient  # noqa: E402
from isco08.pipeline import JsonlCheckpoint, _timing_summary, run_pipeline  # noqa: E402


def _normalize_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename: dict[str, str] = {}
    for column in frame.columns:
        lowered = column.lower()
        if lowered == "id":
            rename[column] = "id"
        elif lowered == "title":
            rename[column] = "title"
    frame = frame.rename(columns=rename)
    missing = {"id", "title"} - set(frame.columns)
    if missing:
        raise SystemExit(f"El archivo de entrada debe tener ID y title. Faltan: {missing}")
    frame = frame[["id", "title"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame["title"] = frame["title"].astype(str)
    if (frame["title"].str.strip() == "").any():
        raise SystemExit("Hay titulos vacios en la entrada.")
    if (frame["id"].str.strip() == "").any():
        raise SystemExit("Hay IDs vacios en la entrada.")
    if frame["id"].duplicated().any():
        raise SystemExit("Hay IDs duplicados en la entrada.")
    return frame


def _final_summary(frame: pd.DataFrame, client: OllamaClient, elapsed: float) -> dict:
    corrected = (
        frame["corrected"].astype(str).str.lower().eq("true").sum()
        if not frame.empty
        else 0
    )
    fallback = (
        frame["fallback_used"].astype(str).str.lower().eq("true").sum()
        if not frame.empty
        else 0
    )
    distance = pd.to_numeric(frame.get("distance"), errors="coerce")
    similarity = pd.to_numeric(frame.get("similarity"), errors="coerce")
    return {
        "ads": int(len(frame)),
        "corrected": int(corrected),
        "fallback_used": int(fallback),
        "mean_distance": round(float(distance.mean()), 6)
        if distance is not None and not distance.dropna().empty
        else None,
        "mean_similarity": round(float(similarity.mean()), 6)
        if similarity is not None and not similarity.dropna().empty
        else None,
        "timing": _timing_summary(frame),
        "elapsed_s": round(elapsed, 2),
        "llm": client.summary(),
    }


def main() -> None:
    settings = get_settings()
    paths = settings.paths()
    parser = argparse.ArgumentParser(
        description="Clasifica titulos de empleo con retriever por distancia coseno y Gemma local."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=paths.input / "empleos_1000_titles.csv",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--run-name",
        default="titles",
        help="Nombre de la salida: predictions_<run>.csv/.xlsx.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Borra el checkpoint y empieza de cero. Sin esto, continua.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = todos.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe el archivo de entrada: {args.input}")
    ads = _normalize_input(args.input)
    labels = load_labels(paths.isco)
    candidates, candidate_embeddings = load_candidate_embeddings(
        paths.embeddings,
        settings.model_slug,
        "all",
    )

    checkpoint_path = paths.checkpoints / f"{args.run_name}.jsonl"
    csv_path = paths.outputs / f"predictions_{args.run_name}.csv"
    xlsx_path = paths.outputs / f"predictions_{args.run_name}.xlsx"
    summary_path = paths.outputs / f"run_summary_{args.run_name}.json"
    if args.fresh:
        for path in (
            checkpoint_path,
            csv_path,
            xlsx_path,
            summary_path,
            paths.outputs / "predictions.csv",
            paths.outputs / "predictions.xlsx",
        ):
            if path.exists():
                path.unlink()

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_ctx=settings.ollama_num_ctx,
        num_gpu=settings.ollama_num_gpu,
        thinking=settings.ollama_thinking,
        keep_alive=settings.ollama_keep_alive,
        cache_path=paths.cache / "ollama.sqlite",
        cache_enabled=settings.cache_enabled,
        dry_run=False,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
    if not client.available:
        raise SystemExit(
            f"Ollama no responde en {settings.ollama_base_url} o falta el modelo "
            f"{settings.ollama_model}. Arranca Ollama y descarga el modelo."
        )

    embedder = GraniteEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    print(
        f"[run] entrada={args.input} filas={len(ads)} batch={args.batch_size} "
        f"modelo={settings.ollama_model} thinking={settings.ollama_thinking} "
        f"ctx={settings.ollama_num_ctx} metrica=cosine_distance"
    )
    print(f"[run] checkpoint={checkpoint_path}")

    started = time.monotonic()
    target = min(args.limit, len(ads)) if args.limit else len(ads)
    while True:
        checkpoint = JsonlCheckpoint(checkpoint_path)
        done = len(checkpoint.load())
        if done >= target:
            break
        batch = min(args.batch_size, target - done)
        print(f"[run] lote: {batch} titulos (hechos {done}/{target})", flush=True)
        _, batch_summary = run_pipeline(
            ads=ads,
            labels=labels,
            candidates=candidates,
            candidate_embeddings=candidate_embeddings,
            embedder=embedder,
            client=client,
            settings=settings,
            checkpoint_path=checkpoint_path,
            output_dir=paths.outputs,
            run_name=args.run_name,
            limit=batch,
        )
        print(
            f"[run] progreso: {done + batch}/{target} | "
            f"lote={batch_summary.get('elapsed_s')}s",
            flush=True,
        )

    if not csv_path.exists():
        raise SystemExit("No se genero la salida. Revisa el checkpoint y los errores.")
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    summary = _final_summary(frame, client, time.monotonic() - started)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(csv_path, paths.outputs / "predictions.csv")
    if xlsx_path.exists():
        shutil.copy2(xlsx_path, paths.outputs / "predictions.xlsx")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[run] salida: {csv_path}")
    print(f"[run] resumen: {summary_path}")


if __name__ == "__main__":
    main()
