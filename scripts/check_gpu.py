#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isco08.config import Settings, get_settings  # noqa: E402


def _print_settings(settings: Settings) -> None:
    raw_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "1")
    try:
        configured_parallel = int(raw_parallel)
    except ValueError:
        configured_parallel = 1
    effective_parallel = max(1, configured_parallel, settings.max_workers)

    print("[gpu-check] configuracion efectiva (.env + entorno):")
    print(f"  OLLAMA_BASE_URL={settings.ollama_base_url}")
    print(f"  OLLAMA_MODEL={settings.ollama_model}")
    print(f"  OLLAMA_NUM_CTX={settings.ollama_num_ctx}")
    print(f"  OLLAMA_NUM_GPU={settings.ollama_num_gpu}")
    print(
        f"  OLLAMA_NUM_PARALLEL={raw_parallel} "
        f"(make ollama-start usara {effective_parallel})"
    )
    print(f"  OLLAMA_BIN={os.getenv('OLLAMA_BIN', 'ollama (PATH)')}")
    if os.getenv("OLLAMA_MODELS"):
        print(f"  OLLAMA_MODELS={os.getenv('OLLAMA_MODELS')}")
    print(f"  MAX_WORKERS={settings.max_workers}")
    print(f"  EMBEDDING_MODEL={settings.embedding_model}")
    print(f"  EMBEDDING_DEVICE={settings.embedding_device}")
    print(f"  EMBEDDING_BATCH_SIZE={settings.embedding_batch_size}")


def _check_torch(settings: Settings) -> None:
    try:
        import torch
    except ImportError:
        print("[gpu-check] PyTorch: NO instalado")
        return

    print(f"[gpu-check] PyTorch: {torch.__version__}")
    print(f"[gpu-check] CUDA build: {torch.version.cuda}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"[gpu-check] CUDA disponible: {cuda_available}")

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
                memory = (
                    f"VRAM libre/total: {free / 1024**3:.2f}/"
                    f"{total / 1024**3:.2f} GB"
                )
            except Exception:  # noqa: BLE001
                memory = f"VRAM total: {props.total_memory / 1024**3:.2f} GB"
            print(f"[gpu-check] GPU {index}: {props.name} | {memory}")
    else:
        print("[gpu-check] No hay GPU CUDA visible para PyTorch.")

    requested = settings.embedding_device
    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        print(f"[gpu-check] EMBEDDING_DEVICE=auto se resolveria a: {resolved}")
    elif requested.startswith("cuda"):
        if not cuda_available:
            print(
                "[gpu-check] AVISO: EMBEDDING_DEVICE=cuda pero "
                "torch.cuda.is_available() es False."
            )
        elif ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                print(f"[gpu-check] AVISO: dispositivo CUDA invalido: {requested}")
            else:
                if index < 0 or index >= torch.cuda.device_count():
                    print(
                        f"[gpu-check] AVISO: GPU {index} no existe; hay "
                        f"{torch.cuda.device_count()} GPU(s)."
                    )
                else:
                    print(f"[gpu-check] EMBEDDING_DEVICE={requested} disponible.")
    elif requested == "cpu":
        print("[gpu-check] EMBEDDING_DEVICE=cpu: el embedder no usara la GPU.")
    elif requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        print(f"[gpu-check] MPS disponible: {bool(mps and mps.is_available())}")
    else:
        print(f"[gpu-check] AVISO: EMBEDDING_DEVICE desconocido: {requested}")


def _check_ollama(settings: Settings) -> None:
    try:
        import httpx
    except ImportError:
        print("[gpu-check] httpx no instalado; no se consulta Ollama.")
        return

    try:
        response = httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/ps",
            timeout=5.0,
        )
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu-check] Ollama no responde o no tiene /api/ps: {exc}")
        return

    if not models:
        print("[gpu-check] Ollama: ningun modelo cargado ahora mismo.")
    else:
        print("[gpu-check] Ollama: modelos cargados:")
        for item in models:
            size_vram = float(item.get("size_vram", 0) or 0) / 1024**3
            size = float(item.get("size", 0) or 0) / 1024**3
            print(
                f"  - {item.get('name', '?')} | VRAM {size_vram:.2f} GB "
                f"| total {size:.2f} GB | ctx {item.get('context_length', '?')}"
            )

    print(
        "[gpu-check] Para arrancar Ollama con ese paralelismo usa "
        "`make ollama-start` o `make ollama-start-bg`. "
        "Un servidor ya iniciado no se actualiza solo."
    )


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[gpu-check] Error de configuracion: {exc}") from exc

    _print_settings(settings)
    _check_torch(settings)
    _check_ollama(settings)


if __name__ == "__main__":
    main()
