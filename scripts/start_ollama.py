#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isco08.config import get_settings  # noqa: E402


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(
            f"[ollama-start] {name} debe ser un entero: {raw!r}"
        ) from exc


def _server_host(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    if parsed.hostname:
        return f"{parsed.hostname}:{parsed.port or 11434}"
    return base_url


def _check_port(host: str) -> None:
    parsed = urlparse(f"//{host}")
    hostname = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    if hostname in {"", "0.0.0.0"}:
        hostname = "127.0.0.1"
    elif hostname == "::":
        hostname = "::1"

    try:
        with socket.create_connection((hostname, port), timeout=0.5):
            pass
    except OSError:
        return

    raise SystemExit(
        f"[ollama-start] Ya hay algo escuchando en {host}. "
        "Deten el servidor actual o usa --host con otro puerto."
    )


def _resolve_binary(value: str) -> str:
    expanded = os.path.expanduser(value)
    candidate = Path(expanded)
    if candidate.is_file():
        if not os.access(candidate, os.X_OK):
            raise SystemExit(f"[ollama-start] No es ejecutable: {candidate}")
        return str(candidate.resolve())

    found = shutil.which(expanded)
    if found:
        return found

    raise SystemExit(
        f"[ollama-start] No se encontro el binario de Ollama: {value!r}. "
        "Instalalo o define OLLAMA_BIN=/ruta/a/ollama."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Arranca Ollama con OLLAMA_NUM_PARALLEL >= MAX_WORKERS."
        )
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=0,
        help=(
            "Slots paralelos del servidor. 0 = usar OLLAMA_NUM_PARALLEL "
            "del .env y elevarlo a MAX_WORKERS si hace falta."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Contexto del servidor. 0 = usar OLLAMA_NUM_CTX del .env.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="Host:puerto del servidor. Por defecto, el de OLLAMA_BASE_URL.",
    )
    parser.add_argument(
        "--ollama-bin",
        default="",
        help="Ruta al binario de Ollama. Por defecto, OLLAMA_BIN o 'ollama'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra la configuracion y el comando sin arrancar Ollama.",
    )
    args = parser.parse_args()

    if args.num_parallel < 0:
        parser.error("--num-parallel debe ser >= 0")
    if args.context_length < 0:
        parser.error("--context-length debe ser >= 0")

    settings = get_settings()
    max_workers = settings.max_workers
    env_parallel = _env_int("OLLAMA_NUM_PARALLEL", 0)
    requested_parallel = args.num_parallel or env_parallel or max_workers
    effective_parallel = max(1, requested_parallel, max_workers)

    context_length = args.context_length or settings.ollama_num_ctx
    host = (
        _server_host(args.host.strip())
        if args.host.strip()
        else _server_host(settings.ollama_base_url)
    )
    binary_value = (
        args.ollama_bin.strip()
        or os.getenv("OLLAMA_BIN", "").strip()
        or "ollama"
    )
    binary = _resolve_binary(binary_value)

    print(f"[ollama-start] binario={binary}")
    print(f"[ollama-start] host={host}")
    print(
        f"[ollama-start] MAX_WORKERS={max_workers} "
        f"OLLAMA_NUM_PARALLEL={effective_parallel}"
    )
    if requested_parallel < max_workers:
        print(
            f"[ollama-start] OLLAMA_NUM_PARALLEL solicitado "
            f"({requested_parallel}) se eleva a {max_workers} "
            "para cubrir MAX_WORKERS."
        )
    print(f"[ollama-start] OLLAMA_CONTEXT_LENGTH={context_length}")
    print(
        "[ollama-start] Cada slot paralelo reserva su propio KV cache; "
        "verifica la VRAM con `ollama ps`."
    )

    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    env["OLLAMA_NUM_PARALLEL"] = str(effective_parallel)
    env["OLLAMA_CONTEXT_LENGTH"] = str(context_length)
    env.setdefault("OLLAMA_KEEP_ALIVE", settings.ollama_keep_alive)

    if args.dry_run:
        print(
            "[ollama-start] dry-run: "
            f"OLLAMA_HOST={host} "
            f"OLLAMA_NUM_PARALLEL={effective_parallel} "
            f"OLLAMA_CONTEXT_LENGTH={context_length} "
            f"{binary} serve"
        )
        return

    _check_port(host)
    print("[ollama-start] ejecutando: ollama serve ...", flush=True)
    os.execvpe(binary, [binary, "serve"], env)


if __name__ == "__main__":
    main()
