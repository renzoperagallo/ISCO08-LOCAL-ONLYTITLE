from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_start_ollama(tmp_path, *args: str) -> subprocess.CompletedProcess:
    project = Path(__file__).resolve().parents[1]
    fake_binary = tmp_path / "ollama"
    fake_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_binary.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MAX_WORKERS": "4",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_NUM_CTX": "8192",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_BIN": str(fake_binary),
        }
    )
    return subprocess.run(
        [sys.executable, str(project / "scripts" / "start_ollama.py"), *args],
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        check=False,
    )


def test_start_ollama_raises_parallel_to_max_workers(tmp_path):
    result = _run_start_ollama(tmp_path, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "OLLAMA_NUM_PARALLEL=4" in result.stdout
    assert "se eleva a 4" in result.stdout
    assert "OLLAMA_CONTEXT_LENGTH=8192" in result.stdout


def test_start_ollama_accepts_a_higher_parallel_value(tmp_path):
    result = _run_start_ollama(tmp_path, "--num-parallel", "6", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "OLLAMA_NUM_PARALLEL=6" in result.stdout
