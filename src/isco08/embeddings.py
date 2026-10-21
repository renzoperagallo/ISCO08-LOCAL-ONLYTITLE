from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _sha256_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class GraniteEmbedder:
    """Encoder local basado en sentence-transformers.

    El modelo por defecto es ibm-granite/granite-embedding-311m-multilingual-r2
    (768 dimensiones, Apache 2.0).
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-embedding-311m-multilingual-r2",
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _resolve_device(self) -> str:
        if self.requested_device in {"cpu", "cuda"}:
            if self.requested_device == "cuda":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("Se pidio cuda pero no hay GPU disponible.")
                except ImportError as exc:
                    raise RuntimeError("PyTorch no esta instalado.") from exc
            return self.requested_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Falta sentence-transformers. Ejecuta `make install`."
            ) from exc
        self._device = self._resolve_device()
        self._model = SentenceTransformer(self.model_name, device=self._device)

    @property
    def device(self) -> str:
        if self._device is None:
            self.load()
        return str(self._device)

    @property
    def dimension(self) -> int:
        self.load()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        self.load()
        size = batch_size or self.batch_size
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message and self._device == "cuda":
                import torch

                torch.cuda.empty_cache()
                self._model = None
                self.requested_device = "cpu"
                self._device = None
                return self.encode(texts, batch_size=size, show_progress_bar=show_progress_bar)
            raise
        return l2_normalize(np.asarray(embeddings, dtype=np.float32))


def build_candidate_documents(
    labels: pd.DataFrame,
    examples: pd.DataFrame,
    split: str = "all",
    max_examples: int = 10,
    description_max_chars: int = 2000,
) -> pd.DataFrame:
    """Construye el texto que representa a cada codigo CIUO.

    El texto sigue la idea del repo original: etiqueta + descripcion +
    ocupaciones de ejemplo. `split` permite excluir los ejemplos de test
    para que la evaluacion no tenga fuga de datos.
    """
    if split not in {"all", "train", "test"}:
        raise ValueError("split debe ser all, train o test.")
    if split == "all":
        subset = examples
    else:
        subset = examples[examples["split"] == split]

    grouped: dict[str, list[str]] = {}
    for code, group in subset.sort_values(["code", "order"]).groupby("code", sort=False):
        grouped[str(code)] = [str(title) for title in group["title"].tolist()]

    rows: list[dict] = []
    for _, label_row in labels.sort_values("code").iterrows():
        code = str(label_row["code"]).zfill(4)
        label = str(label_row["label"]).strip()
        description = str(label_row["description"]).strip()
        if len(description) > description_max_chars:
            description = description[:description_max_chars].rstrip()
        titles = grouped.get(code, [])[:max_examples]
        text = f"{label}\n{description}\nEjemplos: " + "; ".join(titles)
        rows.append(
            {
                "code": code,
                "label": label,
                "description": description,
                "examples": json.dumps(titles, ensure_ascii=False),
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def load_candidates(path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(path)
    candidates["code"] = candidates["code"].astype(str).str.zfill(4)
    candidates["examples"] = candidates["examples"].map(
        lambda value: json.loads(value) if isinstance(value, str) and value else []
    )
    return candidates


def candidate_cache_paths(
    out_dir: Path,
    model_slug: str,
    split: str,
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    return (
        out_dir / f"{model_slug}_{split}.npy",
        out_dir / f"candidates_{split}.parquet",
        out_dir / f"{model_slug}_{split}_meta.json",
    )


def build_candidate_embeddings(
    candidates: pd.DataFrame,
    embedder: GraniteEmbedder,
    out_dir: Path,
    model_slug: str,
    split: str,
    force: bool = False,
) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path, parquet_path, meta_path = candidate_cache_paths(out_dir, model_slug, split)
    texts = candidates["text"].tolist()
    text_hash = _sha256_texts(texts)

    if not force and npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model") == embedder.model_name
            and meta.get("texts_sha256") == text_hash
            and meta.get("count") == len(texts)
        ):
            return np.load(npy_path)

    embeddings = embedder.encode(texts, show_progress_bar=True)
    np.save(npy_path, embeddings)
    candidates.to_parquet(parquet_path, index=False)
    meta = {
        "model": embedder.model_name,
        "device": embedder.device,
        "dimension": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "split": split,
        "texts_sha256": text_hash,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return embeddings


def load_candidate_embeddings(
    out_dir: Path,
    model_slug: str,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    npy_path, parquet_path, _ = candidate_cache_paths(out_dir, model_slug, split)
    if not npy_path.exists() or not parquet_path.exists():
        raise FileNotFoundError(
            f"No existen embeddings para split={split}. "
            "Ejecuta `make embed` o `python scripts/build_embeddings.py --split both`."
        )
    candidates = load_candidates(parquet_path)
    embeddings = np.load(npy_path)
    if len(candidates) != len(embeddings):
        raise ValueError(
            f"Desalineacion: {len(candidates)} candidatos y {len(embeddings)} embeddings."
        )
    return candidates, embeddings


def encode_queries_cached(
    texts: Sequence[str],
    embedder: GraniteEmbedder,
    cache_path: Path,
    force: bool = False,
    show_progress_bar: bool = True,
) -> np.ndarray:
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    texts = [str(text) for text in texts]
    text_hash = _sha256_texts(texts)

    if not force and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model") == embedder.model_name
            and meta.get("texts_sha256") == text_hash
            and meta.get("count") == len(texts)
        ):
            return np.load(cache_path)

    embeddings = embedder.encode(texts, show_progress_bar=show_progress_bar)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model": embedder.model_name,
                "count": len(texts),
                "dimension": int(embeddings.shape[1]),
                "texts_sha256": text_hash,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings
