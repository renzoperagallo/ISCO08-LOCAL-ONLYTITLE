from __future__ import annotations

import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def cosine_distance(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve indices y distancias coseno de los k candidatos mas cercanos.

    distance = 1 - cosine_similarity. Se ordena de menor a mayor distancia.
    """
    queries = np.asarray(query_embeddings, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            f"Dimensiones incompatibles: consultas {queries.shape}, "
            f"candidatos {candidates.shape}."
        )
    queries = _normalize(queries)
    candidates = _normalize(candidates)
    k = max(1, min(int(k), candidates.shape[0]))
    similarities = queries @ candidates.T
    distances = 1.0 - similarities
    indices = np.argsort(distances, axis=1)[:, :k]
    selected = np.take_along_axis(distances, indices, axis=1)
    return indices, selected.astype(np.float32)


def top_k_one(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices, distances = cosine_distance(
        np.asarray(query_embedding, dtype=np.float32).reshape(1, -1),
        candidate_embeddings,
        k,
    )
    return indices[0], distances[0]
