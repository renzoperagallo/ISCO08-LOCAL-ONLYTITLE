from __future__ import annotations

import numpy as np

from isco08.retriever import cosine_distance, top_k_one


def test_cosine_distance_orders_nearest_first():
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    indices, distances = cosine_distance(query, candidates, 3)
    assert indices[0].tolist() == [0, 1, 2]
    assert distances[0][0] == 0.0
    assert distances[0][1] == 1.0
    assert distances[0][2] == 2.0


def test_cosine_distance_normalizes_unnormalized_vectors():
    candidates = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    query = np.array([[10.0, 0.0]], dtype=np.float32)
    indices, distances = cosine_distance(query, candidates, 1)
    assert indices[0].tolist() == [0]
    assert distances[0][0] == 0.0


def test_top_k_one():
    candidates = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices, distances = top_k_one(np.array([0.0, 1.0]), candidates, 1)
    assert indices.tolist() == [1]
    assert distances[0] == 0.0
