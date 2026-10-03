from unittest import result

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.analytics.semantic_neighborhoods import (
    NeighborhoodConfig,
    add_book_identity_to_neighbors,
    build_neighborhood_assignments,
    calculate_neighborhood_coherence,
    calculate_parameter_stability,
    discover_semantic_neighborhoods,
    find_nearest_neighbors,
    normalize_embeddings,
    summarize_neighborhood_solution,
    validate_embedding_matrix,
)


def make_clustered_embeddings(
    n_samples: int = 80,
) -> np.ndarray:
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=3,
        n_features=6,
        cluster_std=0.35,
        random_state=42,
    )

    X = X.astype(float)

    X /= np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    return X


def test_validate_embedding_matrix_accepts_valid_matrix():
    embeddings = np.ones((10, 4))

    validate_embedding_matrix(
        embeddings
    )


def test_validate_embedding_matrix_rejects_1d_array():
    embeddings = np.ones(10)

    try:
        validate_embedding_matrix(
            embeddings
        )
    except ValueError as exc:
        assert "2-dimensional" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_normalize_embeddings_returns_unit_vectors():
    embeddings = np.array(
        [
            [3.0, 4.0],
            [1.0, 0.0],
        ]
    )

    result = normalize_embeddings(
        embeddings
    )

    norms = np.linalg.norm(
        result,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
    )


def test_find_nearest_neighbors_returns_k_rows_per_book():
    embeddings = make_clustered_embeddings(
        30
    )

    result = find_nearest_neighbors(
        embeddings,
        k=5,
    )

    assert len(result) == 30 * 5

    counts = (
        result.groupby(
            "query_embedding_row"
        )
        .size()
    )

    assert counts.tolist() == [5] * 30


def test_nearest_neighbors_do_not_include_self():
    embeddings = make_clustered_embeddings(
        20
    )

    result = find_nearest_neighbors(
        embeddings,
        k=4,
    )

    assert not (
        result["query_embedding_row"]
        == result["neighbor_embedding_row"]
    ).any()


def test_add_book_identity_to_neighbors():
    embeddings = make_clustered_embeddings(
        10
    )

    neighbors = find_nearest_neighbors(
        embeddings,
        k=2,
    )

    metadata = pd.DataFrame(
        {
            "embedding_row": range(10),
            "canonical_book_id": [
                f"book_{i}"
                for i in range(10)
            ],
            "title": [
                f"Book {i}"
                for i in range(10)
            ],
            "author": [
                f"Author {i}"
                for i in range(10)
            ],
        }
    )

    result = add_book_identity_to_neighbors(
        neighbors,
        metadata,
    )

    assert len(result) == 20

    assert {
        "query_book_id",
        "neighbor_book_id",
        "query_title",
        "neighbor_title",
    }.issubset(result.columns)


def test_discover_semantic_neighborhoods_allows_noise():
    embeddings = make_clustered_embeddings(
        80
    )

    labels, model = (
        discover_semantic_neighborhoods(
            embeddings,
            NeighborhoodConfig(
                min_cluster_size=5,
                min_samples=3,
            ),
        )
    )

    assert len(labels) == 80

    assert hasattr(
        model,
        "probabilities_",
    )

    assert len(
        np.unique(labels)
    ) >= 2


def test_discover_semantic_neighborhoods_can_produce_noise():
    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(80, 12)
    )

    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    labels, _ = (
        discover_semantic_neighborhoods(
            embeddings,
            NeighborhoodConfig(
                min_cluster_size=8,
                min_samples=5,
            ),
        )
    )

    assert len(labels) == 80

    assert -1 in labels


def test_neighborhood_coherence_returns_assigned_neighborhoods():
    embeddings = make_clustered_embeddings(
        60
    )

    labels, _ = (
        discover_semantic_neighborhoods(
            embeddings,
            NeighborhoodConfig(
                min_cluster_size=5,
                min_samples=3,
            ),
        )
    )

    result = calculate_neighborhood_coherence(
        embeddings,
        labels,
    )

    assert (
        "neighborhood_id"
        in result.columns
    )

    assert (
        "coherence_mean"
        in result.columns
    )

    assert not (
        result["neighborhood_id"]
        == -1
    ).any()


def test_solution_summary_counts_noise():
    labels = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            -1,
        ]
    )

    embeddings = np.eye(6)

    result = summarize_neighborhood_solution(
        embeddings,
        labels,
    )

    assert result["book_count"] == 6
    assert result["neighborhood_count"] == 2
    assert result["assigned_books"] == 5
    assert result["noise_books"] == 1
    assert result["noise_pct"] == pytest.approx(1 / 6)


def test_parameter_stability_returns_one_row_per_setting():
    embeddings = make_clustered_embeddings(
        80
    )

    result = calculate_parameter_stability(
        embeddings,
        min_cluster_sizes=(5, 8, 12),
        min_samples=3,
    )

    assert result[
        "min_cluster_size"
    ].tolist() == [5, 8, 12]

    assert (
        result[
            "parameter_stability_ari"
        ]
        .iloc[1:]
        .notna()
        .all()
    )


def test_build_neighborhood_assignments():
    metadata = pd.DataFrame(
        {
            "embedding_row": [0, 1, 2],
            "canonical_book_id": [
                "book_1",
                "book_2",
                "book_3",
            ],
            "title": [
                "A",
                "B",
                "C",
            ],
            "author": [
                "X",
                "Y",
                "Z",
            ],
        }
    )

    labels = np.array(
        [0, -1, 1]
    )

    probabilities = np.array(
        [0.9, 0.0, 0.8]
    )

    result = build_neighborhood_assignments(
        metadata,
        labels,
        probabilities,
    )

    assert len(result) == 3

    assert result[
        "neighborhood_id"
    ].tolist() == [0, -1, 1]

    assert result[
        "is_noise"
    ].tolist() == [
        False,
        True,
        False,
    ]