import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.reading_dna import (
    ClusterConfig,
    add_book_attributes,
    aggregate_reader_attributes,
    assign_clusters,
    build_attribute_combinations,
    build_cluster_descriptions,
    deduplicate_books,
    extract_book_attributes,
    normalize_attribute,
    select_final_k,
    tune_cluster_count,
)


def test_normalize_attribute_removes_metadata_noise():
    assert normalize_attribute("Goodreads") == ""
    assert normalize_attribute("  Feminist Fiction  ") == "feminist fiction"
    assert normalize_attribute("New York") == ""


def test_extract_book_attributes_is_multilabel():
    row = pd.Series(
        {
            "subjects": [
                "Horror",
                "Feminist fiction",
                "Speculative fiction",
                "fiction",
                "Goodreads",
            ]
        }
    )

    assert extract_book_attributes(row) == [
        "feminist fiction",
        "horror",
        "speculative fiction",
    ]


def test_add_book_attributes():
    df = pd.DataFrame(
        [
            {"subjects": ["Horror", "Feminism"]},
            {"subjects": ["Romance"]},
        ]
    )

    result = add_book_attributes(df)

    assert result["attribute_count"].tolist() == [2, 1]


def test_deduplicate_books_by_source_id():
    df = pd.DataFrame(
        [
            {"source_book_id": "1", "title": "A", "author": "X"},
            {"source_book_id": "1", "title": "A", "author": "X"},
            {"source_book_id": "2", "title": "B", "author": "Y"},
        ]
    )

    result = deduplicate_books(df)

    assert len(result) == 2


def test_select_final_k_respects_guardrails():
    tuning = pd.DataFrame(
        [
            {
                "k": 6,
                "silhouette_mean": 0.30,
                "stability_ari_mean": 0.90,
                "min_cluster_pct": 0.03,
            },
            {
                "k": 8,
                "silhouette_mean": 0.35,
                "stability_ari_mean": 0.50,
                "min_cluster_pct": 0.02,
            },
            {
                "k": 10,
                "silhouette_mean": 0.31,
                "stability_ari_mean": 0.85,
                "min_cluster_pct": 0.02,
            },
        ]
    )

    assert select_final_k(tuning) == 10


def test_select_final_k_falls_back_when_no_candidate_is_eligible():
    tuning = pd.DataFrame(
        [
            {
                "k": 6,
                "silhouette_mean": 0.30,
                "stability_ari_mean": 0.50,
                "min_cluster_pct": 0.005,
            },
            {
                "k": 8,
                "silhouette_mean": 0.35,
                "stability_ari_mean": 0.40,
                "min_cluster_pct": 0.004,
            },
        ]
    )

    assert select_final_k(tuning) == 8


def test_assign_clusters_returns_requested_count():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(60, 8))
    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    labels, centroids, model = assign_clusters(
        embeddings,
        k=4,
    )

    assert len(labels) == 60
    assert centroids.shape == (4, 8)
    assert len(np.unique(labels)) == 4
    assert model.n_clusters == 4


def test_cluster_descriptions_use_attributes():
    df = pd.DataFrame(
        [
            {"attributes": ["horror", "feminist fiction"]},
            {"attributes": ["horror", "speculative fiction"]},
            {"attributes": ["romance"]},
        ]
    )

    result = build_cluster_descriptions(
        df,
        np.array([0, 0, 1]),
        top_terms=2,
    )

    row = result[result["cluster_id"] == 0].iloc[0]

    assert row["book_count"] == 2
    assert "horror" in row["top_attributes"]


def test_reader_attribute_aggregation_keeps_evidence_streams_separate():
    df = pd.DataFrame(
        [
            {
                "attributes": ["horror", "feminism"],
                "user_rating": 5,
                "reading_status": "read",
            },
            {
                "attributes": ["horror"],
                "user_rating": 2,
                "reading_status": "read",
            },
            {
                "attributes": ["horror", "feminism"],
                "user_rating": 0,
                "reading_status": "to_read",
            },
        ]
    )

    result = aggregate_reader_attributes(df, "you")

    horror = result[result["attribute"] == "horror"].iloc[0]

    assert horror["book_count"] == 3
    assert horror["positive_count"] == 1
    assert horror["negative_count"] == 1
    assert horror["observed_count"] == 2
    assert horror["intent_count"] == 1
    assert horror["preference_rate"] == 0.5


def test_attribute_combinations_are_pairwise():
    df = pd.DataFrame(
        [
            {
                "attributes": ["horror", "feminism", "speculative"],
                "user_rating": 5,
                "reading_status": "read",
            }
        ]
    )

    result = build_attribute_combinations(df, "you")

    assert len(result) == 3
    assert set(result["attribute_1"]) == {
        "feminism",
        "feminism",
        "horror",
    }


def test_tune_cluster_count_returns_requested_range():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(80, 6))
    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    result = tune_cluster_count(
        embeddings,
        ClusterConfig(min_k=2, max_k=4),
    )

    assert result["k"].tolist() == [2, 3, 4]
    assert result["stability_ari_mean"].notna().all()
