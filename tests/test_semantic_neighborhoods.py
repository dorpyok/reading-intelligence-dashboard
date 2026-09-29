import numpy as np
import pandas as pd
import pytest

from src.analytics.semantic_neighborhoods import (
    find_nearest_neighbors,
    summarize_neighborhoods,
    validate_embedding_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding_row": [0, 1, 2, 3],
            "canonical_book_id": [
                "book_a",
                "book_b",
                "book_c",
                "book_d",
            ],
            "title": [
                "Book A",
                "Book B",
                "Book C",
                "Book D",
            ],
            "author": [
                "Author A",
                "Author B",
                "Author C",
                "Author D",
            ],
        }
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_embedding_artifacts(
    sample_embeddings,
    sample_metadata,
):
    validate_embedding_artifacts(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
    )


def test_validation_rejects_row_mismatch(
    sample_embeddings,
    sample_metadata,
):
    metadata = sample_metadata.iloc[:3].copy()

    with pytest.raises(ValueError):
        validate_embedding_artifacts(
            embeddings=sample_embeddings,
            metadata=metadata,
        )


def test_validation_rejects_duplicate_ids(
    sample_embeddings,
    sample_metadata,
):
    metadata = sample_metadata.copy()

    metadata.loc[
        1,
        "canonical_book_id",
    ] = "book_a"

    with pytest.raises(ValueError):
        validate_embedding_artifacts(
            embeddings=sample_embeddings,
            metadata=metadata,
        )


def test_validation_rejects_nonsequential_rows(
    sample_embeddings,
    sample_metadata,
):
    metadata = sample_metadata.copy()

    metadata.loc[
        2,
        "embedding_row",
    ] = 99

    with pytest.raises(ValueError):
        validate_embedding_artifacts(
            embeddings=sample_embeddings,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Neighborhood tests
# ---------------------------------------------------------------------------

def test_nearest_neighbors_excludes_self(
    sample_embeddings,
    sample_metadata,
):
    neighborhoods = find_nearest_neighbors(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        k=2,
    )

    self_matches = (
        neighborhoods["canonical_book_id"]
        == neighborhoods["neighbor_book_id"]
    )

    assert not self_matches.any()


def test_nearest_neighbors_returns_k_per_book(
    sample_embeddings,
    sample_metadata,
):
    neighborhoods = find_nearest_neighbors(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        k=2,
    )

    counts = (
        neighborhoods
        .groupby("canonical_book_id")
        .size()
    )

    assert counts.tolist() == [2, 2, 2, 2]


def test_nearest_neighbors_are_ranked(
    sample_embeddings,
    sample_metadata,
):
    neighborhoods = find_nearest_neighbors(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        k=2,
    )

    for _, group in neighborhoods.groupby(
        "canonical_book_id"
    ):
        similarities = (
            group
            .sort_values("neighbor_rank")
            ["cosine_similarity"]
            .to_numpy()
        )

        assert np.all(
            similarities[:-1]
            >= similarities[1:]
        )


def test_obvious_neighbor_is_ranked_first(
    sample_embeddings,
    sample_metadata,
):
    neighborhoods = find_nearest_neighbors(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        k=1,
    )

    book_a = neighborhoods[
        neighborhoods["canonical_book_id"]
        == "book_a"
    ].iloc[0]

    assert book_a["neighbor_book_id"] == "book_b"

    assert book_a["cosine_similarity"] > 0.99


def test_invalid_k_is_rejected(
    sample_embeddings,
    sample_metadata,
):
    with pytest.raises(ValueError):
        find_nearest_neighbors(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            k=0,
        )


def test_k_cannot_equal_book_count(
    sample_embeddings,
    sample_metadata,
):
    with pytest.raises(ValueError):
        find_nearest_neighbors(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            k=len(sample_metadata),
        )


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

def test_summarize_neighborhoods(
    sample_embeddings,
    sample_metadata,
):
    neighborhoods = find_nearest_neighbors(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        k=2,
    )

    summary = summarize_neighborhoods(
        neighborhoods
    )

    assert len(summary) == 4

    assert set(
        summary["canonical_book_id"]
    ) == {
        "book_a",
        "book_b",
        "book_c",
        "book_d",
    }

    assert (
        summary["neighbor_count"]
        .tolist()
        == [2, 2, 2, 2]
    )