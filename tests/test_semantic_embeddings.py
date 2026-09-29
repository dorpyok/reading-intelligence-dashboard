from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.semantic_embeddings import (
    EMBEDDING_DIMENSIONS,
    load_saved_embeddings,
    save_embeddings,
    validate_semantic_books,
)


def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_book_id": [
                "book_001",
                "book_002",
                "book_003",
            ],
            "title": [
                "Book One",
                "Book Two",
                "Book Three",
            ],
            "author": [
                "Author One",
                "Author Two",
                "Author Three",
            ],
            "semantic_text": [
                "Book One Author One A story about magic.",
                "Book Two Author Two A story about friendship.",
                "Book Three Author Three A story about mystery.",
            ],
        }
    )


def test_valid_semantic_books_pass_validation():
    dataframe = make_valid_dataframe()

    validate_semantic_books(dataframe)


def test_missing_required_column_fails():
    dataframe = make_valid_dataframe().drop(
        columns=["semantic_text"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_semantic_books(dataframe)


def test_missing_canonical_id_fails():
    dataframe = make_valid_dataframe()
    dataframe.loc[0, "canonical_book_id"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_semantic_books(dataframe)


def test_duplicate_canonical_id_fails():
    dataframe = make_valid_dataframe()
    dataframe.loc[1, "canonical_book_id"] = "book_001"

    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_semantic_books(dataframe)


def test_missing_semantic_text_fails():
    dataframe = make_valid_dataframe()
    dataframe.loc[0, "semantic_text"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_semantic_books(dataframe)


def test_empty_semantic_text_fails():
    dataframe = make_valid_dataframe()
    dataframe.loc[0, "semantic_text"] = "   "

    with pytest.raises(ValueError, match="empty semantic text"):
        validate_semantic_books(dataframe)


def test_save_and_load_embeddings(tmp_path: Path):
    dataframe = make_valid_dataframe()

    embeddings = np.random.default_rng(42).normal(
        size=(len(dataframe), EMBEDDING_DIMENSIONS)
    ).astype(np.float32)

    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.csv"

    save_embeddings(
        embeddings=embeddings,
        dataframe=dataframe,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
    )

    loaded_embeddings, loaded_metadata = load_saved_embeddings(
        embedding_path=embedding_path,
        metadata_path=metadata_path,
    )

    np.testing.assert_array_equal(
        embeddings,
        loaded_embeddings,
    )

    assert loaded_metadata["canonical_book_id"].tolist() == [
        "book_001",
        "book_002",
        "book_003",
    ]

    assert loaded_metadata["embedding_row"].tolist() == [
        0,
        1,
        2,
    ]


def test_saved_embedding_dimension_is_validated(tmp_path: Path):
    dataframe = make_valid_dataframe()

    embeddings = np.random.default_rng(42).normal(
        size=(len(dataframe), 10)
    ).astype(np.float32)

    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.csv"

    np.save(
        embedding_path,
        embeddings,
    )

    metadata = dataframe[
        [
            "canonical_book_id",
            "title",
            "author",
        ]
    ].copy()

    metadata.insert(
        0,
        "embedding_row",
        range(len(metadata)),
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected embedding dimension",
    ):
        load_saved_embeddings(
            embedding_path=embedding_path,
            metadata_path=metadata_path,
        )