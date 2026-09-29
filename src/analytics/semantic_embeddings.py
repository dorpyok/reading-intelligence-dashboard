from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


def validate_semantic_books(
    dataframe: pd.DataFrame,
    id_column: str = "canonical_book_id",
    text_column: str = "semantic_text",
) -> None:
    """
    Validate the semantic book corpus before embedding.

    Raises:
        ValueError: If required columns are missing or invalid.
    """
    required_columns = {id_column, text_column}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Semantic book dataframe is empty.")

    if dataframe[id_column].isna().any():
        raise ValueError(
            f"Column '{id_column}' contains missing values."
        )

    if dataframe[id_column].duplicated().any():
        duplicates = dataframe.loc[
            dataframe[id_column].duplicated(keep=False),
            id_column,
        ].tolist()

        raise ValueError(
            f"Column '{id_column}' contains duplicate IDs. "
            f"Examples: {duplicates[:10]}"
        )

    if dataframe[text_column].isna().any():
        raise ValueError(
            f"Column '{text_column}' contains missing values."
        )

    empty_text = (
        dataframe[text_column]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_text.any():
        raise ValueError(
            f"Column '{text_column}' contains empty semantic text."
        )


def load_semantic_books(
    path: Path,
    id_column: str = "canonical_book_id",
    text_column: str = "semantic_text",
) -> pd.DataFrame:
    """
    Load and validate the semantic book corpus.
    """
    dataframe = pd.read_csv(path)

    validate_semantic_books(
        dataframe,
        id_column=id_column,
        text_column=text_column,
    )

    return dataframe


def load_embedding_model(
    model_name: str = MODEL_NAME,
) -> SentenceTransformer:
    """
    Load the sentence-transformer model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: Iterable[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Generate sentence embeddings.

    Returns:
        NumPy array with shape:
        (number_of_books, embedding_dimensions)
    """
    text_list = [str(text).strip() for text in texts]

    if not text_list:
        raise ValueError("No semantic text supplied for embedding.")

    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected {EMBEDDING_DIMENSIONS}, "
            f"got {embeddings.shape[1]}."
        )

    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    embedding_path: Path,
    metadata_path: Path,
    id_column: str = "canonical_book_id",
) -> None:
    """
    Save embeddings and the metadata needed to map rows back to books.
    """
    if len(embeddings) != len(dataframe):
        raise ValueError(
            "Embedding row count does not match dataframe row count."
        )

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embedding_path, embeddings)

    metadata = dataframe[
        [
            id_column,
            "title",
            "author",
        ]
    ].copy()

    metadata.insert(
        0,
        "embedding_row",
        np.arange(len(metadata)),
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )


def load_saved_embeddings(
    embedding_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load and validate a previously generated embedding dataset.
    """
    embeddings = np.load(embedding_path)
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {embeddings.shape}."
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected {EMBEDDING_DIMENSIONS}, "
            f"got {embeddings.shape[1]}."
        )

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embedding row count does not match metadata row count."
        )

    if metadata["embedding_row"].tolist() != list(range(len(metadata))):
        raise ValueError(
            "Embedding metadata rows are not sequential."
        )

    if metadata["canonical_book_id"].duplicated().any():
        raise ValueError(
            "Embedding metadata contains duplicate canonical book IDs."
        )

    return embeddings, metadata