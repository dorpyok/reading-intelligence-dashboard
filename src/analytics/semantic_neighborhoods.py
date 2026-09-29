from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_K = 20


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embedding_artifacts(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    Validate that embeddings and metadata are aligned and usable.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D NumPy array."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding row count does not match metadata row count."
        )

    if "embedding_row" not in metadata.columns:
        raise ValueError(
            "Embedding metadata is missing 'embedding_row'."
        )

    if "canonical_book_id" not in metadata.columns:
        raise ValueError(
            "Embedding metadata is missing 'canonical_book_id'."
        )

    if "title" not in metadata.columns:
        raise ValueError(
            "Embedding metadata is missing 'title'."
        )

    if "author" not in metadata.columns:
        raise ValueError(
            "Embedding metadata is missing 'author'."
        )

    expected_rows = np.arange(len(metadata))

    actual_rows = metadata["embedding_row"].to_numpy()

    if not np.array_equal(actual_rows, expected_rows):
        raise ValueError(
            "Embedding metadata rows are not sequentially aligned."
        )

    if metadata["canonical_book_id"].duplicated().any():
        raise ValueError(
            "Embedding metadata contains duplicate canonical book IDs."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrix contains NaN or infinite values."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embedding_artifacts(
    embedding_path: Path,
    metadata_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load and validate the saved semantic embedding artifacts.
    """

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding metadata file not found: {metadata_path}"
        )

    embeddings = np.load(
        embedding_path
    )

    metadata = pd.read_csv(
        metadata_path
    )

    validate_embedding_artifacts(
        embeddings=embeddings,
        metadata=metadata,
    )

    return embeddings, metadata


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Normalize embedding rows to unit length.

    This allows cosine similarity to be calculated using
    a dot product.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Embedding matrix contains a zero-length vector."
        )

    return embeddings / norms


# ---------------------------------------------------------------------------
# Nearest neighbors
# ---------------------------------------------------------------------------

def find_nearest_neighbors(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """
    Find the k nearest semantic neighbors for every book.

    The book itself is excluded from its own neighborhood.

    Returns a DataFrame with one row per query-book / neighbor pair.
    """

    validate_embedding_artifacts(
        embeddings=embeddings,
        metadata=metadata,
    )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    n_books = len(embeddings)

    if n_books < 2:
        raise ValueError(
            "At least two books are required to calculate neighborhoods."
        )

    if k >= n_books:
        raise ValueError(
            f"k must be smaller than the number of books "
            f"({n_books})."
        )

    normalized_embeddings = _normalize_embeddings(
        embeddings
    )

    similarity_matrix = (
        normalized_embeddings
        @ normalized_embeddings.T
    )

    # Never allow a book to select itself.
    np.fill_diagonal(
        similarity_matrix,
        -np.inf,
    )

    rows = []

    for query_index in range(n_books):

        similarities = similarity_matrix[
            query_index
        ]

        # Get the k highest similarities without sorting
        # the entire row.
        candidate_indices = np.argpartition(
            similarities,
            -k,
        )[-k:]

        # Sort those k candidates from highest to lowest.
        candidate_indices = candidate_indices[
            np.argsort(
                similarities[candidate_indices]
            )[::-1]
        ]

        query_id = metadata.iloc[
            query_index
        ]["canonical_book_id"]

        query_title = metadata.iloc[
            query_index
        ]["title"]

        query_author = metadata.iloc[
            query_index
        ]["author"]

        for rank, neighbor_index in enumerate(
            candidate_indices,
            start=1,
        ):
            rows.append(
                {
                    "canonical_book_id": query_id,
                    "title": query_title,
                    "author": query_author,
                    "neighbor_rank": rank,
                    "neighbor_book_id": metadata.iloc[
                        neighbor_index
                    ]["canonical_book_id"],
                    "neighbor_title": metadata.iloc[
                        neighbor_index
                    ]["title"],
                    "neighbor_author": metadata.iloc[
                        neighbor_index
                    ]["author"],
                    "cosine_similarity": float(
                        similarities[neighbor_index]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Neighborhood summary
# ---------------------------------------------------------------------------

def summarize_neighborhoods(
    neighborhoods: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize neighborhood similarity statistics by book.
    """

    if neighborhoods.empty:
        raise ValueError(
            "Neighborhood dataframe is empty."
        )

    required_columns = {
        "canonical_book_id",
        "neighbor_rank",
        "cosine_similarity",
    }

    missing_columns = (
        required_columns
        - set(neighborhoods.columns)
    )

    if missing_columns:
        raise ValueError(
            "Neighborhood dataframe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        neighborhoods
        .groupby(
            "canonical_book_id",
            as_index=False,
        )
        .agg(
            neighbor_count=(
                "neighbor_rank",
                "count",
            ),
            mean_similarity=(
                "cosine_similarity",
                "mean",
            ),
            median_similarity=(
                "cosine_similarity",
                "median",
            ),
            min_similarity=(
                "cosine_similarity",
                "min",
            ),
            max_similarity=(
                "cosine_similarity",
                "max",
            ),
        )
    )

    return summary