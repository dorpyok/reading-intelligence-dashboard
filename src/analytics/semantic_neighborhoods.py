from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_K_NEIGHBORS = 20
DEFAULT_MIN_CLUSTER_SIZE = 8
DEFAULT_MIN_SAMPLES = 3


@dataclass(frozen=True)
class NeighborhoodConfig:
    """
    Configuration for density-based semantic neighborhood discovery.

    HDBSCAN does not require a predetermined number of neighborhoods.
    Books that do not belong to a sufficiently dense region may remain
    unassigned as noise (-1).
    """

    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    min_samples: int = DEFAULT_MIN_SAMPLES
    cluster_selection_method: str = "eom"


def validate_embedding_matrix(
    embeddings: np.ndarray,
) -> None:
    """Validate the semantic embedding matrix."""

    if not isinstance(embeddings, np.ndarray):
        raise TypeError(
            "Embeddings must be a numpy array."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be 2-dimensional."
        )

    if embeddings.shape[0] < 2:
        raise ValueError(
            "At least two embeddings are required."
        )

    if embeddings.shape[1] < 2:
        raise ValueError(
            "Embeddings must contain at least two dimensions."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain non-finite values."
        )


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings.

    For normalized vectors, Euclidean distance is monotonically related
    to cosine distance, making it suitable for HDBSCAN neighborhood
    discovery while preserving the semantic geometry of the embeddings.
    """

    validate_embedding_matrix(embeddings)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / np.clip(
        norms,
        1e-12,
        None,
    )


def find_nearest_neighbors(
    embeddings: np.ndarray,
    k: int = DEFAULT_K_NEIGHBORS,
) -> pd.DataFrame:
    """
    Find the top-k cosine nearest neighbors for every book.

    This remains useful as a local semantic graph even though the
    Reading DNA neighborhood representation is now density-based.
    """

    embeddings = normalize_embeddings(
        embeddings
    )

    n_books = embeddings.shape[0]

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    if k >= n_books:
        raise ValueError(
            "k must be smaller than the number "
            "of books."
        )

    similarities = cosine_similarity(
        embeddings
    )

    np.fill_diagonal(
        similarities,
        -np.inf,
    )

    neighbor_indices = np.argpartition(
        -similarities,
        kth=k - 1,
        axis=1,
    )[:, :k]

    rows: list[dict] = []

    for query_index in range(n_books):
        indices = neighbor_indices[
            query_index
        ]

        indices = indices[
            np.argsort(
                -similarities[
                    query_index,
                    indices,
                ]
            )
        ]

        for rank, neighbor_index in enumerate(
            indices,
            start=1,
        ):
            rows.append(
                {
                    "query_embedding_row": int(
                        query_index
                    ),
                    "neighbor_embedding_row": int(
                        neighbor_index
                    ),
                    "rank": int(rank),
                    "similarity": float(
                        similarities[
                            query_index,
                            neighbor_index,
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def add_book_identity_to_neighbors(
    neighbors: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add canonical book IDs and titles to nearest-neighbor relationships.
    """

    required_metadata = {
        "embedding_row",
        "canonical_book_id",
        "title",
        "author",
    }

    missing = (
        required_metadata
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Embedding metadata is missing "
            f"required columns: {sorted(missing)}"
        )

    query_metadata = metadata.rename(
        columns={
            "embedding_row": "query_embedding_row",
            "canonical_book_id": "query_book_id",
            "title": "query_title",
            "author": "query_author",
        }
    )

    neighbor_metadata = metadata.rename(
        columns={
            "embedding_row": "neighbor_embedding_row",
            "canonical_book_id": "neighbor_book_id",
            "title": "neighbor_title",
            "author": "neighbor_author",
        }
    )

    result = neighbors.merge(
        query_metadata[
            [
                "query_embedding_row",
                "query_book_id",
                "query_title",
                "query_author",
            ]
        ],
        on="query_embedding_row",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        neighbor_metadata[
            [
                "neighbor_embedding_row",
                "neighbor_book_id",
                "neighbor_title",
                "neighbor_author",
            ]
        ],
        on="neighbor_embedding_row",
        how="left",
        validate="many_to_one",
    )

    return result


def discover_semantic_neighborhoods(
    embeddings: np.ndarray,
    config: NeighborhoodConfig = NeighborhoodConfig(),
) -> tuple[np.ndarray, HDBSCAN]:
    """
    Discover fine-grained semantic neighborhoods.

    Returns:
        labels:
            Integer neighborhood IDs.
            -1 means HDBSCAN considered the book noise/unassigned.

        clusterer:
            Fitted sklearn HDBSCAN model.
    """

    embeddings = normalize_embeddings(
        embeddings
    )

    if config.min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    if config.min_samples < 1:
        raise ValueError(
            "min_samples must be at least 1."
        )

    if config.min_cluster_size > len(
        embeddings
    ):
        raise ValueError(
            "min_cluster_size cannot exceed "
            "the number of embeddings."
        )

    clusterer = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        cluster_selection_method=(
            config.cluster_selection_method
        ),
        allow_single_cluster=False,
    )

    labels = clusterer.fit_predict(
        embeddings
    )

    return labels, clusterer


def calculate_neighborhood_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate semantic coherence for each discovered neighborhood.

    Coherence is the mean cosine similarity of books in the neighborhood
    to that neighborhood's normalized centroid.

    Noise (-1) is excluded from neighborhood coherence.
    """

    embeddings = normalize_embeddings(
        embeddings
    )

    labels = np.asarray(labels)

    if len(labels) != len(embeddings):
        raise ValueError(
            "Labels and embeddings must have "
            "the same number of rows."
        )

    rows: list[dict] = []

    for neighborhood_id in sorted(
        set(labels)
    ):
        if neighborhood_id == -1:
            continue

        indices = np.where(
            labels == neighborhood_id
        )[0]

        neighborhood_embeddings = (
            embeddings[indices]
        )

        centroid = neighborhood_embeddings.mean(
            axis=0
        )

        centroid_norm = np.linalg.norm(
            centroid
        )

        if centroid_norm == 0:
            coherence = np.nan
        else:
            centroid = (
                centroid
                / centroid_norm
            )

            coherence = float(
                np.mean(
                    neighborhood_embeddings
                    @ centroid
                )
            )

        rows.append(
            {
                "neighborhood_id": int(
                    neighborhood_id
                ),
                "book_count": int(
                    len(indices)
                ),
                "coherence_mean": coherence,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "neighborhood_id",
                "book_count",
                "coherence_mean",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("neighborhood_id")
        .reset_index(drop=True)
    )


def summarize_neighborhood_solution(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """
    Produce high-level diagnostics for a neighborhood solution.
    """

    embeddings = normalize_embeddings(
        embeddings
    )

    labels = np.asarray(labels)

    if len(labels) != len(embeddings):
        raise ValueError(
            "Labels and embeddings must have "
            "the same number of rows."
        )

    total_books = len(labels)

    noise_count = int(
        (labels == -1).sum()
    )

    assigned_count = (
        total_books - noise_count
    )

    assigned_pct = (
        assigned_count / total_books
    )

    neighborhood_ids = sorted(
        set(labels)
        - {-1}
    )

    sizes = [
        int(
            (labels == neighborhood_id).sum()
        )
        for neighborhood_id in neighborhood_ids
    ]

    coherence = (
        calculate_neighborhood_coherence(
            embeddings,
            labels,
        )
    )

    return {
        "book_count": total_books,
        "neighborhood_count": len(
            neighborhood_ids
        ),
        "assigned_books": assigned_count,
        "noise_books": noise_count,
        "assigned_pct": float(
            assigned_pct
        ),
        "noise_pct": float(
            1 - assigned_pct
        ),
        "min_neighborhood_size": (
            min(sizes)
            if sizes
            else 0
        ),
        "median_neighborhood_size": (
            float(np.median(sizes))
            if sizes
            else 0.0
        ),
        "max_neighborhood_size": (
            max(sizes)
            if sizes
            else 0
        ),
        "mean_coherence": (
            float(
                coherence[
                    "coherence_mean"
                ].mean()
            )
            if not coherence.empty
            else np.nan
        ),
        "median_coherence": (
            float(
                coherence[
                    "coherence_mean"
                ].median()
            )
            if not coherence.empty
            else np.nan
        ),
    }


def calculate_parameter_stability(
    embeddings: np.ndarray,
    min_cluster_sizes: tuple[int, ...],
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Evaluate stability across neighborhood granularity settings.

    Stability is measured with adjusted Rand index between neighboring
    parameter settings.

    This deliberately does not choose a winner. It produces evidence for
    selecting the neighborhood granularity after inspection.
    """

    solutions: dict[int, np.ndarray] = {}

    for min_cluster_size in min_cluster_sizes:
        labels, _ = discover_semantic_neighborhoods(
            embeddings,
            NeighborhoodConfig(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            ),
        )

        solutions[
            min_cluster_size
        ] = labels

    rows: list[dict] = []

    ordered_sizes = sorted(
        solutions
    )

    for index, min_cluster_size in enumerate(
        ordered_sizes
    ):
        labels = solutions[
            min_cluster_size
        ]

        summary = summarize_neighborhood_solution(
            embeddings,
            labels,
        )

        stability_values: list[float] = []

        if index > 0:
            previous_size = (
                ordered_sizes[index - 1]
            )

            stability_values.append(
                adjusted_rand_score(
                    solutions[
                        previous_size
                    ],
                    labels,
                )
            )

        if index < len(
            ordered_sizes
        ) - 1:
            next_size = (
                ordered_sizes[index + 1]
            )

            stability_values.append(
                adjusted_rand_score(
                    labels,
                    solutions[
                        next_size
                    ],
                )
            )

        rows.append(
            {
                "min_cluster_size": int(
                    min_cluster_size
                ),
                "min_samples": int(
                    min_samples
                ),
                "parameter_stability_ari": (
                    float(
                        np.mean(
                            stability_values
                        )
                    )
                    if stability_values
                    else np.nan
                ),
                **summary,
            }
        )

    return pd.DataFrame(rows)


def build_neighborhood_assignments(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build the persistent book-level neighborhood assignment artifact.
    """

    required = {
        "embedding_row",
        "canonical_book_id",
        "title",
        "author",
    }

    missing = (
        required
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Metadata is missing required "
            f"columns: {sorted(missing)}"
        )

    labels = np.asarray(labels)

    if len(labels) != len(metadata):
        raise ValueError(
            "Labels and metadata must have "
            "the same number of rows."
        )

    if probabilities is None:
        probabilities = np.full(
            len(labels),
            np.nan,
        )
    else:
        probabilities = np.asarray(
            probabilities
        )

        if len(probabilities) != len(
            labels
        ):
            raise ValueError(
                "Probabilities and labels must "
                "have the same length."
            )

    result = metadata[
        [
            "embedding_row",
            "canonical_book_id",
            "title",
            "author",
        ]
    ].copy()

    result["neighborhood_id"] = (
        labels.astype(int)
    )

    result["neighborhood_probability"] = (
        probabilities
    )

    result["is_noise"] = (
        result["neighborhood_id"] == -1
    )

    return result