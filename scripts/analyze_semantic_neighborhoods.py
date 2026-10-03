from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.analytics.semantic_neighborhoods import (
    NeighborhoodConfig,
    add_book_identity_to_neighbors,
    build_neighborhood_assignments,
    calculate_parameter_stability,
    discover_semantic_neighborhoods,
    find_nearest_neighbors,
    summarize_neighborhood_solution,
)


CANONICAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
)

EMBEDDINGS_PATH = (
    CANONICAL_DIR
    / "semantic_embeddings.npy"
)

METADATA_PATH = (
    CANONICAL_DIR
    / "semantic_embeddings_metadata.csv"
)

NEIGHBORHOODS_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhoods.csv"
)

NEIGHBOR_SUMMARY_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhood_summary.csv"
)

ASSIGNMENTS_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhood_assignments.csv"
)

TUNING_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhood_tuning.csv"
)


K_NEIGHBORS = 20

MIN_CLUSTER_SIZE = 8

MIN_SAMPLES = 3

GRANULARITY_VALUES = (
    5,
    8,
    12,
    16,
)


def main() -> None:

    print(
        "Loading semantic embedding artifacts..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding and metadata row counts "
            "do not match."
        )

    metadata = (
        metadata
        .sort_values("embedding_row")
        .reset_index(drop=True)
    )

    print(
        f"Books: {len(embeddings):,}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    # ---------------------------------------------------------------
    # Existing local semantic neighbor graph
    # ---------------------------------------------------------------

    print()
    print(
        "Building top-20 semantic neighbor graph..."
    )

    neighbors = find_nearest_neighbors(
        embeddings,
        k=K_NEIGHBORS,
    )

    neighbors = add_book_identity_to_neighbors(
        neighbors,
        metadata,
    )

    neighbors.to_csv(
        NEIGHBORHOODS_PATH,
        index=False,
    )

    summary = (
        neighbors.groupby(
            "query_book_id"
        )
        .agg(
            mean_similarity=(
                "similarity",
                "mean",
            ),
            median_similarity=(
                "similarity",
                "median",
            ),
            min_similarity=(
                "similarity",
                "min",
            ),
            max_similarity=(
                "similarity",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        NEIGHBOR_SUMMARY_PATH,
        index=False,
    )

    print(
        f"Generated "
        f"{len(neighbors):,} "
        "book-neighbor relationships."
    )

    print(
        f"Mean similarity: "
        f"{neighbors['similarity'].mean():.4f}"
    )

    print(
        f"Median similarity: "
        f"{neighbors['similarity'].median():.4f}"
    )

    # ---------------------------------------------------------------
    # Density-based semantic neighborhoods
    # ---------------------------------------------------------------

    print()
    print(
        "Discovering density-based semantic "
        "neighborhoods..."
    )

    labels, clusterer = (
        discover_semantic_neighborhoods(
            embeddings,
            NeighborhoodConfig(
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_samples=MIN_SAMPLES,
            ),
        )
    )

    probabilities = getattr(
        clusterer,
        "probabilities_",
        None,
    )

    assignments = (
        build_neighborhood_assignments(
            metadata,
            labels,
            probabilities,
        )
    )

    assignments.to_csv(
        ASSIGNMENTS_PATH,
        index=False,
    )

    solution = (
        summarize_neighborhood_solution(
            embeddings,
            labels,
        )
    )

    print()
    print(
        "Density-based neighborhood summary:"
    )

    for key, value in solution.items():
        print(
            f"  {key}: {value}"
        )

    # ---------------------------------------------------------------
    # Granularity stability
    # ---------------------------------------------------------------

    print()
    print(
        "Evaluating neighborhood granularity..."
    )

    tuning = calculate_parameter_stability(
        embeddings,
        min_cluster_sizes=GRANULARITY_VALUES,
        min_samples=MIN_SAMPLES,
    )

    tuning.to_csv(
        TUNING_PATH,
        index=False,
    )

    print()
    print(
        tuning.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved semantic neighborhood artifacts:"
    )

    print(
        f"  {NEIGHBORHOODS_PATH}"
    )

    print(
        f"  {NEIGHBOR_SUMMARY_PATH}"
    )

    print(
        f"  {ASSIGNMENTS_PATH}"
    )

    print(
        f"  {TUNING_PATH}"
    )


if __name__ == "__main__":
    main()