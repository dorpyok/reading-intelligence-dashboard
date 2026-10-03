"""
Run Leiden community detection on the frozen semantic similarity graph.

Input
-----
data/processed/canonical/semantic_neighborhoods.csv

The input is the existing top-k semantic similarity graph generated from
the frozen all-MiniLM-L6-v2 embeddings.

Each book is a node.
Each semantic neighbor relationship is a weighted edge.
The directed top-k graph is symmetrized before Leiden community detection.

Outputs
-------
data/processed/canonical/leiden/
    leiden_tuning.csv
    leiden_assignments.csv
    leiden_community_summary.csv

This is an experiment only. It does not modify the existing Model C outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhoods.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
)

TUNING_OUTPUT = OUTPUT_DIR / "leiden_tuning.csv"
ASSIGNMENTS_OUTPUT = OUTPUT_DIR / "leiden_assignments.csv"
COMMUNITY_SUMMARY_OUTPUT = OUTPUT_DIR / "leiden_community_summary.csv"


# Start with a broad range. We will narrow this after seeing the results.
RESOLUTION_VALUES = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
]

# Multiple seeds let us measure whether the solution is stable.
RANDOM_SEEDS = [42, 123, 2026]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = {
    "query_embedding_row",
    "neighbor_embedding_row",
    "rank",
    "similarity",
    "query_book_id",
    "query_title",
    "query_author",
    "neighbor_book_id",
    "neighbor_title",
    "neighbor_author",
}


def validate_input(df: pd.DataFrame) -> None:
    """Validate the expected semantic-neighborhood schema."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "semantic_neighborhoods.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("semantic_neighborhoods.csv is empty.")

    if df["query_book_id"].isna().any():
        raise ValueError("query_book_id contains missing values.")

    if df["neighbor_book_id"].isna().any():
        raise ValueError("neighbor_book_id contains missing values.")

    if df["similarity"].isna().any():
        raise ValueError("similarity contains missing values.")

    if not np.isfinite(df["similarity"].to_numpy(dtype=float)).all():
        raise ValueError("similarity contains non-finite values.")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_undirected_edge_table(
    neighborhoods: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the directed top-k graph into an undirected weighted graph.

    The source file contains relationships such as:

        A -> B
        B -> A

    or potentially only:

        A -> B

    For reciprocal relationships, the edge weight is the mean similarity.
    For one-way relationships, the observed similarity is retained.

    This avoids treating A -> B and B -> A as two independent edges.
    """

    edges = neighborhoods[
        [
            "query_book_id",
            "neighbor_book_id",
            "similarity",
        ]
    ].copy()

    edges["book_a"] = edges[["query_book_id", "neighbor_book_id"]].min(axis=1)
    edges["book_b"] = edges[["query_book_id", "neighbor_book_id"]].max(axis=1)

    # Remove self-loops.
    edges = edges[edges["book_a"] != edges["book_b"]].copy()

    # Symmetrize the graph.
    #
    # If A -> B and B -> A both exist, average the similarities.
    # If only one direction exists, retain that similarity.
    edges = (
        edges.groupby(["book_a", "book_b"], as_index=False)
        .agg(
            similarity=("similarity", "mean"),
            directed_relationships=("similarity", "size"),
        )
    )

    return edges


def build_graph(
    neighborhoods: pd.DataFrame,
) -> tuple[ig.Graph, pd.DataFrame]:
    """
    Build an undirected weighted igraph graph.

    Returns
    -------
    graph
        Undirected weighted igraph graph.

    node_metadata
        One row per graph node.
    """

    node_records = {}

    for row in neighborhoods.itertuples(index=False):
        node_records[row.query_book_id] = {
            "book_id": row.query_book_id,
            "title": row.query_title,
            "author": row.query_author,
        }

        node_records[row.neighbor_book_id] = {
            "book_id": row.neighbor_book_id,
            "title": row.neighbor_title,
            "author": row.neighbor_author,
        }

    node_metadata = pd.DataFrame(node_records.values())

    node_metadata = node_metadata.sort_values(
        "book_id"
    ).reset_index(drop=True)

    node_index = {
        book_id: index
        for index, book_id in enumerate(node_metadata["book_id"])
    }

    edges = build_undirected_edge_table(neighborhoods)

    graph_edges = [
        (
            node_index[row.book_a],
            node_index[row.book_b],
        )
        for row in edges.itertuples(index=False)
    ]

    weights = edges["similarity"].astype(float).tolist()

    graph = ig.Graph(
        n=len(node_metadata),
        edges=graph_edges,
        directed=False,
    )

    graph.es["weight"] = weights

    return graph, node_metadata


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------


def run_leiden(
    graph: ig.Graph,
    resolution: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """
    Run Leiden using the resolution-parameterized modularity formulation.

    RBConfigurationVertexPartition gives us a resolution parameter:
        lower resolution  -> broader communities
        higher resolution -> finer communities
    """

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    labels = np.full(graph.vcount(), -1, dtype=int)

    for community_id, members in enumerate(partition):
        labels[members] = community_id

    return labels, float(partition.modularity)


# ---------------------------------------------------------------------------
# Community metrics
# ---------------------------------------------------------------------------


def calculate_community_metrics(
    graph: ig.Graph,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate structural metrics for every community.

    Metrics are based on the actual semantic graph rather than the
    embeddings directly.
    """

    edge_source = np.asarray(graph.get_edgelist(), dtype=int)
    weights = np.asarray(graph.es["weight"], dtype=float)

    rows = []

    community_ids = sorted(np.unique(labels))

    for community_id in community_ids:
        member_nodes = np.where(labels == community_id)[0]
        member_set = set(member_nodes.tolist())

        size = len(member_nodes)

        internal_mask = np.array(
            [
                source in member_set and target in member_set
                for source, target in edge_source
            ],
            dtype=bool,
        )

        internal_weights = weights[internal_mask]

        internal_edge_count = len(internal_weights)

        if internal_edge_count > 0:
            mean_internal_similarity = float(
                internal_weights.mean()
            )
            median_internal_similarity = float(
                np.median(internal_weights)
            )
        else:
            mean_internal_similarity = np.nan
            median_internal_similarity = np.nan

        possible_edges = size * (size - 1) / 2

        if possible_edges > 0:
            internal_edge_density = (
                internal_edge_count / possible_edges
            )
        else:
            internal_edge_density = np.nan

        rows.append(
            {
                "community_id": community_id,
                "community_size": size,
                "internal_edge_count": internal_edge_count,
                "internal_edge_density": internal_edge_density,
                "mean_internal_similarity": mean_internal_similarity,
                "median_internal_similarity": median_internal_similarity,
            }
        )

    return pd.DataFrame(rows)


def summarize_partition(
    graph: ig.Graph,
    labels: np.ndarray,
    resolution: float,
    seed: int,
    modularity: float,
) -> dict:
    """Return a single row describing a Leiden partition."""

    community_sizes = pd.Series(labels).value_counts()

    community_metrics = calculate_community_metrics(
        graph,
        labels,
    )

    assigned = len(labels)
    number_of_communities = len(community_sizes)

    return {
        "resolution": resolution,
        "seed": seed,
        "nodes": graph.vcount(),
        "edges": graph.ecount(),
        "communities": number_of_communities,
        "assigned_books": assigned,
        "min_community_size": int(community_sizes.min()),
        "median_community_size": float(
            community_sizes.median()
        ),
        "mean_community_size": float(
            community_sizes.mean()
        ),
        "max_community_size": int(community_sizes.max()),
        "singleton_communities": int(
            (community_sizes == 1).sum()
        ),
        "modularity": modularity,
        "mean_internal_similarity": float(
            community_metrics[
                "mean_internal_similarity"
            ].mean()
        ),
        "median_internal_similarity": float(
            community_metrics[
                "median_internal_similarity"
            ].median()
        ),
        "_labels": labels,
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def calculate_stability(
    partition_results: list[dict],
) -> list[dict]:
    """
    Calculate pairwise Adjusted Rand Index across seeds.

    ARI = 1 means identical community assignments.
    ARI near 0 means little agreement beyond chance.
    """

    output = []

    grouped: dict[float, list[dict]] = {}

    for result in partition_results:
        grouped.setdefault(result["resolution"], []).append(result)

    for resolution, results in grouped.items():

        for i in range(len(results)):
            for j in range(i + 1, len(results)):

                ari = adjusted_rand_score(
                    results[i]["_labels"],
                    results[j]["_labels"],
                )

                output.append(
                    {
                        "resolution": resolution,
                        "seed_a": results[i]["seed"],
                        "seed_b": results[j]["seed"],
                        "adjusted_rand_index": float(ari),
                    }
                )

    return output


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print("LEIDEN SEMANTIC COMMUNITY DETECTION")
    print("=" * 72)

    print(f"\nInput: {INPUT_PATH}")

    neighborhoods = pd.read_csv(INPUT_PATH)

    validate_input(neighborhoods)

    print(f"Input relationships: {len(neighborhoods):,}")
    print(
        "Unique query books:",
        neighborhoods["query_book_id"].nunique(),
    )
    print(
        "Unique neighbor books:",
        neighborhoods["neighbor_book_id"].nunique(),
    )

    graph, node_metadata = build_graph(
        neighborhoods
    )

    print("\nGraph")
    print("-" * 72)
    print(f"Nodes: {graph.vcount():,}")
    print(f"Edges: {graph.ecount():,}")
    print(f"Directed input relationships: {len(neighborhoods):,}")

    weights = np.asarray(
        graph.es["weight"],
        dtype=float,
    )

    print(f"Mean edge similarity: {weights.mean():.4f}")
    print(f"Median edge similarity: {np.median(weights):.4f}")
    print(f"Minimum edge similarity: {weights.min():.4f}")
    print(f"Maximum edge similarity: {weights.max():.4f}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_results = []

    print("\nRunning Leiden")
    print("-" * 72)

    for resolution in RESOLUTION_VALUES:

        print(
            f"\nResolution {resolution}"
        )

        for seed in RANDOM_SEEDS:

            labels, modularity = run_leiden(
                graph=graph,
                resolution=resolution,
                seed=seed,
            )

            result = summarize_partition(
                graph=graph,
                labels=labels,
                resolution=resolution,
                seed=seed,
                modularity=modularity,
            )

            partition_results.append(result)

            print(
                f"  seed={seed}: "
                f"{result['communities']} communities | "
                f"median size="
                f"{result['median_community_size']:.1f} | "
                f"modularity="
                f"{result['modularity']:.4f} | "
                f"mean internal similarity="
                f"{result['mean_internal_similarity']:.4f}"
            )

    # -----------------------------------------------------------------------
    # Stability
    # -----------------------------------------------------------------------

    stability = calculate_stability(
        partition_results
    )

    stability_df = pd.DataFrame(stability)

    if not stability_df.empty:
        stability_summary = (
            stability_df
            .groupby("resolution", as_index=False)
            .agg(
                mean_ari=(
                    "adjusted_rand_index",
                    "mean",
                ),
                min_ari=(
                    "adjusted_rand_index",
                    "min",
                ),
                max_ari=(
                    "adjusted_rand_index",
                    "max",
                ),
            )
        )
    else:
        stability_summary = pd.DataFrame(
            columns=[
                "resolution",
                "mean_ari",
                "min_ari",
                "max_ari",
            ]
        )

    # -----------------------------------------------------------------------
    # Tuning table
    # -----------------------------------------------------------------------

    tuning_rows = []

    for result in partition_results:

        tuning_rows.append(
            {
                key: value
                for key, value in result.items()
                if key != "_labels"
            }
        )

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df = tuning_df.merge(
        stability_summary,
        on="resolution",
        how="left",
    )

    tuning_df.to_csv(
        TUNING_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Choose a representative partition for each resolution.
    #
    # We use seed 42 as the reproducible representative.
    # This is NOT the final selected resolution.
    # -----------------------------------------------------------------------

    assignment_rows = []
    community_summary_rows = []

    for resolution in RESOLUTION_VALUES:

        representative = next(
            result
            for result in partition_results
            if result["resolution"] == resolution
            and result["seed"] == 42
        )

        labels = representative["_labels"]

        community_metrics = calculate_community_metrics(
            graph,
            labels,
        )

        community_metrics.insert(
            0,
            "resolution",
            resolution,
        )

        community_summary_rows.append(
            community_metrics
        )

        for node_index, community_id in enumerate(labels):

            book = node_metadata.iloc[node_index]

            assignment_rows.append(
                {
                    "resolution": resolution,
                    "embedding_row": node_index,
                    "canonical_book_id": book["book_id"],
                    "title": book["title"],
                    "author": book["author"],
                    "community_id": int(community_id),
                }
            )

    assignments_df = pd.DataFrame(
        assignment_rows
    )

    assignments_df.to_csv(
        ASSIGNMENTS_OUTPUT,
        index=False,
    )

    community_summary_df = pd.concat(
        community_summary_rows,
        ignore_index=True,
    )

    community_summary_df.to_csv(
        COMMUNITY_SUMMARY_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Final console summary
    # -----------------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("LEIDEN TUNING SUMMARY")
    print("=" * 72)

    display_columns = [
        "resolution",
        "communities",
        "min_community_size",
        "median_community_size",
        "max_community_size",
        "modularity",
        "mean_internal_similarity",
        "mean_ari",
    ]

    summary = (
        tuning_df[
            tuning_df["seed"] == 42
        ][display_columns]
        .sort_values("resolution")
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOutput files")
    print("-" * 72)
    print(TUNING_OUTPUT)
    print(ASSIGNMENTS_OUTPUT)
    print(COMMUNITY_SUMMARY_OUTPUT)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()