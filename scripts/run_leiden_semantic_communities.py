"""
Finalize semantic book communities using the frozen semantic graph.

Production Model C configuration
--------------------------------
Embedding model:
    all-MiniLM-L6-v2

Semantic representation:
    Title + Author + Description + curated Open Library Subjects

Graph:
    Weighted, symmetrized thresholded k-nearest-neighbor graph

Graph parameters:
    K = 20
    Minimum cosine similarity = 0.40
    Edge weight = cosine similarity

Community detection:
    Leiden
    RBConfigurationVertexPartition
    Resolution = 3.0
    Seed = 42

This script intentionally does NOT perform parameter tuning.
Parameter selection was completed during the semantic community
detection experiments and is documented separately.

Outputs
-------
data/processed/canonical/leiden/
    leiden_assignments.csv
    leiden_community_summary.csv
    leiden_graph_diagnostics.csv
    leiden_edges.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEMANTIC_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_books.csv"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_embeddings.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
)

ASSIGNMENTS_PATH = OUTPUT_DIR / "leiden_assignments.csv"
SUMMARY_PATH = OUTPUT_DIR / "leiden_community_summary.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "leiden_graph_diagnostics.csv"
EDGES_PATH = OUTPUT_DIR / "leiden_edges.csv"


# Frozen Model C parameters.
K = 20
MIN_SIMILARITY = 0.40

LEIDEN_RESOLUTION = 3.0
LEIDEN_SEED = 42


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    """
    Validate and normalize the embedding matrix.

    The saved semantic embeddings are expected to already be normalized,
    but normalization is repeated here defensively so cosine similarity
    calculations are explicit and deterministic.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2-dimensional embedding matrix. "
            f"Received shape {embeddings.shape}."
        )

    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            "Embedding row count does not match semantic metadata: "
            f"{embeddings.shape[0]} embeddings vs "
            f"{expected_rows} semantic books."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains zero-length vectors.")

    normalized = embeddings / norms[:, None]

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_thresholded_knn_graph(
    embeddings: np.ndarray,
    k: int,
    min_similarity: float,
) -> tuple[ig.Graph, pd.DataFrame]:
    """
    Build a weighted, symmetrized thresholded kNN graph.

    Each book considers its K nearest neighbors.

    An edge is retained only when:

        cosine_similarity >= min_similarity

    The graph is then symmetrized:

        A -> B
        B -> A

    become one undirected weighted edge.

    If both directions exist, the maximum similarity is retained.

    This deliberately allows variable degree, including zero-degree
    books. Books are not forced into a community.
    """

    n_books = embeddings.shape[0]

    if n_books < 2:
        raise ValueError("At least two books are required.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    k = min(k, n_books - 1)

    print()
    print("Building cosine similarity matrix...")
    similarity_matrix = embeddings @ embeddings.T

    # Numerical protection.
    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    )

    # Never allow self-neighbors.
    np.fill_diagonal(similarity_matrix, -np.inf)

    print(f"Books: {n_books:,}")
    print(f"K: {k}")
    print(f"Minimum similarity: {min_similarity:.2f}")

    # ------------------------------------------------------------------
    # Find K nearest neighbors for every book.
    #
    # argpartition avoids a full sort across every row.
    # ------------------------------------------------------------------

    neighbor_indices = np.argpartition(
        similarity_matrix,
        -k,
        axis=1,
    )[:, -k:]

    rows = np.repeat(
        np.arange(n_books),
        k,
    )

    cols = neighbor_indices.reshape(-1)

    similarities = similarity_matrix[
        rows,
        cols,
    ]

    candidate_edges = pd.DataFrame(
        {
            "source": rows,
            "target": cols,
            "similarity": similarities,
        }
    )

    candidate_edges = candidate_edges[
        candidate_edges["similarity"] >= min_similarity
    ].copy()

    print(
        f"Directed candidate edges retained: "
        f"{len(candidate_edges):,}"
    )

    # ------------------------------------------------------------------
    # Symmetrize the graph.
    #
    # A -> B and B -> A represent the same undirected relationship.
    # Canonicalize the node pair so duplicates can be grouped.
    # ------------------------------------------------------------------

    candidate_edges["node_a"] = candidate_edges[
        ["source", "target"]
    ].min(axis=1)

    candidate_edges["node_b"] = candidate_edges[
        ["source", "target"]
    ].max(axis=1)

    edges = (
        candidate_edges
        .groupby(
            ["node_a", "node_b"],
            as_index=False,
        )["similarity"]
        .max()
        .rename(
            columns={
                "node_a": "source",
                "node_b": "target",
            }
        )
    )

    edges = edges[
        edges["source"] != edges["target"]
    ].copy()

    edges = edges.sort_values(
        ["source", "target"]
    ).reset_index(drop=True)

    print(
        f"Undirected graph edges: "
        f"{len(edges):,}"
    )

    # ------------------------------------------------------------------
    # Build igraph.
    # ------------------------------------------------------------------

    graph = ig.Graph(
        n=n_books,
        edges=list(
            zip(
                edges["source"].astype(int),
                edges["target"].astype(int),
            )
        ),
        directed=False,
    )

    graph.es["weight"] = (
        edges["similarity"]
        .astype(float)
        .tolist()
    )

    return graph, edges


# ---------------------------------------------------------------------------
# Graph diagnostics
# ---------------------------------------------------------------------------


def calculate_graph_diagnostics(
    graph: ig.Graph,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate structural diagnostics for the final graph."""

    n_nodes = graph.vcount()
    n_edges = graph.ecount()

    possible_edges = n_nodes * (n_nodes - 1) / 2

    density = (
        n_edges / possible_edges
        if possible_edges > 0
        else 0.0
    )

    degrees = np.asarray(
        graph.degree(),
        dtype=np.int64,
    )

    isolated_count = int(
        np.sum(degrees == 0)
    )

    components = graph.components()

    component_sizes = np.asarray(
        components.sizes(),
        dtype=np.int64,
    )

    largest_component = (
        int(component_sizes.max())
        if len(component_sizes)
        else 0
    )

    largest_component_pct = (
        largest_component / n_nodes
        if n_nodes
        else 0.0
    )

    similarities = edges["similarity"].to_numpy(
        dtype=float
    )

    diagnostics = {
        "graph": "thresholded_knn",
        "k": K,
        "min_similarity": MIN_SIMILARITY,
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "isolated_nodes": isolated_count,
        "isolated_pct": (
            isolated_count / n_nodes
            if n_nodes
            else 0.0
        ),
        "components": len(component_sizes),
        "largest_component": largest_component,
        "largest_component_pct": largest_component_pct,
        "mean_similarity": (
            float(np.mean(similarities))
            if len(similarities)
            else np.nan
        ),
        "median_similarity": (
            float(np.median(similarities))
            if len(similarities)
            else np.nan
        ),
        "min_similarity": (
            float(np.min(similarities))
            if len(similarities)
            else np.nan
        ),
        "max_similarity": (
            float(np.max(similarities))
            if len(similarities)
            else np.nan
        ),
        "mean_degree": (
            float(np.mean(degrees))
            if len(degrees)
            else np.nan
        ),
        "median_degree": (
            float(np.median(degrees))
            if len(degrees)
            else np.nan
        ),
        "max_degree": (
            int(np.max(degrees))
            if len(degrees)
            else 0
        ),
    }

    return pd.DataFrame([diagnostics])


# ---------------------------------------------------------------------------
# Leiden community detection
# ---------------------------------------------------------------------------


def run_leiden(
    graph: ig.Graph,
    resolution: float,
    seed: int,
) -> tuple[leidenalg.VertexPartition, np.ndarray]:
    """
    Run Leiden using RBConfigurationVertexPartition.

    Returns:
        partition
        membership array
    """

    if graph.ecount() == 0:
        raise ValueError(
            "Cannot run Leiden on a graph with zero edges."
        )

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    membership = np.asarray(
        partition.membership,
        dtype=np.int64,
    )

    return partition, membership


# ---------------------------------------------------------------------------
# Community metrics
# ---------------------------------------------------------------------------


def calculate_community_summary(
    graph: ig.Graph,
    edges: pd.DataFrame,
    membership: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate community-level structural metrics.

    Internal similarity is calculated only from graph edges whose two
    endpoints belong to the same community.

    Singleton communities therefore have no internal edges and receive
    NaN for internal similarity.
    """

    if len(membership) != graph.vcount():
        raise ValueError(
            "Membership length does not match graph node count."
        )

    community_ids = np.unique(membership)

    rows: list[dict] = []

    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    weights = edges["similarity"].to_numpy(dtype=float)

    same_community = (
        membership[source]
        == membership[target]
    )

    internal_edges = edges.loc[
        same_community
    ].copy()

    internal_edges["community_id"] = membership[
        internal_edges["source"].to_numpy(
            dtype=np.int64
        )
    ]

    for community_id in community_ids:

        node_mask = membership == community_id

        book_count = int(
            np.sum(node_mask)
        )

        community_internal = internal_edges[
            internal_edges["community_id"]
            == community_id
        ]

        internal_similarities = (
            community_internal["similarity"]
            .to_numpy(dtype=float)
        )

        rows.append(
            {
                "community_id": int(community_id),
                "book_count": book_count,
                "internal_edge_count": int(
                    len(internal_similarities)
                ),
                "mean_internal_similarity": (
                    float(
                        np.mean(
                            internal_similarities
                        )
                    )
                    if len(internal_similarities)
                    else np.nan
                ),
                "median_internal_similarity": (
                    float(
                        np.median(
                            internal_similarities
                        )
                    )
                    if len(internal_similarities)
                    else np.nan
                ),
                "min_internal_similarity": (
                    float(
                        np.min(
                            internal_similarities
                        )
                    )
                    if len(internal_similarities)
                    else np.nan
                ),
                "max_internal_similarity": (
                    float(
                        np.max(
                            internal_similarities
                        )
                    )
                    if len(internal_similarities)
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "book_count",
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------------
# Assignment artifact
# ---------------------------------------------------------------------------


def build_assignments(
    semantic_books: pd.DataFrame,
    membership: np.ndarray,
    partition: leidenalg.VertexPartition,
) -> pd.DataFrame:
    """Build the final book-to-community assignment artifact."""

    if len(membership) != len(semantic_books):
        raise ValueError(
            "Membership length does not match semantic book count."
        )

    assignments = semantic_books.copy()

    assignments["embedding_row"] = np.arange(
        len(assignments),
        dtype=np.int64,
    )

    assignments["community_id"] = membership

    # Leiden's community membership is a hard assignment.
    #
    # We do not invent a probability here. A future version could add
    # a separate community-strength metric if needed.
    assignments["community_probability"] = 1.0

    # Put the most useful columns first.
    preferred_columns = [
        "canonical_book_id",
        "title",
        "author",
        "embedding_row",
        "community_id",
        "community_probability",
    ]

    remaining_columns = [
        column
        for column in assignments.columns
        if column not in preferred_columns
    ]

    assignments = assignments[
        preferred_columns + remaining_columns
    ]

    return assignments


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def validate_outputs(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    semantic_books: pd.DataFrame,
    graph: ig.Graph,
) -> None:
    """Validate final Model C artifacts before reporting success."""

    require_columns(
        assignments,
        [
            "canonical_book_id",
            "title",
            "author",
            "embedding_row",
            "community_id",
            "community_probability",
        ],
        "Leiden assignments",
    )

    require_columns(
        summary,
        [
            "community_id",
            "book_count",
        ],
        "Leiden community summary",
    )

    if len(assignments) != len(semantic_books):
        raise ValueError(
            "Final assignment count does not match semantic corpus."
        )

    if assignments["canonical_book_id"].duplicated().any():
        raise ValueError(
            "Duplicate canonical book IDs found in final assignments."
        )

    if assignments["community_id"].isna().any():
        raise ValueError(
            "Some books have no community assignment."
        )

    if assignments["community_id"].nunique() != len(summary):
        raise ValueError(
            "Community summary count does not match assignments."
        )

    if assignments["community_id"].nunique() != len(
        np.unique(
            assignments["community_id"]
        )
    ):
        raise ValueError(
            "Unexpected community ID structure."
        )

    # Every graph node must have one assignment.
    if graph.vcount() != len(assignments):
        raise ValueError(
            "Graph node count does not match assignment count."
        )

    # Community sizes should sum to the number of books.
    if summary["book_count"].sum() != len(
        assignments
    ):
        raise ValueError(
            "Community sizes do not sum to corpus size."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print("FINALIZE LEIDEN SEMANTIC COMMUNITIES")
    print("=" * 72)

    print()
    print("Frozen configuration:")
    print(f"  KNN K:                  {K}")
    print(f"  Minimum similarity:     {MIN_SIMILARITY:.2f}")
    print(
        "  Leiden algorithm:       "
        "RBConfigurationVertexPartition"
    )
    print(
        f"  Leiden resolution:      {LEIDEN_RESOLUTION:.2f}"
    )
    print(f"  Leiden seed:            {LEIDEN_SEED}")

    # ------------------------------------------------------------------
    # Load semantic corpus.
    # ------------------------------------------------------------------

    if not SEMANTIC_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Semantic books file not found:\n"
            f"{SEMANTIC_BOOKS_PATH}"
        )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    semantic_books = pd.read_csv(
        SEMANTIC_BOOKS_PATH
    )

    require_columns(
        semantic_books,
        [
            "canonical_book_id",
            "title",
            "author",
        ],
        "Semantic books",
    )

    if semantic_books[
        "canonical_book_id"
    ].duplicated().any():

        raise ValueError(
            "Semantic corpus contains duplicate "
            "canonical_book_id values."
        )

    print()
    print(
        f"Semantic books loaded: "
        f"{len(semantic_books):,}"
    )

    # ------------------------------------------------------------------
    # Load frozen embeddings.
    # ------------------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    embeddings = validate_embeddings(
        embeddings,
        expected_rows=len(semantic_books),
    )

    print(
        f"Embeddings loaded: "
        f"{embeddings.shape[0]:,} × "
        f"{embeddings.shape[1]}"
    )

    # ------------------------------------------------------------------
    # Build final graph.
    # ------------------------------------------------------------------

    graph, edges = build_thresholded_knn_graph(
        embeddings=embeddings,
        k=K,
        min_similarity=MIN_SIMILARITY,
    )

    print()
    print(
        f"Graph nodes: "
        f"{graph.vcount():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.ecount():,}"
    )

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------

    diagnostics = calculate_graph_diagnostics(
        graph=graph,
        edges=edges,
    )

    print()
    print("Graph diagnostics:")
    print(
        diagnostics[
            [
                "nodes",
                "edges",
                "density",
                "isolated_nodes",
                "isolated_pct",
                "components",
                "largest_component",
                "largest_component_pct",
                "mean_similarity",
                "median_similarity",
                "mean_degree",
                "median_degree",
                "max_degree",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Run final Leiden.
    # ------------------------------------------------------------------

    print()
    print("Running final Leiden community detection...")

    partition, membership = run_leiden(
        graph=graph,
        resolution=LEIDEN_RESOLUTION,
        seed=LEIDEN_SEED,
    )

    community_count = int(
        len(np.unique(membership))
    )

    print(
        f"Communities discovered: "
        f"{community_count:,}"
    )

    # ------------------------------------------------------------------
    # Build final artifacts.
    # ------------------------------------------------------------------

    assignments = build_assignments(
        semantic_books=semantic_books,
        membership=membership,
        partition=partition,
    )

    summary = calculate_community_summary(
        graph=graph,
        edges=edges,
        membership=membership,
    )

    validate_outputs(
        assignments=assignments,
        summary=summary,
        semantic_books=semantic_books,
        graph=graph,
    )

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        ASSIGNMENTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_PATH,
        index=False,
    )

    edge_output = edges.copy()

    edge_output["source_book_id"] = (
        semantic_books.iloc[
            edge_output["source"]
        ]["canonical_book_id"]
        .to_numpy()
    )

    edge_output["source_title"] = (
        semantic_books.iloc[
            edge_output["source"]
        ]["title"]
        .to_numpy()
    )

    edge_output["source_author"] = (
        semantic_books.iloc[
            edge_output["source"]
        ]["author"]
        .to_numpy()
    )

    edge_output["target_book_id"] = (
        semantic_books.iloc[
            edge_output["target"]
        ]["canonical_book_id"]
        .to_numpy()
    )

    edge_output["target_title"] = (
        semantic_books.iloc[
            edge_output["target"]
        ]["title"]
        .to_numpy()
    )

    edge_output["target_author"] = (
        semantic_books.iloc[
            edge_output["target"]
        ]["author"]
        .to_numpy()
    )

    edge_output.to_csv(
        EDGES_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL MODEL C ARTIFACTS CREATED")
    print("=" * 72)

    print()
    print(
        f"Books:                 "
        f"{len(assignments):,}"
    )

    print(
        f"Communities:           "
        f"{community_count:,}"
    )

    print(
        f"Largest community:     "
        f"{summary['book_count'].max():,}"
    )

    print(
        f"Median community size:  "
        f"{summary['book_count'].median():.1f}"
    )

    print(
        f"Smallest community:     "
        f"{summary['book_count'].min():,}"
    )

    print(
        f"Isolated books:         "
        f"{int(diagnostics.iloc[0]['isolated_nodes']):,}"
    )

    print()
    print("Top communities by size:")

    display_columns = [
        "community_id",
        "book_count",
        "internal_edge_count",
        "mean_internal_similarity",
        "median_internal_similarity",
    ]

    print(
        summary[
            display_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Outputs:")
    print(f"  {ASSIGNMENTS_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {DIAGNOSTICS_PATH}")
    print(f"  {EDGES_PATH}")

    print()
    print("Model C semantic communities finalized.")
    print(
        "Next step: inspect representative communities "
        "before freezing semantic community interpretation."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("ERROR:")
        print(exc)
        sys.exit(1)