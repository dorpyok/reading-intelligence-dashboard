"""
Inspect books that are isolated from the thresholded semantic graph.

Purpose
-------
Investigate books that have no retained edges under:

    K = 20
    cosine similarity >= 0.40

This is a diagnostic tool only.

It does NOT:
    - modify the graph
    - change embeddings
    - tune thresholds
    - assign books to communities
    - select a production configuration

The goal is to understand whether isolated books are:

    1. genuine semantic outliers,
    2. just below the similarity threshold,
    3. poorly enriched,
    4. or affected by another data-quality issue.

Usage
-----
Inspect all isolated books:

    python scripts/inspect_semantic_outliers.py

Inspect a specific book:

    python scripts/inspect_semantic_outliers.py --book "Book Title"

Change the number of nearest neighbors displayed:

    python scripts/inspect_semantic_outliers.py --top 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"
LEIDEN_DIR = CANONICAL_DIR / "leiden"

EMBEDDINGS_PATH = (
    CANONICAL_DIR / "semantic_embeddings.npy"
)

METADATA_PATH = (
    CANONICAL_DIR / "semantic_embeddings_metadata.csv"
)

GRAPH_PATH = (
    LEIDEN_DIR
    / "leiden_graphs"
    / "knn_k20_threshold040_edges.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_K = 20

TARGET_THRESHOLD = 0.40

INSPECTION_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


# ============================================================================
# LOADING
# ============================================================================


def load_metadata() -> pd.DataFrame:
    """
    Load semantic metadata.
    """
    print(
        f"Loading metadata:\n  {METADATA_PATH}"
    )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    required_columns = [
        "canonical_book_id",
        "title",
        "author",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"  Books loaded: {len(metadata):,}"
    )

    return metadata


def load_embeddings() -> np.ndarray:
    """
    Load the frozen semantic embedding matrix.

    The saved embeddings are normalized again defensively so that
    the dot product is cosine similarity.
    """
    print(
        f"Loading embeddings:\n  {EMBEDDINGS_PATH}"
    )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n{EMBEDDINGS_PATH}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected a 2-dimensional embedding matrix. "
            f"Received shape: {embeddings.shape}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "Embedding matrix contains non-finite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        zero_count = int(
            np.sum(norms == 0)
        )

        raise ValueError(
            f"Found {zero_count} zero-length embeddings."
        )

    normalized = (
        embeddings / norms
    ).astype(
        np.float32
    )

    print(
        f"  Shape: {normalized.shape}"
    )

    print(
        f"  Finite: {np.isfinite(normalized).all()}"
    )

    return normalized


def load_threshold_graph() -> pd.DataFrame:
    """
    Load the K=20 / threshold=.40 graph.

    This graph is only used to identify which books have zero
    retained relationships.
    """
    print(
        f"Loading thresholded graph:\n  {GRAPH_PATH}"
    )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Thresholded graph not found:\n"
            f"{GRAPH_PATH}\n\n"
            "Run the Leiden graph experiment first."
        )

    graph_edges = pd.read_csv(
        GRAPH_PATH
    )

    required_columns = [
        "source_book_id",
        "target_book_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in graph_edges.columns
    ]

    if missing_columns:
        raise ValueError(
            "Graph is missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"  Edges loaded: {len(graph_edges):,}"
    )

    return graph_edges


# ============================================================================
# IDENTIFY ISOLATED BOOKS
# ============================================================================


def identify_isolated_books(
    metadata: pd.DataFrame,
    graph_edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify books that have zero edges in the thresholded graph.
    """
    connected_ids = set(
        graph_edges[
            "source_book_id"
        ]
    ) | set(
        graph_edges[
            "target_book_id"
        ]
    )

    isolated = metadata[
        ~metadata[
            "canonical_book_id"
        ].isin(
            connected_ids
        )
    ].copy()

    isolated = isolated.sort_values(
        ["title", "author"],
        na_position="last",
    )

    return isolated


# ============================================================================
# METADATA HELPERS
# ============================================================================


def get_metadata_value(
    row: pd.Series,
    column: str,
    default: str = "",
) -> str:
    """
    Safely retrieve a metadata value.
    """
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return str(value)


def print_book_header(
    book: pd.Series,
    position: int,
    total: int,
) -> None:
    """
    Print identifying information for a book.
    """
    title = get_metadata_value(
        book,
        "title",
        "[Untitled]",
    )

    author = get_metadata_value(
        book,
        "author",
        "[Unknown author]",
    )

    book_id = get_metadata_value(
        book,
        "canonical_book_id",
        "[Unknown ID]",
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"ISOLATED BOOK {position}/{total}"
    )

    print(
        "=" * 100
    )

    print(
        f"Title:      {title}"
    )

    print(
        f"Author:     {author}"
    )

    print(
        f"Book ID:    {book_id}"
    )

    if "cleaned_subjects" in book.index:
        subjects = get_metadata_value(
            book,
            "cleaned_subjects",
        )

        print(
            f"Subjects:   "
            f"{subjects or '[none]'}"
        )

    if "subject_count" in book.index:
        subject_count = get_metadata_value(
            book,
            "subject_count",
        )

        print(
            f"Subject count: {subject_count}"
        )

    if "semantic_text" in book.index:
        semantic_text = get_metadata_value(
            book,
            "semantic_text",
        )

        word_count = (
            len(
                semantic_text.split()
            )
            if semantic_text
            else 0
        )

        print(
            f"Semantic text length: "
            f"{word_count:,} words"
        )


# ============================================================================
# NEAREST-NEIGHBOR ANALYSIS
# ============================================================================


def calculate_top_neighbors(
    embeddings: np.ndarray,
    book_index: int,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the strongest semantic neighbors for one book.

    Returns:
        neighbor_indices
        similarities
    """
    query_embedding = embeddings[
        book_index
    ]

    similarities = (
        embeddings @ query_embedding
    )

    # Remove self-similarity.
    similarities[
        book_index
    ] = -np.inf

    number_available = (
        len(similarities) - 1
    )

    actual_top_n = min(
        top_n,
        number_available,
    )

    candidate_indices = np.argpartition(
        similarities,
        -actual_top_n,
    )[-actual_top_n:]

    candidate_indices = (
        candidate_indices[
            np.argsort(
                -similarities[
                    candidate_indices
                ]
            )
        ]
    )

    candidate_similarities = (
        similarities[
            candidate_indices
        ]
    )

    return (
        candidate_indices,
        candidate_similarities,
    )


def print_neighbor_analysis(
    book: pd.Series,
    book_index: int,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    top_n: int,
) -> dict:
    """
    Print nearest-neighbor analysis and return a consistent
    analysis dictionary.
    """
    (
        neighbor_indices,
        similarities,
    ) = calculate_top_neighbors(
        embeddings,
        book_index,
        top_n,
    )

    print()
    print(
        f"TOP {top_n} SEMANTIC NEIGHBORS"
    )

    print(
        "-" * 100
    )

    print(
        f"{'Rank':<6}"
        f"{'Similarity':<12}"
        f"{'>=.40':<8}"
        f"{'>=.45':<8}"
        f"{'>=.50':<8}"
        f"{'>=.55':<8}"
        f"{'>=.60':<8}"
        f"Title"
    )

    print(
        "-" * 100
    )

    for rank, (
        neighbor_index,
        similarity,
    ) in enumerate(
        zip(
            neighbor_indices,
            similarities,
        ),
        start=1,
    ):
        neighbor = metadata.iloc[
            neighbor_index
        ]

        title = get_metadata_value(
            neighbor,
            "title",
            "[Untitled]",
        )

        flags = [
            (
                "YES"
                if similarity >= threshold
                else "NO"
            )
            for threshold in INSPECTION_THRESHOLDS
        ]

        print(
            f"{rank:<6}"
            f"{similarity:<12.4f}"
            f"{flags[0]:<8}"
            f"{flags[1]:<8}"
            f"{flags[2]:<8}"
            f"{flags[3]:<8}"
            f"{flags[4]:<8}"
            f"{title}"
        )

    strongest_similarity = float(
        similarities[0]
    )

    print()

    print(
        "Strongest semantic similarity: "
        f"{strongest_similarity:.4f}"
    )

    threshold_counts = {}

    for threshold in INSPECTION_THRESHOLDS:
        count = int(
            np.sum(
                similarities >= threshold
            )
        )

        threshold_counts[
            threshold
        ] = count

        print(
            f"Neighbors >= {threshold:.2f}: "
            f"{count}"
        )

    return {
        "strongest_similarity": (
            strongest_similarity
        ),
        "threshold_counts": (
            threshold_counts
        ),
    }


# ============================================================================
# INTERPRETATION
# ============================================================================


def print_interpretation(
    analysis: dict,
) -> None:
    """
    Provide a simple diagnostic interpretation.

    This function does not modify model behavior.
    """
    strongest_similarity = analysis[
        "strongest_similarity"
    ]

    threshold_counts = analysis[
        "threshold_counts"
    ]

    count_040 = threshold_counts.get(
        0.40,
        0,
    )

    count_045 = threshold_counts.get(
        0.45,
        0,
    )

    count_050 = threshold_counts.get(
        0.50,
        0,
    )

    print()
    print(
        "DIAGNOSTIC INTERPRETATION"
    )

    print(
        "-" * 100
    )

    if strongest_similarity >= 0.40:
        print(
            "Unexpected result: this book has a "
            ">= .40 neighbor and should not be isolated."
        )

    elif strongest_similarity >= 0.38:
        print(
            "Near-threshold singleton: the strongest "
            "semantic relationship is just below .40."
        )

        print(
            "This may be a boundary case rather than "
            "a genuine semantic outlier."
        )

    elif strongest_similarity >= 0.30:
        print(
            "Weakly connected semantic profile: the book "
            "has related books, but none reach .40."
        )

        print(
            "This may represent a genuinely unusual book "
            "or a weaker semantic representation."
        )

    else:
        print(
            "Strong semantic isolation: even the nearest "
            "books have relatively low similarity."
        )

        print(
            "Investigate the book's metadata and semantic "
            "representation before treating this as a graph issue."
        )

    print()

    print(
        f"Relationships at >= .40: {count_040}"
    )

    print(
        f"Relationships at >= .45: {count_045}"
    )

    print(
        f"Relationships at >= .50: {count_050}"
    )

    if count_040 == 0:
        print(
            "At .40: no retained semantic relationships."
        )

    if count_045 == 0:
        print(
            "At .45: no relationships reach the stricter threshold."
        )

    if count_050 == 0:
        print(
            "At .50: no relationships reach the high-confidence threshold."
        )


# ============================================================================
# BOOK LOOKUP
# ============================================================================


def find_book(
    metadata: pd.DataFrame,
    query: str,
) -> pd.Series:
    """
    Find a book by exact normalized title.

    This is intentionally conservative.
    """
    query_normalized = (
        query
        .strip()
        .casefold()
    )

    titles = (
        metadata[
            "title"
        ]
        .fillna("")
        .astype(str)
    )

    exact_matches = titles[
        titles.str.casefold()
        == query_normalized
    ]

    if len(exact_matches) == 1:
        return metadata.loc[
            exact_matches.index[0]
        ]

    normalized_titles = (
        titles
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
        .str.casefold()
    )

    normalized_matches = metadata[
        normalized_titles
        == query_normalized
    ]

    if len(normalized_matches) == 1:
        return normalized_matches.iloc[0]

    if len(normalized_matches) > 1:
        raise ValueError(
            f"Multiple books matched '{query}'. "
            "Use a more specific title."
        )

    raise ValueError(
        f"Book not found: {query}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect books isolated from the "
            "K=20 / similarity=.40 semantic graph."
        )
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help=(
            "Number of nearest neighbors to inspect "
            "(default: 20)."
        ),
    )

    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help=(
            "Inspect one specific book instead of "
            "all isolated books."
        ),
    )

    args = parser.parse_args()

    if args.top < 1:
        raise ValueError(
            "--top must be at least 1."
        )

    print(
        "=" * 100
    )

    print(
        "SEMANTIC OUTLIER INSPECTOR"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Graph configuration: "
        f"K={TARGET_K}, "
        f"similarity >= {TARGET_THRESHOLD:.2f}"
    )

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------

    metadata = load_metadata()

    embeddings = load_embeddings()

    graph_edges = load_threshold_graph()

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Metadata and embeddings have different row counts: "
            f"{len(metadata):,} metadata rows vs "
            f"{len(embeddings):,} embeddings."
        )

    # ------------------------------------------------------------------
    # Specific book mode
    # ------------------------------------------------------------------

    if args.book is not None:
        book = find_book(
            metadata,
            args.book,
        )

        book_index = int(
            book.name
        )

        print()

        print_book_header(
            book,
            1,
            1,
        )

        analysis = print_neighbor_analysis(
            book,
            book_index,
            embeddings,
            metadata,
            args.top,
        )

        print_interpretation(
            analysis
        )

        return

    # ------------------------------------------------------------------
    # Identify isolated books
    # ------------------------------------------------------------------

    isolated = identify_isolated_books(
        metadata,
        graph_edges,
    )

    print()

    print(
        f"Books in corpus:       "
        f"{len(metadata):,}"
    )

    print(
        f"Graph edges:            "
        f"{len(graph_edges):,}"
    )

    print(
        f"Isolated books:         "
        f"{len(isolated):,}"
    )

    if len(isolated) == 0:
        print()

        print(
            "No isolated books found."
        )

        return

    # ------------------------------------------------------------------
    # Inspect each isolated book
    # ------------------------------------------------------------------

    summary_rows = []

    total_isolated = len(
        isolated
    )

    for position, (
        book_index,
        book,
    ) in enumerate(
        isolated.iterrows(),
        start=1,
    ):
        print_book_header(
            book,
            position,
            total_isolated,
        )

        analysis = print_neighbor_analysis(
            book,
            int(book_index),
            embeddings,
            metadata,
            args.top,
        )

        print_interpretation(
            analysis
        )

        threshold_counts = analysis[
            "threshold_counts"
        ]

        summary_rows.append(
            {
                "embedding_row": int(
                    book_index
                ),
                "canonical_book_id": book[
                    "canonical_book_id"
                ],
                "title": book[
                    "title"
                ],
                "author": book[
                    "author"
                ],
                "strongest_similarity": analysis[
                    "strongest_similarity"
                ],
                "neighbors_ge_040": threshold_counts.get(
                    0.40,
                    0,
                ),
                "neighbors_ge_045": threshold_counts.get(
                    0.45,
                    0,
                ),
                "neighbors_ge_050": threshold_counts.get(
                    0.50,
                    0,
                ),
                "neighbors_ge_055": threshold_counts.get(
                    0.55,
                    0,
                ),
                "neighbors_ge_060": threshold_counts.get(
                    0.60,
                    0,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    print()

    print(
        "=" * 100
    )

    print(
        "ISOLATED BOOK SUMMARY"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "title",
        "author",
        "strongest_similarity",
        "neighbors_ge_040",
        "neighbors_ge_045",
        "neighbors_ge_050",
        "neighbors_ge_055",
        "neighbors_ge_060",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "The goal is to determine whether these books "
        "are legitimate semantic outliers or evidence "
        "of a problem with the graph or semantic representation."
    )

    print()

    print(
        "No model parameters were changed."
    )


if __name__ == "__main__":
    main()