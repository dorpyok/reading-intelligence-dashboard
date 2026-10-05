"""
Inspect finalized Leiden semantic communities.

Read-only diagnostic tool.

Usage
-----

Inspect the largest communities:

    python scripts/inspect_leiden_communities.py

Inspect a specific community:

    python scripts/inspect_leiden_communities.py --community 4

Inspect the 10 largest communities:

    python scripts/inspect_leiden_communities.py --top 10

Show more books per community:

    python scripts/inspect_leiden_communities.py --top 10 --books 12

Inspect communities by internal similarity:

    python scripts/inspect_leiden_communities.py --sort cohesion --top 10

Inspect the least cohesive communities:

    python scripts/inspect_leiden_communities.py \
        --sort cohesion \
        --ascending \
        --top 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEIDEN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
)

ASSIGNMENTS_PATH = (
    LEIDEN_DIR
    / "leiden_assignments.csv"
)

SUMMARY_PATH = (
    LEIDEN_DIR
    / "leiden_community_summary.csv"
)

EDGES_PATH = (
    LEIDEN_DIR
    / "leiden_edges.csv"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load finalized Leiden artifacts."""

    required_paths = [
        ASSIGNMENTS_PATH,
        SUMMARY_PATH,
        EDGES_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    edges = pd.read_csv(
        EDGES_PATH
    )

    required_assignment_columns = [
        "canonical_book_id",
        "title",
        "author",
        "embedding_row",
        "community_id",
    ]

    missing = [
        column
        for column in required_assignment_columns
        if column not in assignments.columns
    ]

    if missing:
        raise ValueError(
            "Leiden assignments are missing columns: "
            f"{missing}"
        )

    required_summary_columns = [
        "community_id",
        "book_count",
        "mean_internal_similarity",
        "median_internal_similarity",
    ]

    missing = [
        column
        for column in required_summary_columns
        if column not in summary.columns
    ]

    if missing:
        raise ValueError(
            "Leiden summary is missing columns: "
            f"{missing}"
        )

    required_edge_columns = [
        "source",
        "target",
        "similarity",
    ]

    missing = [
        column
        for column in required_edge_columns
        if column not in edges.columns
    ]

    if missing:
        raise ValueError(
            "Leiden edges are missing columns: "
            f"{missing}"
        )

    return assignments, summary, edges


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect finalized Leiden semantic communities."
        )
    )

    parser.add_argument(
        "--community",
        type=int,
        default=None,
        help="Inspect one specific community.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "Number of communities to display "
            "(default: 10)."
        ),
    )

    parser.add_argument(
        "--books",
        type=int,
        default=8,
        help=(
            "Number of representative books to show "
            "per community (default: 8)."
        ),
    )

    parser.add_argument(
        "--sort",
        choices=[
            "size",
            "cohesion",
        ],
        default="size",
        help=(
            "Sort communities by size or internal "
            "cohesion (default: size)."
        ),
    )

    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Book-level centrality
# ---------------------------------------------------------------------------


def calculate_book_centrality(
    assignments: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate book-level internal semantic connectivity.

    For each book we calculate:

        internal_edge_count
        mean_internal_similarity

    These metrics describe how strongly the book connects to other
    books in its own Leiden community.

    The returned dataframe retains every assignment row.
    """

    assignments_indexed = assignments.set_index(
        "embedding_row"
    )["community_id"]

    source = edges[
        "source"
    ].to_numpy(
        dtype=np.int64
    )

    target = edges[
        "target"
    ].to_numpy(
        dtype=np.int64
    )

    source_community = (
        assignments_indexed
        .reindex(source)
        .to_numpy()
    )

    target_community = (
        assignments_indexed
        .reindex(target)
        .to_numpy()
    )

    internal_mask = (
        source_community
        == target_community
    )

    internal_edges = edges.loc[
        internal_mask
    ].copy()

    # Each undirected edge contributes to both endpoints.
    source_edges = internal_edges[
        [
            "source",
            "similarity",
        ]
    ].rename(
        columns={
            "source": "embedding_row"
        }
    )

    target_edges = internal_edges[
        [
            "target",
            "similarity",
        ]
    ].rename(
        columns={
            "target": "embedding_row"
        }
    )

    book_edges = pd.concat(
        [
            source_edges,
            target_edges,
        ],
        ignore_index=True,
    )

    centrality = (
        book_edges
        .groupby(
            "embedding_row",
            as_index=False,
        )
        .agg(
            internal_edge_count=(
                "similarity",
                "size",
            ),
            mean_internal_similarity=(
                "similarity",
                "mean",
            ),
        )
    )

    result = assignments.merge(
        centrality,
        on="embedding_row",
        how="left",
    )

    result[
        "internal_edge_count"
    ] = (
        result[
            "internal_edge_count"
        ]
        .fillna(0)
        .astype(int)
    )

    return result


# ---------------------------------------------------------------------------
# Representative books
# ---------------------------------------------------------------------------


def get_representative_books(
    community_books: pd.DataFrame,
    number_of_books: int,
) -> pd.DataFrame:
    """
    Select representative books from a community.

    Books with strong internal connectivity are preferred.

    This function expects book-level centrality metrics to already
    exist in community_books.
    """

    required_columns = [
        "title",
        "author",
        "mean_internal_similarity",
        "internal_edge_count",
    ]

    missing = [
        column
        for column in required_columns
        if column not in community_books.columns
    ]

    if missing:
        raise ValueError(
            "Community book data is missing "
            f"centrality columns: {missing}"
        )

    ranked = community_books.sort_values(
        [
            "mean_internal_similarity",
            "internal_edge_count",
            "title",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    )

    return ranked.head(
        number_of_books
    )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_community_header(
    community: pd.Series,
) -> None:
    """Print community-level information."""

    community_id = int(
        community["community_id"]
    )

    book_count = int(
        community["book_count"]
    )

    mean_similarity = community[
        "mean_internal_similarity"
    ]

    median_similarity = community[
        "median_internal_similarity"
    ]

    internal_edges = int(
        community["internal_edge_count"]
    )

    print()
    print("=" * 72)
    print(
        f"COMMUNITY {community_id}"
    )
    print("=" * 72)

    print(
        f"Books:                    "
        f"{book_count:,}"
    )

    print(
        f"Internal edges:           "
        f"{internal_edges:,}"
    )

    if pd.notna(mean_similarity):
        print(
            f"Mean internal similarity: "
            f"{mean_similarity:.4f}"
        )
    else:
        print(
            "Mean internal similarity: N/A"
        )

    if pd.notna(median_similarity):
        print(
            f"Median internal similarity:"
            f" {median_similarity:.4f}"
        )
    else:
        print(
            "Median internal similarity: N/A"
        )


def print_representative_books(
    books: pd.DataFrame,
) -> None:
    """Print representative books."""

    print()
    print("Representative books:")

    if books.empty:
        print("  No books found.")
        return

    for position, (_, book) in enumerate(
        books.iterrows(),
        start=1,
    ):

        title = str(
            book["title"]
        )

        author = str(
            book["author"]
        )

        similarity = book[
            "mean_internal_similarity"
        ]

        edge_count = int(
            book["internal_edge_count"]
        )

        if pd.notna(similarity):
            similarity_text = (
                f"{similarity:.3f}"
            )
        else:
            similarity_text = "n/a"

        print(
            f"  {position:>2}. "
            f"{title} — {author}"
        )

        print(
            f"      internal mean similarity: "
            f"{similarity_text} | "
            f"internal connections: "
            f"{edge_count}"
        )


# ---------------------------------------------------------------------------
# Single community
# ---------------------------------------------------------------------------


def inspect_community(
    community_id: int,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    books_per_community: int,
) -> None:
    """Inspect one community."""

    matches = summary[
        summary["community_id"]
        == community_id
    ]

    if matches.empty:
        available = sorted(
            summary[
                "community_id"
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Community {community_id} was not found.\n"
            f"Available IDs: {available}"
        )

    community = matches.iloc[0]

    print_community_header(
        community
    )

    # IMPORTANT:
    # Centrality must already be present before filtering.
    community_books = assignments[
        assignments["community_id"]
        == community_id
    ].copy()

    representatives = (
        get_representative_books(
            community_books,
            books_per_community,
        )
    )

    print_representative_books(
        representatives
    )

    print()
    print(
        "Interpretation note:"
    )
    print(
        "These are central/representative books "
        "according to graph structure."
    )
    print(
        "They are not automatically a genre label "
        "or human-defined category."
    )


# ---------------------------------------------------------------------------
# Ranked communities
# ---------------------------------------------------------------------------


def inspect_ranked_communities(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    sort_by: str,
    ascending: bool,
    top_n: int,
    books_per_community: int,
) -> None:
    """Inspect ranked communities."""

    if sort_by == "size":
        sort_columns = [
            "book_count",
            "mean_internal_similarity",
        ]
    else:
        sort_columns = [
            "mean_internal_similarity",
            "book_count",
        ]

    ranked = summary.sort_values(
        sort_columns,
        ascending=[
            ascending,
            ascending,
        ],
        na_position="last",
    )

    selected = ranked.head(
        top_n
    )

    print()
    print("=" * 72)
    print(
        f"TOP {len(selected)} COMMUNITIES "
        f"BY {sort_by.upper()}"
    )
    print("=" * 72)

    for _, community in selected.iterrows():

        community_id = int(
            community["community_id"]
        )

        print_community_header(
            community
        )

        community_books = assignments[
            assignments["community_id"]
            == community_id
        ].copy()

        representatives = (
            get_representative_books(
                community_books,
                books_per_community,
            )
        )

        print_representative_books(
            representatives
        )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def print_overall_diagnostics(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print overall Model C diagnostics."""

    community_sizes = (
        summary[
            "book_count"
        ]
        .astype(int)
    )

    singleton_count = int(
        np.sum(
            community_sizes == 1
        )
    )

    small_count = int(
        np.sum(
            community_sizes <= 5
        )
    )

    print()
    print("=" * 72)
    print("LEIDEN COMMUNITY OVERVIEW")
    print("=" * 72)

    print(
        f"Books:                    "
        f"{len(assignments):,}"
    )

    print(
        f"Communities:              "
        f"{len(summary):,}"
    )

    print(
        f"Largest community:        "
        f"{community_sizes.max():,}"
    )

    print(
        f"Median community size:    "
        f"{community_sizes.median():.1f}"
    )

    print(
        f"Smallest community:       "
        f"{community_sizes.min():,}"
    )

    print(
        f"Singleton communities:    "
        f"{singleton_count:,}"
    )

    print(
        f"Communities <= 5 books:   "
        f"{small_count:,}"
    )

    print(
        f"Mean community cohesion:  "
        f"{summary['mean_internal_similarity'].mean():.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    args = parse_args()

    print("=" * 72)
    print("LEIDEN SEMANTIC COMMUNITY INSPECTOR")
    print("=" * 72)

    assignments, summary, edges = load_artifacts()

    print_overall_diagnostics(
        assignments,
        summary,
    )

    # ---------------------------------------------------------------
    # Calculate book-level centrality ONCE before any community
    # filtering.
    # ---------------------------------------------------------------

    assignments = calculate_book_centrality(
        assignments,
        edges,
    )

    # ---------------------------------------------------------------
    # Inspect one community.
    # ---------------------------------------------------------------

    if args.community is not None:

        inspect_community(
            community_id=args.community,
            assignments=assignments,
            summary=summary,
            books_per_community=args.books,
        )

        return

    # ---------------------------------------------------------------
    # Inspect ranked communities.
    # ---------------------------------------------------------------

    inspect_ranked_communities(
        assignments=assignments,
        summary=summary,
        sort_by=args.sort,
        ascending=args.ascending,
        top_n=args.top,
        books_per_community=args.books,
    )

    print()
    print("=" * 72)
    print("INSPECTION COMPLETE")
    print("=" * 72)

    print()
    print("Next question:")

    print(
        "Do the representative books within each "
        "community form a recognizable semantic pattern?"
    )

    print()
    print(
        "Remember: a community is a network-derived "
        "semantic neighborhood, not a predefined genre."
    )


if __name__ == "__main__":
    main()