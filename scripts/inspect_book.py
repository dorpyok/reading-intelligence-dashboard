"""
Inspect the semantic and network intelligence for one book.

This is a read-only diagnostic and content-development tool.
It does not modify or regenerate any ML artifacts.

Current intelligence sources:
    - Semantic book metadata
    - Frozen semantic similarity graph
    - Leiden community assignments
    - Leiden community summaries

Examples
--------
Standard report:

    python scripts/inspect_book.py "House of Earth and Blood"

Instagram-oriented report:

    python scripts/inspect_book.py "House of Earth and Blood" \
        --format instagram

Inspect a specific Leiden resolution:

    python scripts/inspect_book.py "House of Earth and Blood" \
        --resolution 3.0

Show more neighbors:

    python scripts/inspect_book.py "House of Earth and Blood" \
        --top 10

JSON output:

    python scripts/inspect_book.py "House of Earth and Blood" \
        --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEMANTIC_NEIGHBORHOODS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhoods.csv"
)

SEMANTIC_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_books.csv"
)

LEIDEN_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
    / "leiden_assignments.csv"
)

LEIDEN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
    / "leiden_community_summary.csv"
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_RESOLUTION = 3.0
DEFAULT_TOP_N = 5
DEFAULT_COMMUNITY_EXAMPLES = 10

# Fuzzy matches are suggestions only.
# They are NEVER automatically accepted.
FUZZY_SUGGESTION_THRESHOLD = 0.65


# ============================================================================
# Formatting helpers
# ============================================================================


def clean_display_text(value: object) -> str:
    """Return a clean display string."""

    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_text(value: object) -> str:
    """
    Normalize text for matching.

    Punctuation is removed and whitespace is normalized.
    """

    text = clean_display_text(value).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_title(value: object) -> str:
    """
    Normalize a book title while preserving its meaningful words.
    """

    return normalize_text(value)


def strip_series_suffix(title: str) -> str:
    """
    Remove common series/volume suffixes from a title.

    Examples
    --------
    House of Earth and Blood (Crescent City, #1)
        -> House of Earth and Blood

    Red Rising (Red Rising, #1)
        -> Red Rising

    This is intentionally conservative.
    """

    title = clean_display_text(title)

    # Remove parenthetical suffixes containing common series indicators.
    stripped = re.sub(
        r"\s*\((?:[^)]*(?:#\s*\d+|book\s+\d+|volume\s+\d+|vol\.\s*\d+)[^)]*)\)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # Also handle bracketed volume indicators.
    stripped = re.sub(
        r"\s*\[(?:[^\]]*(?:#\s*\d+|book\s+\d+|volume\s+\d+|vol\.\s*\d+)[^\]]*)\]\s*$",
        "",
        stripped,
        flags=re.IGNORECASE,
    )

    return stripped.strip()


def title_without_series_suffix(value: object) -> str:
    """Normalize a title after removing a likely series suffix."""

    return normalize_text(
        strip_series_suffix(
            clean_display_text(value)
        )
    )


def similarity_score(
    query: str,
    candidate: str,
) -> float:
    """Calculate fuzzy title similarity."""

    return SequenceMatcher(
        None,
        normalize_title(query),
        normalize_title(candidate),
    ).ratio()


def truncate_text(
    text: str,
    max_length: int = 350,
) -> str:
    """Create a readable description excerpt."""

    text = clean_display_text(text)

    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(
        " ",
        1,
    )[0]

    return truncated + "..."


# ============================================================================
# Subject parsing
# ============================================================================


def parse_subjects(value: object) -> list[str]:
    """
    Parse the cleaned subject field safely.

    Handles:
        - Python list representations
        - JSON-like lists
        - pipe-delimited strings
        - semicolon-delimited strings
        - comma-delimited strings
        - single values
    """

    text = clean_display_text(value)

    if not text:
        return []

    # Try Python literal parsing first because the project artifacts may
    # contain strings such as:
    #
    # ['fantasy fiction', 'angels', 'murder']
    if text.startswith("[") and text.endswith("]"):

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    clean_display_text(item)
                    for item in parsed
                    if clean_display_text(item)
                ]

        except (ValueError, SyntaxError):
            pass

    if "|" in text:
        values = text.split("|")

    elif ";" in text:
        values = text.split(";")

    elif "," in text:
        values = text.split(",")

    else:
        values = [text]

    return [
        clean_display_text(value)
        for value in values
        if clean_display_text(value)
    ]


# ============================================================================
# File validation and loading
# ============================================================================


def validate_file(path: Path) -> None:
    """Ensure an expected artifact exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired artifact was not found:\n"
            f"{path}\n\n"
            f"Run the relevant pipeline step before using "
            f"inspect_book.py."
        )


def load_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the existing book-intelligence artifacts."""

    paths = [
        SEMANTIC_NEIGHBORHOODS_PATH,
        SEMANTIC_BOOKS_PATH,
        LEIDEN_ASSIGNMENTS_PATH,
        LEIDEN_SUMMARY_PATH,
    ]

    for path in paths:
        validate_file(path)

    neighborhoods = pd.read_csv(
        SEMANTIC_NEIGHBORHOODS_PATH
    )

    semantic_books = pd.read_csv(
        SEMANTIC_BOOKS_PATH
    )

    assignments = pd.read_csv(
        LEIDEN_ASSIGNMENTS_PATH
    )

    community_summary = pd.read_csv(
        LEIDEN_SUMMARY_PATH
    )

    return (
        neighborhoods,
        semantic_books,
        assignments,
        community_summary,
    )


# ============================================================================
# Book matching
# ============================================================================


def find_exact_title_match(
    query: str,
    semantic_books: pd.DataFrame,
) -> tuple[pd.Series | None, str | None]:
    """
    Find an exact or normalized title match.

    Returns
    -------
    row, match_method
    """

    query_clean = clean_display_text(query)

    # ------------------------------------------------------------------------
    # Exact title
    # ------------------------------------------------------------------------

    exact = semantic_books[
        semantic_books["title"]
        .map(clean_display_text)
        == query_clean
    ]

    if not exact.empty:
        return (
            exact.iloc[0],
            "exact title",
        )

    # ------------------------------------------------------------------------
    # Normalized title
    # ------------------------------------------------------------------------

    query_normalized = normalize_title(
        query_clean
    )

    normalized_titles = semantic_books[
        "title"
    ].map(normalize_title)

    normalized = semantic_books[
        normalized_titles == query_normalized
    ]

    if not normalized.empty:
        return (
            normalized.iloc[0],
            "normalized title",
        )

    # ------------------------------------------------------------------------
    # Series suffix removed
    #
    # This is specifically what makes:
    #
    # House of Earth and Blood
    #
    # safely match:
    #
    # House of Earth and Blood (Crescent City, #1)
    # ------------------------------------------------------------------------

    query_base = title_without_series_suffix(
        query_clean
    )

    title_bases = semantic_books[
        "title"
    ].map(title_without_series_suffix)

    series_match = semantic_books[
        title_bases == query_base
    ]

    if not series_match.empty:

        # If multiple editions/series records share the base title,
        # return the first deterministic match.
        return (
            series_match.iloc[0],
            "title with series suffix normalized",
        )

    return None, None


def find_fuzzy_suggestions(
    query: str,
    semantic_books: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find possible fuzzy matches.

    These are suggestions only.
    They are never silently accepted.
    """

    candidates = semantic_books[
        [
            "canonical_book_id",
            "title",
            "author",
        ]
    ].copy()

    candidates["match_score"] = candidates[
        "title"
    ].map(
        lambda title: similarity_score(
            query,
            title,
        )
    )

    candidates = candidates[
        candidates["match_score"]
        >= FUZZY_SUGGESTION_THRESHOLD
    ]

    candidates = candidates.sort_values(
        [
            "match_score",
            "title",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return candidates.head(
        top_n
    ).reset_index(drop=True)


def find_book(
    query: str,
    semantic_books: pd.DataFrame,
) -> tuple[
    pd.Series | None,
    str | None,
    pd.DataFrame,
]:
    """
    Find a book safely.

    Exact/normalized/series-suffix matches are accepted.

    Fuzzy matches are returned separately as suggestions and are
    NEVER automatically substituted.
    """

    book, method = find_exact_title_match(
        query,
        semantic_books,
    )

    if book is not None:
        return (
            book,
            method,
            pd.DataFrame(),
        )

    suggestions = find_fuzzy_suggestions(
        query,
        semantic_books,
    )

    return (
        None,
        None,
        suggestions,
    )


# ============================================================================
# Semantic neighbors
# ============================================================================


def get_neighbors(
    book_id: str,
    neighborhoods: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return the strongest semantic neighbors."""

    rows = neighborhoods[
        neighborhoods["query_book_id"]
        == book_id
    ].copy()

    if rows.empty:
        return pd.DataFrame(
            columns=[
                "neighbor_book_id",
                "neighbor_title",
                "neighbor_author",
                "similarity",
            ]
        )

    rows["similarity"] = pd.to_numeric(
        rows["similarity"],
        errors="coerce",
    )

    rows = rows.dropna(
        subset=["similarity"]
    )

    rows = rows.sort_values(
        "similarity",
        ascending=False,
    )

    rows = rows.drop_duplicates(
        subset=["neighbor_book_id"]
    )

    return rows.head(
        top_n
    ).reset_index(drop=True)[
        [
            "neighbor_book_id",
            "neighbor_title",
            "neighbor_author",
            "similarity",
        ]
    ]


# ============================================================================
# Leiden community
# ============================================================================


def get_book_community(
    book_id: str,
    assignments: pd.DataFrame,
    community_summary: pd.DataFrame,
    resolution: float,
) -> tuple[
    pd.Series | None,
    pd.Series | None,
]:
    """Return a book's Leiden assignment and community summary."""

    resolution_assignments = assignments[
        assignments["resolution"].round(6)
        == round(resolution, 6)
    ].copy()

    if resolution_assignments.empty:

        available = sorted(
            assignments["resolution"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"No Leiden assignments exist for resolution "
            f"{resolution}.\n\n"
            f"Available resolutions: {available}"
        )

    book_match = resolution_assignments[
        resolution_assignments[
            "canonical_book_id"
        ]
        == book_id
    ]

    if book_match.empty:
        return None, None

    assignment = book_match.iloc[0]

    summary_match = community_summary[
        (
            community_summary[
                "resolution"
            ].round(6)
            == round(resolution, 6)
        )
        & (
            community_summary[
                "community_id"
            ]
            == assignment[
                "community_id"
            ]
        )
    ]

    if summary_match.empty:
        return assignment, None

    return (
        assignment,
        summary_match.iloc[0],
    )


def get_community_books(
    book_id: str,
    assignments: pd.DataFrame,
    resolution: float,
    limit: int,
) -> pd.DataFrame:
    """
    Return other books in the same community.

    This is intentionally deterministic. A future version can replace
    alphabetical examples with graph-centrality representatives.
    """

    resolution_assignments = assignments[
        assignments["resolution"].round(6)
        == round(resolution, 6)
    ].copy()

    target = resolution_assignments[
        resolution_assignments[
            "canonical_book_id"
        ]
        == book_id
    ]

    if target.empty:
        return pd.DataFrame()

    community_id = target.iloc[0][
        "community_id"
    ]

    community_books = resolution_assignments[
        resolution_assignments[
            "community_id"
        ]
        == community_id
    ].copy()

    community_books = community_books[
        community_books[
            "canonical_book_id"
        ]
        != book_id
    ]

    return (
        community_books
        .sort_values(
            [
                "title",
                "author",
            ],
            na_position="last",
        )
        .head(limit)
    )


# ============================================================================
# Intelligence construction
# ============================================================================


def build_book_intelligence(
    query: str,
    neighborhoods: pd.DataFrame,
    semantic_books: pd.DataFrame,
    assignments: pd.DataFrame,
    community_summary: pd.DataFrame,
    resolution: float,
    top_n: int,
    community_examples: int,
) -> dict:
    """Build a structured intelligence record."""

    book, match_method, suggestions = find_book(
        query,
        semantic_books,
    )

    # ------------------------------------------------------------------------
    # Book not found
    # ------------------------------------------------------------------------

    if book is None:

        return {
            "found": False,
            "query": query,
            "suggestions": suggestions.to_dict(
                orient="records"
            ),
        }

    book_id = book[
        "canonical_book_id"
    ]

    # ------------------------------------------------------------------------
    # Semantic profile
    # ------------------------------------------------------------------------

    subjects = []

    if "cleaned_subjects" in book.index:
        subjects = parse_subjects(
            book["cleaned_subjects"]
        )

    description = ""

    if "description" in book.index:
        description = clean_display_text(
            book["description"]
        )

    if not description and "semantic_text" in book.index:
        description = clean_display_text(
            book["semantic_text"]
        )

    # ------------------------------------------------------------------------
    # Neighbors
    # ------------------------------------------------------------------------

    neighbors = get_neighbors(
        book_id,
        neighborhoods,
        top_n,
    )

    neighbor_records = []

    for row in neighbors.itertuples(
        index=False
    ):

        neighbor_records.append(
            {
                "book_id": row.neighbor_book_id,
                "title": clean_display_text(
                    row.neighbor_title
                ),
                "author": clean_display_text(
                    row.neighbor_author
                ),
                "similarity": float(
                    row.similarity
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Community
    # ------------------------------------------------------------------------

    assignment, community = get_book_community(
        book_id,
        assignments,
        community_summary,
        resolution,
    )

    community_record = None

    if assignment is not None:

        community_record = {
            "resolution": float(
                resolution
            ),
            "community_id": int(
                assignment[
                    "community_id"
                ]
            ),
        }

        if community is not None:

            community_record.update(
                {
                    "community_size": int(
                        community[
                            "community_size"
                        ]
                    ),
                    "mean_internal_similarity": float(
                        community[
                            "mean_internal_similarity"
                        ]
                    ),
                    "median_internal_similarity": float(
                        community[
                            "median_internal_similarity"
                        ]
                    ),
                    "internal_edge_density": float(
                        community[
                            "internal_edge_density"
                        ]
                    ),
                }
            )

    # ------------------------------------------------------------------------
    # Community examples
    # ------------------------------------------------------------------------

    community_books = get_community_books(
        book_id,
        assignments,
        resolution,
        community_examples,
    )

    community_records = []

    for row in community_books.itertuples(
        index=False
    ):

        community_records.append(
            {
                "book_id": row.canonical_book_id,
                "title": clean_display_text(
                    row.title
                ),
                "author": clean_display_text(
                    row.author
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Output object
    # ------------------------------------------------------------------------

    return {
        "found": True,
        "query": query,
        "book": {
            "canonical_book_id": book_id,
            "title": clean_display_text(
                book["title"]
            ),
            "author": clean_display_text(
                book["author"]
            ),
            "match_method": match_method,
        },
        "semantic_profile": {
            "embedding_model": (
                "all-MiniLM-L6-v2"
            ),
            "embedding_dimensions": 384,
            "semantic_inputs": [
                "Title",
                "Author",
                "Description",
                "Curated Open Library Subjects",
            ],
            "subjects": subjects,
            "description_excerpt": truncate_text(
                description
            ),
        },
        "semantic_neighbors": neighbor_records,
        "leiden_community": community_record,
        "community_examples": community_records,
        "notes": {
            "community_resolution_status": (
                "experimental"
                if resolution == 3.0
                else "experimental"
            ),
            "fuzzy_matching": (
                "suggestions only; "
                "never automatically accepted"
            ),
        },
    }


# ============================================================================
# Not-found output
# ============================================================================


def print_not_found_report(
    intelligence: dict,
) -> None:
    """Print a safe not-found report."""

    print()
    print("=" * 72)
    print("BOOK NOT FOUND")
    print("=" * 72)

    print()
    print(
        f'"{intelligence["query"]}" is not represented '
        f"in the current canonical book corpus."
    )

    print()
    print(
        "The inspector does not automatically substitute "
        "a fuzzy title match."
    )

    suggestions = intelligence[
        "suggestions"
    ]

    if suggestions:

        print()
        print("POSSIBLE MATCHES")
        print("─" * 72)

        for index, row in enumerate(
            suggestions.itertuples(
                index=False
            ),
            start=1,
        ):

            print(
                f"{index}. "
                f"{clean_display_text(row.title)} "
                f"— "
                f"{clean_display_text(row.author)}"
            )

            print(
                f"   title similarity="
                f"{row.match_score:.3f}"
            )

    else:

        print()
        print(
            "No sufficiently similar title "
            "was found."
        )

    print()
    print("=" * 72)


# ============================================================================
# Standard output
# ============================================================================


def print_standard_report(
    intelligence: dict,
) -> None:

    if not intelligence["found"]:

        print_not_found_report(
            intelligence
        )

        return

    book = intelligence["book"]
    profile = intelligence[
        "semantic_profile"
    ]
    neighbors = intelligence[
        "semantic_neighbors"
    ]
    community = intelligence[
        "leiden_community"
    ]
    examples = intelligence[
        "community_examples"
    ]

    print()
    print("=" * 72)
    print("BOOK INTELLIGENCE")
    print("=" * 72)

    print()
    print(
        f"📖 {book['title']}"
    )
    print(
        f"   {book['author']}"
    )

    print()
    print(
        f"Match: {book['match_method']}"
    )

    print()
    print("SEMANTIC PROFILE")
    print("─" * 72)

    print(
        f"Embedding model: "
        f"{profile['embedding_model']}"
    )

    print(
        f"Dimensions: "
        f"{profile['embedding_dimensions']}"
    )

    print(
        "Inputs: "
        + " + ".join(
            profile["semantic_inputs"]
        )
    )

    print()

    if profile["subjects"]:

        print("Subjects:")

        for subject in profile[
            "subjects"
        ][:12]:

            print(
                f"• {subject}"
            )

        if len(
            profile["subjects"]
        ) > 12:

            remaining = (
                len(
                    profile[
                        "subjects"
                    ]
                )
                - 12
            )

            print(
                f"... +{remaining} more"
            )

    else:

        print(
            "Subjects: none available"
        )

    if profile[
        "description_excerpt"
    ]:

        print()
        print(
            "Description excerpt:"
        )

        print(
            profile[
                "description_excerpt"
            ]
        )

    print()
    print("TOP SEMANTIC NEIGHBORS")
    print("─" * 72)

    if not neighbors:

        print(
            "No semantic neighbors found."
        )

    else:

        for index, neighbor in enumerate(
            neighbors,
            start=1,
        ):

            print(
                f"{index}. "
                f"{neighbor['title']} "
                f"— {neighbor['author']}"
            )

            print(
                f"   similarity="
                f"{neighbor['similarity']:.3f}"
            )

    print()
    print("SEMANTIC COMMUNITY")
    print("─" * 72)

    if community is None:

        print(
            "No Leiden community assignment found."
        )

    else:

        print(
            f"Resolution: "
            f"{community['resolution']}"
        )

        print(
            f"Community: "
            f"{community['community_id']}"
        )

        if "community_size" in community:

            print(
                f"Community size: "
                f"{community['community_size']:,}"
            )

            print(
                f"Mean internal similarity: "
                f"{community['mean_internal_similarity']:.3f}"
            )

            print(
                f"Median internal similarity: "
                f"{community['median_internal_similarity']:.3f}"
            )

            print(
                f"Internal edge density: "
                f"{community['internal_edge_density']:.4f}"
            )

            print(
                "\nResolution status: experimental"
            )

    print()
    print("COMMUNITY EXAMPLES")
    print("─" * 72)

    if not examples:

        print(
            "No other books found."
        )

    else:

        for index, example in enumerate(
            examples,
            start=1,
        ):

            print(
                f"{index}. "
                f"{example['title']} "
                f"— {example['author']}"
            )

    print()
    print("=" * 72)


# ============================================================================
# Instagram output
# ============================================================================


def print_instagram_report(
    intelligence: dict,
) -> None:

    if not intelligence["found"]:

        print_not_found_report(
            intelligence
        )

        return

    book = intelligence["book"]
    profile = intelligence[
        "semantic_profile"
    ]
    neighbors = intelligence[
        "semantic_neighbors"
    ]
    community = intelligence[
        "leiden_community"
    ]

    print()
    print("=" * 72)
    print("INSTAGRAM BOOK INTELLIGENCE")
    print("=" * 72)

    print()
    print(
        f"📚 {book['title']}"
    )

    print(
        f"by {book['author']}"
    )

    print()
    print("WHAT THE DATA MODEL SEES")
    print("─" * 72)

    if profile["subjects"]:

        print(
            " • ".join(
                profile["subjects"][:8]
            )
        )

    else:

        print(
            "No curated Open Library subjects."
        )

    print()
    print("TOP SEMANTIC NEIGHBORS")
    print("─" * 72)

    if neighbors:

        for index, neighbor in enumerate(
            neighbors[:5],
            start=1,
        ):

            print(
                f"{index}. "
                f"{neighbor['title']} "
                f"({neighbor['similarity']:.3f})"
            )

    else:

        print(
            "No semantic neighbors found."
        )

    print()

    print(
        "WHERE THIS BOOK LIVES"
    )

    print("─" * 72)

    if community is None:

        print(
            "No Leiden community assignment."
        )

    else:

        print(
            f"Leiden Community "
            f"{community['community_id']}"
        )

        if "community_size" in community:

            print(
                f"{community['community_size']:,} books"
            )

        print(
            f"Resolution: "
            f"{community['resolution']} "
            f"(experimental)"
        )

    print()

    print(
        "The model does not start with predefined "
        "genres. It builds connections between books "
        "based on semantic similarity, then detects "
        "communities in that network."
    )

    print()
    print("=" * 72)


# ============================================================================
# JSON output
# ============================================================================


def print_json_report(
    intelligence: dict,
) -> None:

    print(
        json.dumps(
            intelligence,
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect semantic similarity and Leiden "
            "community intelligence for one book."
        )
    )

    parser.add_argument(
        "book",
        help="Book title to inspect.",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help=(
            "Leiden resolution to inspect. "
            f"Default: {DEFAULT_RESOLUTION}"
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            "Number of semantic neighbors to show. "
            f"Default: {DEFAULT_TOP_N}"
        ),
    )

    parser.add_argument(
        "--community-examples",
        type=int,
        default=DEFAULT_COMMUNITY_EXAMPLES,
        help=(
            "Number of books to show from the community. "
            f"Default: {DEFAULT_COMMUNITY_EXAMPLES}"
        ),
    )

    parser.add_argument(
        "--format",
        choices=[
            "standard",
            "instagram",
            "json",
        ],
        default="standard",
        help="Output format.",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main() -> None:

    args = parse_args()

    if args.top < 1:
        raise ValueError(
            "--top must be at least 1."
        )

    if args.community_examples < 1:
        raise ValueError(
            "--community-examples must be at least 1."
        )

    (
        neighborhoods,
        semantic_books,
        assignments,
        community_summary,
    ) = load_artifacts()

    intelligence = build_book_intelligence(
        query=args.book,
        neighborhoods=neighborhoods,
        semantic_books=semantic_books,
        assignments=assignments,
        community_summary=community_summary,
        resolution=args.resolution,
        top_n=args.top,
        community_examples=args.community_examples,
    )

    if args.format == "instagram":

        print_instagram_report(
            intelligence
        )

    elif args.format == "json":

        print_json_report(
            intelligence
        )

    else:

        print_standard_report(
            intelligence
        )


if __name__ == "__main__":

    try:

        main()

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:

        print(
            f"\nERROR: {exc}"
        )

        sys.exit(1)