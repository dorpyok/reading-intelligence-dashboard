from __future__ import annotations

from pathlib import Path
import ast
import json
import sys

import pandas as pd


# Add the repository root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.metadata_normalization import (
    build_semantic_metadata,
    clean_list,
    clean_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "canonical_books.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_books.csv"
)


REQUIRED_COLUMNS = [
    "canonical_book_id",
    "title",
    "author",
    "description",
    "subjects",
    "subject_people",
    "subject_places",
    "subject_times",
]


def parse_list_field(value: object) -> list[str]:
    """
    Parse an Open Library list-like field from the canonical CSV.

    Canonical CSV values may appear as:
    - Python-style lists
    - JSON lists
    - plain strings
    - missing values
    """

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        return clean_list(value)

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []

    # Try Python literal syntax first.
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return clean_list(parsed)

    except (ValueError, SyntaxError):
        pass

    # Try JSON syntax.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return clean_list(parsed)

    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    # Fall back to treating the entire value as one item.
    return clean_list([text])


def prepare_canonical_books(
    books: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare canonical book metadata for semantic processing.

    List-like Open Library fields are converted into actual Python
    lists before semantic text is generated.
    """

    books = books.copy()

    list_columns = [
        "subjects",
        "subject_people",
        "subject_places",
        "subject_times",
    ]

    for column in list_columns:
        books[column] = books[column].apply(parse_list_field)

    return books


def count_populated(
    books: pd.DataFrame,
    column: str,
) -> int:
    """Count rows with meaningful values in a text column."""

    return int(
        books[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )


def count_list_metadata(
    books: pd.DataFrame,
    column: str,
) -> int:
    """Count books with at least one value in a list-like metadata field."""

    return int(
        books[column]
        .apply(lambda value: len(value) > 0)
        .sum()
    )


def print_section(title: str) -> None:
    """Print a readable section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print_section("SEMANTIC METADATA VALIDATION")

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical books file not found: {INPUT_PATH}"
        )

    books = pd.read_csv(INPUT_PATH)

    print_section("1. INPUT VALIDATION")

    print(f"Canonical books loaded: {len(books):,}")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in books.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_ids = books["canonical_book_id"].duplicated().sum()

    print(f"Required columns: PASS")
    print(f"Duplicate canonical IDs: {duplicate_ids}")

    if duplicate_ids:
        raise ValueError(
            "Canonical books contains duplicate canonical_book_id values."
        )

    print_section("2. PREPARE METADATA")

    books = prepare_canonical_books(books)

    print("Parsed Open Library list fields:")
    print("  subjects")
    print("  subject_people")
    print("  subject_places")
    print("  subject_times")

    print_section("3. SOURCE METADATA COVERAGE")

    print(
        f"Titles populated:       "
        f"{count_populated(books, 'title'):,} / {len(books):,}"
    )

    print(
        f"Authors populated:      "
        f"{count_populated(books, 'author'):,} / {len(books):,}"
    )

    print(
        f"Descriptions populated: "
        f"{count_populated(books, 'description'):,} / {len(books):,}"
    )

    print(
        f"Subjects populated:     "
        f"{count_list_metadata(books, 'subjects'):,} / {len(books):,}"
    )

    print(
        f"People populated:       "
        f"{count_list_metadata(books, 'subject_people'):,} / {len(books):,}"
    )

    print(
        f"Places populated:       "
        f"{count_list_metadata(books, 'subject_places'):,} / {len(books):,}"
    )

    print(
        f"Times populated:        "
        f"{count_list_metadata(books, 'subject_times'):,} / {len(books):,}"
    )

    print_section("4. BUILD SEMANTIC REPRESENTATIONS")

    semantic_books = build_semantic_metadata(books)

    print(
        f"Semantic records created: "
        f"{len(semantic_books):,}"
    )

    if len(semantic_books) != len(books):
        raise ValueError(
            "Semantic record count does not match canonical book count."
        )

    missing_ids = semantic_books["canonical_book_id"].isna().sum()

    print(f"Missing canonical IDs: {missing_ids}")

    if missing_ids:
        raise ValueError(
            "One or more semantic records are missing canonical_book_id."
        )

    semantic_books["semantic_text_length"] = (
        semantic_books["semantic_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    semantic_books["semantic_word_count"] = (
        semantic_books["semantic_text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print_section("5. SEMANTIC TEXT QUALITY")

    empty_semantic_text = (
        semantic_books["semantic_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_titles = (
        semantic_books["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_authors = (
        semantic_books["author"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Empty semantic text: {empty_semantic_text}")
    print(f"Empty titles:        {empty_titles}")
    print(f"Empty authors:       {empty_authors}")

    print()
    print("Semantic text length:")
    print(
        f"  Minimum: {semantic_books['semantic_text_length'].min():,}"
    )
    print(
        f"  Median:  {semantic_books['semantic_text_length'].median():,.0f}"
    )
    print(
        f"  Mean:    {semantic_books['semantic_text_length'].mean():,.1f}"
    )
    print(
        f"  Maximum: {semantic_books['semantic_text_length'].max():,}"
    )

    print()
    print("Semantic word count:")
    print(
        f"  Minimum: {semantic_books['semantic_word_count'].min():,}"
    )
    print(
        f"  Median:  {semantic_books['semantic_word_count'].median():,.0f}"
    )
    print(
        f"  Mean:    {semantic_books['semantic_word_count'].mean():,.1f}"
    )
    print(
        f"  Maximum: {semantic_books['semantic_word_count'].max():,}"
    )

    print_section("6. SEMANTIC REPRESENTATION SAMPLES")

    # Select examples from different metadata richness levels.
    sample_candidates = books.copy()

    sample_candidates["metadata_field_count"] = (
        sample_candidates["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
        + sample_candidates["subjects"]
        .apply(lambda value: len(value) > 0)
        .astype(int)
        + sample_candidates["subject_people"]
        .apply(lambda value: len(value) > 0)
        .astype(int)
        + sample_candidates["subject_places"]
        .apply(lambda value: len(value) > 0)
        .astype(int)
        + sample_candidates["subject_times"]
        .apply(lambda value: len(value) > 0)
        .astype(int)
    )

    sample_rows = []

    # Rich metadata example.
    rich = sample_candidates[
        sample_candidates["metadata_field_count"] >= 4
    ]

    if not rich.empty:
        sample_rows.append(rich.iloc[0])

    # Subject-rich example.
    subject_rich = sample_candidates[
        sample_candidates["subjects"].apply(len) >= 5
    ]

    if not subject_rich.empty:
        sample_rows.append(subject_rich.iloc[0])

    # Sparse example.
    sparse = sample_candidates[
        sample_candidates["metadata_field_count"] == 1
    ]

    if not sparse.empty:
        sample_rows.append(sparse.iloc[0])

    # Deduplicate sample IDs.
    seen_ids = set()

    for row in sample_rows:
        canonical_id = row["canonical_book_id"]

        if canonical_id in seen_ids:
            continue

        seen_ids.add(canonical_id)

        semantic_row = semantic_books[
            semantic_books["canonical_book_id"] == canonical_id
        ].iloc[0]

        print()
        print("-" * 70)
        print(f"Canonical ID: {canonical_id}")
        print(f"Title:        {semantic_row['title']}")
        print(f"Author:       {semantic_row['author']}")
        print()
        print("SEMANTIC TEXT")
        print("-" * 70)
        print(semantic_row["semantic_text"])

    print_section("7. SAVE VALIDATION OUTPUT")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    semantic_books.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")

    print_section("VALIDATION COMPLETE")

    print("Canonical record preservation: PASS")
    print("Semantic record preservation:  PASS")
    print("Canonical IDs preserved:       PASS")

    if empty_semantic_text == 0:
        print("Semantic text completeness:    PASS")
    else:
        print(
            "Semantic text completeness:    REVIEW "
            f"({empty_semantic_text} empty)"
        )

    print()
    print(
        "No embeddings were generated. "
        "This validation only tests the semantic representation."
    )


if __name__ == "__main__":
    main()