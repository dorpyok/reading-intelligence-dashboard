from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metadata_normalization import (
    clean_list,
    clean_subjects,
    clean_text,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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
    / "semantic_representation_comparison.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_list_field(value) -> list[str]:
    """
    Parse a list-like dataframe field.

    Handles:
      - actual Python lists
      - serialized Python lists
      - comma-delimited strings
      - pipe-delimited strings
      - semicolon-delimited strings
      - missing values
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    # Try serialized Python list first.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return parsed

        except (ValueError, SyntaxError):
            pass

    # Fall back to the same list parsing rules used by normalization.
    return clean_list(text)


# ---------------------------------------------------------------------------
# Representation builders
# ---------------------------------------------------------------------------

def build_core_text(row: pd.Series) -> str:
    """
    Representation A:
        Title + Author + Description
    """
    parts = []

    title = clean_text(row.get("title"))
    author = clean_text(row.get("author"))
    description = clean_text(row.get("description"))

    if title:
        parts.append(f"Title: {title}")

    if author:
        parts.append(f"Author: {author}")

    if description:
        parts.append(f"Description: {description}")

    return "\n".join(parts)


def build_raw_subject_text(row: pd.Series) -> str:
    """
    Representation B:
        Core + raw Open Library subjects
    """
    parts = [build_core_text(row)]

    raw_subjects = parse_list_field(row.get("subjects"))

    if raw_subjects:
        subject_text = ", ".join(
            clean_text(subject)
            for subject in raw_subjects
            if clean_text(subject)
        )

        if subject_text:
            parts.append(f"Subjects: {subject_text}")

    return "\n".join(parts)


def build_curated_subject_text(row: pd.Series) -> str:
    """
    Representation C:
        Core + cleaned Open Library subjects.

    IMPORTANT:
    This function explicitly calls clean_subjects() on the parsed raw
    subjects. The cleaned result is what gets added to the representation.
    """
    parts = [build_core_text(row)]

    raw_subjects = parse_list_field(row.get("subjects"))

    curated_subjects = clean_subjects(raw_subjects)

    if curated_subjects:
        parts.append(
            f"Subjects: {', '.join(curated_subjects)}"
        )

    return "\n".join(parts)


def build_full_curated_text(row: pd.Series) -> str:
    """
    Representation D:
        Core
        + curated subjects
        + people
        + places
        + times

    This remains experimental. People/places/times are included here only
    for comparison and are NOT currently part of the proposed production
    semantic representation.
    """
    parts = [build_curated_subject_text(row)]

    people = clean_list(row.get("people"))
    places = clean_list(row.get("places"))
    times = clean_list(row.get("times"))

    if people:
        parts.append(f"People: {people}")

    if places:
        parts.append(f"Places: {places}")

    if times:
        parts.append(f"Times: {times}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe_lengths(
    series: pd.Series,
    label: str,
) -> None:
    char_lengths = series.str.len()
    word_lengths = series.str.split().str.len()

    print(f"{label}")
    print(
        f"  Characters: "
        f"min={char_lengths.min()} "
        f"median={char_lengths.median():.0f} "
        f"mean={char_lengths.mean():.1f} "
        f"max={char_lengths.max()}"
    )
    print(
        f"  Words: "
        f"min={word_lengths.min()} "
        f"median={word_lengths.median():.0f} "
        f"mean={word_lengths.mean():.1f} "
        f"max={word_lengths.max()}"
    )
    print()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("SEMANTIC REPRESENTATION COMPARISON")
    print("=" * 70)
    print(f"Input: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Canonical books: {len(df):,}")
    print()

    # ------------------------------------------------------------------
    # Build representations
    # ------------------------------------------------------------------

    df["representation_core"] = df.apply(
        build_core_text,
        axis=1,
    )

    df["representation_raw_subjects"] = df.apply(
        build_raw_subject_text,
        axis=1,
    )

    df["representation_curated_subjects"] = df.apply(
        build_curated_subject_text,
        axis=1,
    )

    df["representation_full_curated"] = df.apply(
        build_full_curated_text,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    print("=" * 70)
    print("1. REPRESENTATION COUNTS")
    print("=" * 70)

    print(
        f"Core representations:             "
        f"{df['representation_core'].notna().sum():,}"
    )

    print(
        f"Raw subject representations:      "
        f"{df['representation_raw_subjects'].notna().sum():,}"
    )

    print(
        f"Curated subject representations:  "
        f"{df['representation_curated_subjects'].notna().sum():,}"
    )

    print(
        f"Full curated representations:     "
        f"{df['representation_full_curated'].notna().sum():,}"
    )

    print()

    # ------------------------------------------------------------------
    # Length statistics
    # ------------------------------------------------------------------

    print("=" * 70)
    print("2. REPRESENTATION LENGTH")
    print("=" * 70)
    print()

    describe_lengths(
        df["representation_core"],
        "A. CORE",
    )

    describe_lengths(
        df["representation_raw_subjects"],
        "B. CORE + RAW SUBJECTS",
    )

    describe_lengths(
        df["representation_curated_subjects"],
        "C. CORE + CURATED SUBJECTS",
    )

    describe_lengths(
        df["representation_full_curated"],
        "D. FULL CURATED",
    )

    # ------------------------------------------------------------------
    # Incremental metadata
    # ------------------------------------------------------------------

    core_chars = df["representation_core"].str.len()
    raw_chars = df["representation_raw_subjects"].str.len()
    curated_chars = df["representation_curated_subjects"].str.len()
    full_chars = df["representation_full_curated"].str.len()

    raw_added = raw_chars - core_chars
    curated_added = curated_chars - core_chars
    curation_difference = curated_chars - raw_chars
    people_places_times_added = full_chars - curated_chars

    print("=" * 70)
    print("3. INCREMENTAL METADATA")
    print("=" * 70)

    print(
        "Raw subjects add: "
        f"median={raw_added.median():.0f} chars "
        f"mean={raw_added.mean():.1f} chars"
    )

    print(
        "Curated subjects add: "
        f"median={curated_added.median():.0f} chars "
        f"mean={curated_added.mean():.1f} chars"
    )

    print(
        "Curated vs raw difference: "
        f"median={curation_difference.median():.0f} chars "
        f"mean={curation_difference.mean():.1f} chars"
    )

    print(
        "People/places/times add beyond curated subjects: "
        f"median={people_places_times_added.median():.0f} chars "
        f"mean={people_places_times_added.mean():.1f} chars"
    )

    print()

    # ------------------------------------------------------------------
    # Direct cleaner verification
    # ------------------------------------------------------------------

    print("=" * 70)
    print("4. DIRECT CURATION CHECK")
    print("=" * 70)

    sample_title = "House of Earth and Blood (Crescent City, #1)"

    matches = df[
        df["title"].astype(str).str.contains(
            "House of Earth and Blood",
            case=False,
            na=False,
        )
    ]

    if not matches.empty:
        row = matches.iloc[0]

        raw_subjects = parse_list_field(row.get("subjects"))
        curated_subjects = clean_subjects(raw_subjects)

        print(f"BOOK: {sample_title}")
        print()
        print("RAW SUBJECTS:")
        for subject in raw_subjects:
            print(f"  - {subject}")

        print()
        print("CURATED SUBJECTS:")
        for subject in curated_subjects:
            print(f"  + {subject}")

        print()

        removed = [
            subject
            for subject in raw_subjects
            if clean_text(subject).casefold()
            not in {
                clean_text(value).casefold()
                for value in curated_subjects
            }
        ]

        print("REMOVED:")
        for subject in removed:
            print(f"  - {subject}")

        print()

    # ------------------------------------------------------------------
    # Representation samples
    # ------------------------------------------------------------------

    print("=" * 70)
    print("5. REPRESENTATION SAMPLES")
    print("=" * 70)

    sample_titles = [
        "House of Earth and Blood",
        "Make Me Hate You",
        "Fifty Shades of Grey",
    ]

    for title in sample_titles:
        matches = df[
            df["title"].astype(str).str.contains(
                title,
                case=False,
                na=False,
            )
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        print("-" * 70)
        print(f"BOOK: {row.get('title')}")
        print(f"AUTHOR: {row.get('author')}")
        print("-" * 70)

        print()
        print("A. CORE")
        print("-" * 30)
        print(row["representation_core"])

        print()
        print("B. CORE + RAW SUBJECTS")
        print("-" * 30)
        print(row["representation_raw_subjects"])

        print()
        print("C. CORE + CURATED SUBJECTS")
        print("-" * 30)
        print(row["representation_curated_subjects"])

        print()
        print("D. FULL CURATED")
        print("-" * 30)
        print(row["representation_full_curated"])

        print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    comparison_columns = [
        "canonical_book_id",
        "title",
        "author",
        "representation_core",
        "representation_raw_subjects",
        "representation_curated_subjects",
        "representation_full_curated",
    ]

    df[comparison_columns].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("6. SAVE")
    print("=" * 70)
    print(f"Saved comparison: {OUTPUT_PATH}")
    print()

    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("No embeddings were generated.")


if __name__ == "__main__":
    main()