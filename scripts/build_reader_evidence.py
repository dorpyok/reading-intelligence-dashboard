from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.analytics.reader_evidence import (  # noqa: E402
    build_reader_evidence,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CANONICAL_READER_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reader_books.csv"
)

WORK_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "canonical_work_mapping.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reader_evidence.csv"
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_READER_BOOK_COLUMNS = {
    "reader_id",
    "canonical_book_id",
    "reading_status",
    "user_rating",
    "date_read",
}

REQUIRED_WORK_MAPPING_COLUMNS = {
    "canonical_book_id",
    "canonical_work_id",
}


def validate_reader_books(
    reader_books: pd.DataFrame,
) -> None:
    """Validate the canonical reader/book dataset."""
    missing = (
        REQUIRED_READER_BOOK_COLUMNS
        - set(reader_books.columns)
    )

    if missing:
        raise ValueError(
            "Canonical reader books are missing required columns: "
            + ", ".join(sorted(missing))
        )


def validate_work_mapping(
    work_mapping: pd.DataFrame,
) -> None:
    """Validate the canonical work mapping."""
    missing = (
        REQUIRED_WORK_MAPPING_COLUMNS
        - set(work_mapping.columns)
    )

    if missing:
        raise ValueError(
            "Canonical work mapping is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if work_mapping[
        "canonical_book_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical work mapping contains duplicate "
            "canonical_book_id values."
        )

    if work_mapping[
        "canonical_work_id"
    ].isna().any():
        raise ValueError(
            "Canonical work mapping contains missing "
            "canonical_work_id values."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading canonical reader books...")

    reader_books = pd.read_csv(
        CANONICAL_READER_BOOKS_PATH
    )

    validate_reader_books(
        reader_books
    )

    print(
        f"Loaded {len(reader_books):,} "
        "reader/book records."
    )

    print(
        f"Readers: "
        f"{reader_books['reader_id'].nunique():,}"
    )

    print()
    print("Loading canonical work mapping...")

    work_mapping = pd.read_csv(
        WORK_MAPPING_PATH
    )

    validate_work_mapping(
        work_mapping
    )

    print(
        f"Canonical work records: "
        f"{len(work_mapping):,}"
    )

    print(
        f"Canonical works: "
        f"{work_mapping['canonical_work_id'].nunique():,}"
    )

    # -----------------------------------------------------------------------
    # Attach canonical work identity
    # -----------------------------------------------------------------------

    print()
    print(
        "Attaching canonical work identity "
        "to reader/book records..."
    )

    work_columns = [
        "canonical_book_id",
        "canonical_work_id",
    ]

    reader_books = reader_books.merge(
        work_mapping[work_columns],
        on="canonical_book_id",
        how="left",
        validate="many_to_one",
    )

    missing_work_ids = reader_books[
        "canonical_work_id"
    ].isna()

    if missing_work_ids.any():
        missing_count = int(
            missing_work_ids.sum()
        )

        raise ValueError(
            f"{missing_count:,} reader/book records "
            "could not be assigned a canonical_work_id."
        )

    print(
        "Canonical work identity attached successfully."
    )

    print(
        f"Reader/book records: "
        f"{len(reader_books):,}"
    )

    print(
        f"Canonical works represented: "
        f"{reader_books['canonical_work_id'].nunique():,}"
    )

    # -----------------------------------------------------------------------
    # Build evidence
    # -----------------------------------------------------------------------

    print()
    print("Building reader evidence...")

    evidence = build_reader_evidence(
        reader_books
    )

    print(
        f"Evidence records: "
        f"{len(evidence):,}"
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    if len(evidence) != len(reader_books):
        raise ValueError(
            "Evidence record count does not match "
            "the input reader/book record count."
        )

    if (
        evidence["canonical_book_id"].nunique()
        != reader_books["canonical_book_id"].nunique()
    ):
        raise ValueError(
            "Canonical book coverage changed unexpectedly."
        )

    if (
        evidence["reader_id"].nunique()
        != reader_books["reader_id"].nunique()
    ):
        raise ValueError(
            "Reader coverage changed unexpectedly."
        )

    if (
        evidence["canonical_work_id"].nunique()
        != reader_books["canonical_work_id"].nunique()
    ):
        raise ValueError(
            "Canonical work coverage changed unexpectedly."
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    print()
    print("Evidence strength:")
    print(
        evidence[
            "evidence_strength"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Reading status:")
    print(
        evidence[
            "reading_status"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Positive evidence:")
    print(
        evidence[
            "is_positive"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Negative evidence:")
    print(
        evidence[
            "is_negative"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Exposure:")
    print(
        evidence[
            "is_exposure"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Reader-level evidence summary:")

    summary = (
        evidence
        .groupby("reader_id")
        .agg(
            records=(
                "canonical_book_id",
                "count",
            ),
            canonical_works=(
                "canonical_work_id",
                "nunique",
            ),
            read_books=(
                "is_read",
                "sum",
            ),
            positive_books=(
                "is_positive",
                "sum",
            ),
            negative_books=(
                "is_negative",
                "sum",
            ),
            exposure_books=(
                "is_exposure",
                "sum",
            ),
            tbr_books=(
                "is_tbr",
                "sum",
            ),
            currently_reading=(
                "is_currently_reading",
                "sum",
            ),
            dnf_books=(
                "is_dnf",
                "sum",
            ),
            unknown_books=(
                "evidence_strength",
                lambda values: (
                    values == "unknown"
                ).sum(),
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Saved evidence dataset:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()