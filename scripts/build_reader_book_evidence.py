"""
Build the reader/book-level evidence layer.

Input:
    data/processed/canonical/reader_evidence.csv

Output:
    data/processed/canonical/reader_book_evidence.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.reader_book_evidence import (  # noqa: E402
    aggregate_reader_book_evidence,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reader_evidence.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reader_book_evidence.csv"
)


def main() -> None:
    print("Loading reader-level evidence...")

    evidence = pd.read_csv(INPUT_PATH)

    print(f"Input evidence records: {len(evidence):,}")

    aggregated = aggregate_reader_book_evidence(evidence)

    print(
        f"Unique reader/book relationships: "
        f"{len(aggregated):,}"
    )

    print(
        f"Records aggregated: "
        f"{len(evidence) - len(aggregated):,}"
    )

    print("\nReader counts:")

    reader_summary = (
        aggregated.groupby("reader_id")
        .agg(
            reader_book_relationships=(
                "canonical_book_id",
                "count",
            ),
            positive_books=("is_positive", "sum"),
            negative_books=("is_negative", "sum"),
            read_books=("is_read", "sum"),
            exposure_books=("is_exposure", "sum"),
            tbr_books=("is_tbr", "sum"),
        )
        .reset_index()
    )

    print(reader_summary.to_string(index=False))

    print("\nPreference signal breakdown:")

    print(
        aggregated["preference_signal"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nEvidence strength breakdown:")

    print(
        aggregated["evidence_strength"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    duplicate_count = aggregated.duplicated(
        ["reader_id", "canonical_book_id"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "Aggregated output still contains duplicate "
            f"reader/book relationships: {duplicate_count}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    aggregated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")

    print("\nFinal validation:")

    print(
        f"  Input records: "
        f"{len(evidence):,}"
    )

    print(
        f"  Output reader/book relationships: "
        f"{len(aggregated):,}"
    )

    print(
        f"  Unique readers: "
        f"{aggregated['reader_id'].nunique():,}"
    )

    print(
        f"  Unique canonical books: "
        f"{aggregated['canonical_book_id'].nunique():,}"
    )

    print(
        f"  Duplicate reader/book relationships: "
        f"{duplicate_count}"
    )

    print(
        "\nReader/book aggregation complete."
    )


if __name__ == "__main__":
    main()