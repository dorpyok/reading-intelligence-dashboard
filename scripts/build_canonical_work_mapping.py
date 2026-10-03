from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.model.work_mapping import (
    build_canonical_work_mapping,
    validate_work_mapping,
)


CANONICAL_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "canonical_books.csv"
)

RECONCILIATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "work_reconciliation_candidates.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "canonical_work_mapping.csv"
)


def main() -> None:
    print("Loading canonical books...")

    books = pd.read_csv(CANONICAL_BOOKS_PATH)

    print(f"Canonical books: {len(books):,}")

    print("Loading reconciliation audit...")

    audit = pd.read_csv(RECONCILIATION_PATH)

    print(f"Reconciliation candidates: {len(audit):,}")

    mapping = build_canonical_work_mapping(
        books,
        audit,
        minimum_confidence="high",
    )

    validate_work_mapping(
        mapping,
        books,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Canonical work mapping complete.")
    print(f"Output: {OUTPUT_PATH}")

    print()
    print("Mapping summary:")

    print(
        f"  Canonical books: "
        f"{mapping['canonical_book_id'].nunique():,}"
    )

    print(
        f"  Canonical works: "
        f"{mapping['canonical_work_id'].nunique():,}"
    )

    print(
        f"  Reconciled records: "
        f"{(mapping['work_mapping_status'] == 'reconciled').sum():,}"
    )

    print(
        f"  Standalone records: "
        f"{(mapping['work_mapping_status'] == 'standalone').sum():,}"
    )

    work_sizes = (
        mapping.groupby("canonical_work_id")
        .size()
        .sort_values(ascending=False)
    )

    print()
    print("Work group sizes:")

    print(
        work_sizes.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Largest reconciled work groups:")

    print(
        work_sizes.head(10).to_string()
    )


if __name__ == "__main__":
    main()