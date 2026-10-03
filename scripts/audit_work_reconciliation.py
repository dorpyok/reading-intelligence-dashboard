from pathlib import Path
import sys

import pandas as pd


# Allow imports from the repository root when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.model.work_reconciliation import build_reconciliation_audit


CANONICAL_BOOKS_PATH = (
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
    / "work_reconciliation_candidates.csv"
)


def main() -> None:
    print("Loading canonical book corpus...")

    books = pd.read_csv(CANONICAL_BOOKS_PATH)

    print(f"Loaded {len(books):,} canonical books.")

    audit = build_reconciliation_audit(books)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Reconciliation audit complete.")
    print(f"Candidate pairs: {len(audit):,}")
    print(f"Output: {OUTPUT_PATH}")

    if audit.empty:
        print("No candidate pairs were identified.")
        return

    print()
    print("Confidence breakdown:")
    print(
        audit["confidence"]
        .value_counts()
        .reindex(["high", "medium", "low"], fill_value=0)
        .to_string()
    )

    print()
    print("Evidence signals:")

    evidence_columns = [
        "title_match",
        "author_match",
        "isbn_match",
        "work_id_match",
        "edition_id_match",
    ]

    for column in evidence_columns:
        print(
            f"  {column}: "
            f"{int(audit[column].sum()):,}"
        )

    print()
    print("Top candidate pairs:")

    display_columns = [
        "title_a",
        "title_b",
        "author_a",
        "author_b",
        "description_similarity",
        "evidence_score",
        "confidence",
        "evidence_reasons",
    ]

    print(
        audit[display_columns]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()