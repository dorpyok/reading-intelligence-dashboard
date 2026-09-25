from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"

CANONICAL_BOOKS_PATH = CANONICAL_DIR / "canonical_books.csv"
READER_BOOKS_PATH = CANONICAL_DIR / "reader_books.csv"

IDENTITY_AUDIT_PATH = CANONICAL_DIR / "identity_audit.csv"
SHARED_BOOKS_PATH = CANONICAL_DIR / "shared_books.csv"
UNRESOLVED_IDENTITY_PATH = CANONICAL_DIR / "unresolved_identity.csv"

CANONICAL_ID = "canonical_book_id"


def require_columns(
    dataframe: pd.DataFrame,
    required: list[str],
    dataframe_name: str,
) -> None:
    missing = [column for column in required if column not in dataframe.columns]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().lower().split())


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:

    print("=" * 70)
    print("READING INTELLIGENCE DASHBOARD")
    print("CANONICAL IDENTITY AUDIT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    if not CANONICAL_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical books file not found:\n{CANONICAL_BOOKS_PATH}"
        )

    if not READER_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Reader books file not found:\n{READER_BOOKS_PATH}"
        )

    canonical_books = pd.read_csv(CANONICAL_BOOKS_PATH)
    reader_books = pd.read_csv(READER_BOOKS_PATH)

    print(f"\nCanonical books loaded: {len(canonical_books):,}")
    print(f"Reader/book records loaded: {len(reader_books):,}")

    # ------------------------------------------------------------------
    # Validate canonical table
    # ------------------------------------------------------------------

    require_columns(
        canonical_books,
        [
            CANONICAL_ID,
            "title",
            "author",
            "identity_method",
            "identity_confidence",
        ],
        "canonical_books",
    )

    # ------------------------------------------------------------------
    # Validate reader/book relationship table
    # ------------------------------------------------------------------

    require_columns(
        reader_books,
        [
            "reader_name",
            CANONICAL_ID,
            "source_book_id",
            "reading_status",
        ],
        "reader_books",
    )

    # ------------------------------------------------------------------
    # Create audit view.
    #
    # reader_books contains reader-specific evidence.
    # canonical_books contains book-level identity/metadata.
    #
    # Joining them gives us the complete audit grain.
    # ------------------------------------------------------------------

    reader_audit = reader_books.merge(
        canonical_books[
            [
                CANONICAL_ID,
                "title",
                "author",
                "isbn",
                "openlibrary_work_id",
                "openlibrary_edition_id",
                "identity_method",
                "identity_confidence",
            ]
        ],
        on=CANONICAL_ID,
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 1. Basic preservation
    # ------------------------------------------------------------------

    print_section("1. BASIC DATA PRESERVATION")

    source_records = len(reader_books)
    canonical_records = len(canonical_books)

    missing_ids = reader_books[CANONICAL_ID].isna().sum()

    blank_ids = (
        reader_books[CANONICAL_ID]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    missing_canonical_lookup = reader_audit["title"].isna().sum()

    print(f"Reader/book records:          {source_records:,}")
    print(f"Canonical books:              {canonical_records:,}")
    print(f"Missing canonical IDs:        {missing_ids:,}")
    print(f"Blank canonical IDs:          {blank_ids:,}")
    print(
        f"Records without canonical lookup: "
        f"{missing_canonical_lookup:,}"
    )

    preservation_pass = (
        missing_ids == 0
        and blank_ids == 0
        and missing_canonical_lookup == 0
    )

    print(
        "\nRecord preservation: "
        + ("PASS" if preservation_pass else "FAIL")
    )

    # ------------------------------------------------------------------
    # 2. Identity methods
    # ------------------------------------------------------------------

    print_section("2. IDENTITY METHOD")

    identity_method_counts = (
        canonical_books["identity_method"]
        .fillna("missing")
        .value_counts()
        .rename_axis("identity_method")
        .reset_index(name="canonical_books")
    )

    print(identity_method_counts.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Identity confidence
    # ------------------------------------------------------------------

    print_section("3. IDENTITY CONFIDENCE")

    identity_confidence_counts = (
        canonical_books["identity_confidence"]
        .fillna("missing")
        .value_counts()
        .rename_axis("identity_confidence")
        .reset_index(name="canonical_books")
    )

    print(identity_confidence_counts.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Shared canonical books
    # ------------------------------------------------------------------

    print_section("4. CANONICAL BOOKS SHARED BY MULTIPLE READERS")

    reader_counts = (
        reader_books.groupby(CANONICAL_ID)["reader_name"]
        .nunique()
        .reset_index(name="reader_count")
    )

    shared = reader_counts[
        reader_counts["reader_count"] > 1
    ].copy()

    shared = shared.merge(
        canonical_books,
        on=CANONICAL_ID,
        how="left",
        suffixes=("", "_canonical"),
    )

    shared = shared.sort_values(
        ["reader_count", "title"],
        ascending=[False, True],
    )

    print(f"Shared canonical books: {len(shared):,}")

    if len(shared) > 0:
        print("\nTop shared books:")

        print(
            shared[
                [
                    CANONICAL_ID,
                    "title",
                    "author",
                    "reader_count",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(25)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 5. Record collapse
    # ------------------------------------------------------------------

    print_section("5. RECORD COLLAPSE AUDIT")

    record_counts = (
        reader_books.groupby(CANONICAL_ID)
        .size()
        .reset_index(name="source_record_count")
    )

    collapsed = record_counts[
        record_counts["source_record_count"] > 1
    ].copy()

    collapsed = collapsed.merge(
        canonical_books,
        on=CANONICAL_ID,
        how="left",
    )

    collapsed = collapsed.sort_values(
        ["source_record_count", "title"],
        ascending=[False, True],
    )

    print(
        "Canonical books with multiple "
        f"reader/source records: {len(collapsed):,}"
    )

    print(
        "Maximum source records collapsed into one canonical book: "
        f"{collapsed['source_record_count'].max() if len(collapsed) else 0}"
    )

    if len(collapsed) > 0:
        print("\nExamples:")

        print(
            collapsed[
                [
                    CANONICAL_ID,
                    "title",
                    "author",
                    "source_record_count",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(25)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 6. Title + author fallback
    # ------------------------------------------------------------------

    print_section("6. TITLE + AUTHOR FALLBACK AUDIT")

    title_author = canonical_books[
        canonical_books["identity_method"].eq("title_author")
    ].copy()

    print(f"Title + author identities: {len(title_author):,}")

    if len(title_author) > 0:
        print("\nExamples:")

        print(
            title_author[
                [
                    CANONICAL_ID,
                    "title",
                    "author",
                    "isbn",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 7. ISBN identities
    # ------------------------------------------------------------------

    print_section("7. ISBN IDENTITY AUDIT")

    isbn_books = canonical_books[
        canonical_books["identity_method"].eq("isbn")
    ].copy()

    print(f"ISBN identities: {len(isbn_books):,}")

    if len(isbn_books) > 0:
        print("\nExamples:")

        print(
            isbn_books[
                [
                    CANONICAL_ID,
                    "title",
                    "author",
                    "isbn",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(25)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 8. Open Library Work coverage
    # ------------------------------------------------------------------

    print_section("8. OPEN LIBRARY WORK COVERAGE")

    if "openlibrary_work_id" in canonical_books.columns:

        has_work = (
            canonical_books["openlibrary_work_id"]
            .notna()
            & canonical_books["openlibrary_work_id"]
            .astype(str)
            .str.strip()
            .ne("")
        )

        print("Work ID column:            openlibrary_work_id")
        print(f"Books with Work ID:        {has_work.sum():,}")
        print(f"Books without Work ID:     {(~has_work).sum():,}")
        print(
            f"Work coverage:             "
            f"{has_work.mean() * 100:.1f}%"
        )

    else:

        has_work = pd.Series(
            False,
            index=canonical_books.index,
        )

        print("No Open Library Work ID column found.")

    # ------------------------------------------------------------------
    # 9. Unresolved / lower confidence
    # ------------------------------------------------------------------

    print_section("9. UNRESOLVED / LOWER-CONFIDENCE IDENTITY")

    unresolved_canonical = canonical_books[
        canonical_books["identity_method"].isin(
            [
                "title_author",
                "source_book_id",
            ]
        )
        | canonical_books["identity_confidence"].isin(
            [
                "medium",
                "low",
            ]
        )
    ].copy()

    print(
        "Canonical books requiring additional identity review: "
        f"{len(unresolved_canonical):,}"
    )

    # ------------------------------------------------------------------
    # 10. Potential duplicate title/author identities
    # ------------------------------------------------------------------

    print_section("10. POTENTIAL DUPLICATE TITLE + AUTHOR RECORDS")

    diagnostic = canonical_books.copy()

    diagnostic["normalized_title"] = diagnostic["title"].apply(
        normalize_text
    )

    diagnostic["normalized_author"] = diagnostic["author"].apply(
        normalize_text
    )

    diagnostic["title_author_key"] = (
        diagnostic["normalized_title"]
        + " | "
        + diagnostic["normalized_author"]
    )

    duplicate_title_author = (
        diagnostic.groupby("title_author_key")
        .agg(
            canonical_count=(
                CANONICAL_ID,
                "nunique",
            ),
            title=("title", "first"),
            author=("author", "first"),
        )
        .reset_index()
    )

    duplicate_title_author = duplicate_title_author[
        duplicate_title_author["canonical_count"] > 1
    ].sort_values(
        ["canonical_count", "title"],
        ascending=[False, True],
    )

    print(
        "Title/author keys mapping to multiple canonical books: "
        f"{len(duplicate_title_author):,}"
    )

    if len(duplicate_title_author) > 0:

        print("\nPotential duplicate examples:")

        print(
            duplicate_title_author[
                [
                    "title",
                    "author",
                    "canonical_count",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 11. Suspicious canonical collapses
    # ------------------------------------------------------------------

    print_section("11. SUSPICIOUS CANONICAL COLLAPSES")

    collapse_diagnostic = (
        reader_audit.groupby(CANONICAL_ID)
        .agg(
            record_count=(
                "source_book_id",
                "size",
            ),
            unique_titles=(
                "title",
                lambda values: (
                    values.dropna()
                    .map(normalize_text)
                    .nunique()
                ),
            ),
            unique_authors=(
                "author",
                lambda values: (
                    values.dropna()
                    .map(normalize_text)
                    .nunique()
                ),
            ),
        )
        .reset_index()
    )

    suspicious = collapse_diagnostic[
        (
            collapse_diagnostic["unique_titles"] > 1
        )
        | (
            collapse_diagnostic["unique_authors"] > 1
        )
    ].copy()

    suspicious = suspicious.merge(
        canonical_books[
            [
                CANONICAL_ID,
                "title",
                "author",
                "identity_method",
                "identity_confidence",
            ]
        ],
        on=CANONICAL_ID,
        how="left",
    )

    suspicious = suspicious.sort_values(
        [
            "unique_titles",
            "unique_authors",
        ],
        ascending=False,
    )

    print(
        "Canonical IDs containing multiple normalized "
        f"titles/authors: {len(suspicious):,}"
    )

    if len(suspicious) > 0:

        print("\nPotentially suspicious examples:")

        print(
            suspicious[
                [
                    CANONICAL_ID,
                    "title",
                    "author",
                    "record_count",
                    "unique_titles",
                    "unique_authors",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 12. Reader coverage
    # ------------------------------------------------------------------

    print_section("12. READER COVERAGE")

    reader_summary = (
        reader_books.groupby("reader_name")
        .agg(
            reader_records=(
                CANONICAL_ID,
                "size",
            ),
            canonical_books=(
                CANONICAL_ID,
                "nunique",
            ),
        )
        .reset_index()
    )

    print(reader_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 13. Identity audit table
    # ------------------------------------------------------------------

    print_section("13. BUILDING IDENTITY AUDIT")

    audit_rows = []

    for _, row in canonical_books.iterrows():

        canonical_id = row[CANONICAL_ID]

        records = reader_audit[
            reader_audit[CANONICAL_ID].eq(canonical_id)
        ]

        readers = sorted(
            records["reader_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        unique_titles = (
            records["title"]
            .dropna()
            .map(normalize_text)
            .nunique()
        )

        unique_authors = (
            records["author"]
            .dropna()
            .map(normalize_text)
            .nunique()
        )

        requires_review = (
            row["identity_method"]
            in [
                "title_author",
                "source_book_id",
            ]
            or row["identity_confidence"]
            in [
                "medium",
                "low",
            ]
            or unique_titles > 1
            or unique_authors > 1
        )

        audit_rows.append(
            {
                CANONICAL_ID: canonical_id,
                "title": row.get("title"),
                "author": row.get("author"),
                "identity_method": row.get(
                    "identity_method"
                ),
                "identity_confidence": row.get(
                    "identity_confidence"
                ),
                "reader_count": len(readers),
                "readers": "|".join(readers),
                "source_record_count": len(records),
                "unique_source_book_ids": records[
                    "source_book_id"
                ].nunique(),
                "unique_titles": unique_titles,
                "unique_authors": unique_authors,
                "potential_title_conflict": (
                    unique_titles > 1
                ),
                "potential_author_conflict": (
                    unique_authors > 1
                ),
                "requires_review": requires_review,
            }
        )

    identity_audit = pd.DataFrame(audit_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    CANONICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    identity_audit.to_csv(
        IDENTITY_AUDIT_PATH,
        index=False,
    )

    shared.to_csv(
        SHARED_BOOKS_PATH,
        index=False,
    )

    unresolved_canonical.to_csv(
        UNRESOLVED_IDENTITY_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print_section("AUDIT OUTPUTS")

    print(f"Identity audit:       {IDENTITY_AUDIT_PATH}")
    print(f"Shared books:         {SHARED_BOOKS_PATH}")
    print(f"Unresolved identity:  {UNRESOLVED_IDENTITY_PATH}")

    print_section("AUDIT SUMMARY")

    print(
        f"Canonical books:                         "
        f"{len(canonical_books):,}"
    )

    print(
        f"Reader/book records:                     "
        f"{len(reader_books):,}"
    )

    print(
        f"Shared canonical books:                  "
        f"{len(shared):,}"
    )

    print(
        f"Title/author identities:                 "
        f"{len(title_author):,}"
    )

    print(
        f"ISBN identities:                         "
        f"{len(isbn_books):,}"
    )

    print(
        f"Canonical books requiring review:        "
        f"{len(unresolved_canonical):,}"
    )

    print(
        f"Potential title/author duplicate groups: "
        f"{len(duplicate_title_author):,}"
    )

    print(
        f"Suspicious canonical collapses:          "
        f"{len(suspicious):,}"
    )

    print()
    print("=" * 70)
    print("CANONICAL IDENTITY AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()