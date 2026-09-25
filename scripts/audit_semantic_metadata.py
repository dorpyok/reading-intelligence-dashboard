from __future__ import annotations

from pathlib import Path
import ast
import json
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = PROJECT_ROOT / "data" / "processed" / "canonical"
CANONICAL_BOOKS_PATH = CANONICAL_DIR / "canonical_books.csv"

OUTPUT_PATH = CANONICAL_DIR / "semantic_metadata_audit.csv"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IDENTITY_METHODS = [
    "openlibrary_work",
    "isbn",
    "title_author",
    "source_book_id",
]

SEMANTIC_FIELDS = [
    "description",
    "subjects",
    "subject_people",
    "subject_places",
    "subject_times",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def is_missing(value) -> bool:
    if value is None:
        return True

    if pd.isna(value):
        return True

    text = str(value).strip()

    if text == "":
        return True

    if text.lower() in {
        "nan",
        "none",
        "null",
        "[]",
        "{}",
    }:
        return True

    return False


def parse_collection(value) -> list[str]:
    """
    Convert common CSV representations of list-like Open Library metadata
    into a normalized Python list.

    Handles examples such as:

        ['Fantasy', 'Magic']
        ["Fantasy", "Magic"]
        Fantasy, Magic
        Fantasy; Magic
        Fantasy
        NaN
    """

    if is_missing(value):
        return []

    text = str(value).strip()

    # Python-list / JSON-list representations
    if (
        text.startswith("[")
        and text.endswith("]")
    ):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                return [
                    str(item).strip()
                    for item in parsed
                    if not is_missing(item)
                ]
        except (ValueError, SyntaxError):
            pass

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if not is_missing(item)
                ]
        except (ValueError, TypeError):
            pass

    # Delimited fallback
    if ";" in text:
        parts = text.split(";")

    elif "," in text:
        parts = text.split(",")

    else:
        parts = [text]

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def clean_description(value) -> str:
    """
    Basic audit-only description cleanup.

    This does NOT create the final semantic representation.
    """

    if is_missing(value):
        return ""

    text = str(value)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def collection_stats(series: pd.Series) -> pd.Series:
    parsed = series.apply(parse_collection)

    counts = parsed.apply(len)

    return pd.Series(
        {
            "books_with_values": int(
                counts.gt(0).sum()
            ),
            "books_without_values": int(
                counts.eq(0).sum()
            ),
            "coverage_pct": round(
                counts.gt(0).mean() * 100,
                2,
            ),
            "mean_items": round(
                counts.mean(),
                2,
            ),
            "median_items": round(
                counts.median(),
                2,
            ),
            "max_items": int(
                counts.max()
            ),
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("READING INTELLIGENCE DASHBOARD")
    print("CANONICAL SEMANTIC METADATA AUDIT")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    if not CANONICAL_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical books file not found:\n"
            f"{CANONICAL_BOOKS_PATH}"
        )

    books = pd.read_csv(
        CANONICAL_BOOKS_PATH
    )

    print(
        f"\nCanonical books loaded: "
        f"{len(books):,}"
    )

    required_columns = [
        "canonical_book_id",
        "title",
        "author",
        "identity_method",
        "identity_confidence",
        *SEMANTIC_FIELDS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in books.columns
    ]

    if missing_columns:
        raise ValueError(
            "canonical_books.csv is missing expected columns:\n"
            f"{missing_columns}\n\n"
            f"Available columns:\n{list(books.columns)}"
        )

    # -----------------------------------------------------------------------
    # 1. Overall metadata coverage
    # -----------------------------------------------------------------------

    print_section(
        "1. OVERALL SEMANTIC METADATA COVERAGE"
    )

    summary_rows = []

    for field in SEMANTIC_FIELDS:

        if field == "description":

            values = books[field].apply(
                clean_description
            )

            populated = values.str.len().gt(0)

            word_counts = values.apply(
                lambda text: len(
                    text.split()
                )
                if text
                else 0
            )

            row = {
                "field": field,
                "books_with_values": int(
                    populated.sum()
                ),
                "books_without_values": int(
                    (~populated).sum()
                ),
                "coverage_pct": round(
                    populated.mean() * 100,
                    2,
                ),
                "mean_items": round(
                    word_counts.mean(),
                    2,
                ),
                "median_items": round(
                    word_counts.median(),
                    2,
                ),
                "max_items": int(
                    word_counts.max()
                ),
            }

        else:

            stats = collection_stats(
                books[field]
            )

            row = {
                "field": field,
                **stats.to_dict(),
            }

        summary_rows.append(row)

        print(f"\n{field}")

        print(
            f"  With values:    "
            f"{row['books_with_values']:,}"
        )

        print(
            f"  Without values: "
            f"{row['books_without_values']:,}"
        )

        print(
            f"  Coverage:       "
            f"{row['coverage_pct']:.1f}%"
        )

        print(
            f"  Mean items:     "
            f"{row['mean_items']}"
        )

        print(
            f"  Median items:   "
            f"{row['median_items']}"
        )

        print(
            f"  Maximum:        "
            f"{row['max_items']}"
        )

    # -----------------------------------------------------------------------
    # 2. Metadata coverage by identity method
    # -----------------------------------------------------------------------

    print_section(
        "2. METADATA COVERAGE BY IDENTITY METHOD"
    )

    coverage_rows = []

    for identity_method in IDENTITY_METHODS:

        subset = books[
            books["identity_method"]
            .eq(identity_method)
        ]

        if subset.empty:
            continue

        print(
            f"\n{identity_method}: "
            f"{len(subset):,} books"
        )

        for field in SEMANTIC_FIELDS:

            if field == "description":

                populated = (
                    subset[field]
                    .apply(clean_description)
                    .str.len()
                    .gt(0)
                )

            else:

                populated = (
                    subset[field]
                    .apply(
                        lambda value:
                        len(
                            parse_collection(
                                value
                            )
                        ) > 0
                    )
                )

            coverage = (
                populated.mean() * 100
            )

            coverage_rows.append(
                {
                    "identity_method":
                        identity_method,
                    "field": field,
                    "book_count":
                        len(subset),
                    "books_with_values":
                        int(populated.sum()),
                    "coverage_pct":
                        round(
                            coverage,
                            2,
                        ),
                }
            )

            print(
                f"  {field:<18}"
                f"{coverage:6.1f}%"
            )

    # -----------------------------------------------------------------------
    # 3. Open Library Work coverage
    # -----------------------------------------------------------------------

    print_section(
        "3. OPEN LIBRARY WORK IDENTITY COVERAGE"
    )

    has_work = (
        books["openlibrary_work_id"]
        .notna()
        & books["openlibrary_work_id"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    print(
        f"Books with Work ID:    "
        f"{has_work.sum():,}"
    )

    print(
        f"Books without Work ID: "
        f"{(~has_work).sum():,}"
    )

    print(
        f"Work coverage:         "
        f"{has_work.mean() * 100:.1f}%"
    )

    # -----------------------------------------------------------------------
    # 4. Books without Work IDs
    # -----------------------------------------------------------------------

    print_section(
        "4. BOOKS WITHOUT OPEN LIBRARY WORK IDs"
    )

    no_work = books[
        ~has_work
    ].copy()

    print(
        f"Books requiring additional "
        f"identity/enrichment work: {len(no_work):,}"
    )

    if len(no_work) > 0:

        print("\nIdentity method breakdown:")

        print(
            no_work[
                "identity_method"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nSample:"
        )

        print(
            no_work[
                [
                    "canonical_book_id",
                    "title",
                    "author",
                    "isbn",
                    "identity_method",
                    "identity_confidence",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # -----------------------------------------------------------------------
    # 5. Description quality
    # -----------------------------------------------------------------------

    print_section(
        "5. DESCRIPTION QUALITY"
    )

    descriptions = books[
        "description"
    ].apply(clean_description)

    description_words = descriptions.apply(
        lambda text: len(
            text.split()
        )
        if text
        else 0
    )

    print(
        f"Books with descriptions: "
        f"{description_words.gt(0).sum():,}"
    )

    print(
        f"Books without descriptions: "
        f"{description_words.eq(0).sum():,}"
    )

    print(
        f"Median description length: "
        f"{description_words.median():.0f} words"
    )

    print(
        f"Mean description length: "
        f"{description_words.mean():.1f} words"
    )

    print(
        f"Descriptions under 10 words: "
        f"{description_words.between(1, 9).sum():,}"
    )

    print(
        f"Descriptions over 500 words: "
        f"{description_words.gt(500).sum():,}"
    )

    # -----------------------------------------------------------------------
    # 6. Subject distribution
    # -----------------------------------------------------------------------

    print_section(
        "6. SUBJECT METADATA DISTRIBUTION"
    )

    subject_lists = books[
        "subjects"
    ].apply(parse_collection)

    all_subjects = []

    for subjects in subject_lists:
        all_subjects.extend(subjects)

    subject_counts = (
        pd.Series(
            all_subjects,
            dtype="object",
        )
        .value_counts()
    )

    print(
        f"Unique subject labels: "
        f"{len(subject_counts):,}"
    )

    print(
        f"Total subject assignments: "
        f"{len(all_subjects):,}"
    )

    if len(subject_counts) > 0:

        print(
            "\nMost common subject labels:"
        )

        print(
            subject_counts
            .head(40)
            .to_string()
        )

    # -----------------------------------------------------------------------
    # 7. Subject people / places / times
    # -----------------------------------------------------------------------

    print_section(
        "7. SUBJECT PEOPLE / PLACES / TIMES"
    )

    for field in [
        "subject_people",
        "subject_places",
        "subject_times",
    ]:

        stats = collection_stats(
            books[field]
        )

        print(
            f"\n{field}"
        )

        print(
            f"  Coverage: "
            f"{stats['coverage_pct']:.1f}%"
        )

        print(
            f"  Mean items: "
            f"{stats['mean_items']}"
        )

        print(
            f"  Median items: "
            f"{stats['median_items']}"
        )

        print(
            f"  Maximum: "
            f"{stats['max_items']}"
        )

    # -----------------------------------------------------------------------
    # 8. Semantic sparsity
    # -----------------------------------------------------------------------

    print_section(
        "8. SEMANTIC METADATA SPARSITY"
    )

    semantic_presence = pd.DataFrame(
        {
            "description": descriptions.str.len().gt(0),
            "subjects": subject_lists.apply(
                lambda values: len(values) > 0
            ),
            "subject_people": books[
                "subject_people"
            ].apply(
                lambda value:
                len(
                    parse_collection(
                        value
                    )
                ) > 0
            ),
            "subject_places": books[
                "subject_places"
            ].apply(
                lambda value:
                len(
                    parse_collection(
                        value
                    )
                ) > 0
            ),
            "subject_times": books[
                "subject_times"
            ].apply(
                lambda value:
                len(
                    parse_collection(
                        value
                    )
                ) > 0
            ),
        }
    )

    semantic_presence[
        "semantic_field_count"
    ] = semantic_presence.sum(
        axis=1
    )

    print(
        "Books by number of populated "
        "semantic fields:"
    )

    print(
        semantic_presence[
            "semantic_field_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nBooks with NO semantic metadata: "
        f"{semantic_presence['semantic_field_count'].eq(0).sum():,}"
    )

    print(
        "Books with exactly one semantic field: "
        f"{semantic_presence['semantic_field_count'].eq(1).sum():,}"
    )

    print(
        "Books with 3+ semantic fields: "
        f"{semantic_presence['semantic_field_count'].ge(3).sum():,}"
    )

    # -----------------------------------------------------------------------
    # 9. Books with strongest metadata
    # -----------------------------------------------------------------------

    print_section(
        "9. RICHEST SEMANTIC METADATA"
    )

    richness = semantic_presence[
        "semantic_field_count"
    ].sort_values(
        ascending=False
    )

    richest_indices = richness.head(
        10
    ).index

    richest = books.loc[
        richest_indices,
        [
            "canonical_book_id",
            "title",
            "author",
        ],
    ].copy()

    richest[
        "semantic_field_count"
    ] = semantic_presence.loc[
        richest_indices,
        "semantic_field_count",
    ]

    print(
        richest.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------------
    # 10. Sparse metadata examples
    # -----------------------------------------------------------------------

    print_section(
        "10. SPARSE METADATA EXAMPLES"
    )

    sparse = books.copy()

    sparse[
        "semantic_field_count"
    ] = semantic_presence[
        "semantic_field_count"
    ]

    sparse = sparse.sort_values(
        [
            "semantic_field_count",
            "title",
        ],
        ascending=[
            True,
            True,
        ],
    )

    print(
        sparse[
            [
                "canonical_book_id",
                "title",
                "author",
                "identity_method",
                "identity_confidence",
                "semantic_field_count",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    # -----------------------------------------------------------------------
    # 11. Description + subjects sample
    # -----------------------------------------------------------------------

    print_section(
        "11. SAMPLE SEMANTIC METADATA"
    )

    sample = books[
        descriptions.str.len().gt(0)
        & subject_lists.apply(
            lambda values:
            len(values) > 0
        )
    ].sample(
        n=min(
            15,
            (
                descriptions.str.len().gt(0)
                & subject_lists.apply(
                    lambda values:
                    len(values) > 0
                )
            ).sum(),
        ),
        random_state=42,
    )

    for _, row in sample.iterrows():

        description = clean_description(
            row["description"]
        )

        subjects = parse_collection(
            row["subjects"]
        )

        people = parse_collection(
            row["subject_people"]
        )

        places = parse_collection(
            row["subject_places"]
        )

        times = parse_collection(
            row["subject_times"]
        )

        print()
        print(
            f"TITLE: {row['title']}"
        )

        print(
            f"AUTHOR: {row['author']}"
        )

        print(
            f"IDENTITY: "
            f"{row['identity_method']} / "
            f"{row['identity_confidence']}"
        )

        print(
            f"SUBJECTS: "
            f"{subjects[:10]}"
        )

        print(
            f"PEOPLE: "
            f"{people[:10]}"
        )

        print(
            f"PLACES: "
            f"{places[:10]}"
        )

        print(
            f"TIMES: "
            f"{times[:10]}"
        )

        if len(description) > 500:
            description = (
                description[:500]
                + "..."
            )

        print(
            f"DESCRIPTION: "
            f"{description}"
        )

    # -----------------------------------------------------------------------
    # 12. Build audit dataset
    # -----------------------------------------------------------------------

    print_section(
        "12. BUILDING AUDIT DATASET"
    )

    audit_rows = []

    for _, row in books.iterrows():

        description = clean_description(
            row["description"]
        )

        subjects = parse_collection(
            row["subjects"]
        )

        people = parse_collection(
            row["subject_people"]
        )

        places = parse_collection(
            row["subject_places"]
        )

        times = parse_collection(
            row["subject_times"]
        )

        audit_rows.append(
            {
                "canonical_book_id":
                    row["canonical_book_id"],
                "title":
                    row["title"],
                "author":
                    row["author"],
                "identity_method":
                    row["identity_method"],
                "identity_confidence":
                    row["identity_confidence"],
                "has_work_id":
                    bool(
                        not is_missing(
                            row[
                                "openlibrary_work_id"
                            ]
                        )
                    ),
                "description_words":
                    len(
                        description.split()
                    )
                    if description
                    else 0,
                "subject_count":
                    len(subjects),
                "subject_people_count":
                    len(people),
                "subject_places_count":
                    len(places),
                "subject_times_count":
                    len(times),
                "semantic_field_count":
                    sum(
                        [
                            bool(description),
                            len(subjects) > 0,
                            len(people) > 0,
                            len(places) > 0,
                            len(times) > 0,
                        ]
                    ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print_section(
        "AUDIT OUTPUT"
    )

    print(
        f"Saved semantic metadata audit:\n"
        f"{OUTPUT_PATH}"
    )

    print_section(
        "SEMANTIC METADATA AUDIT COMPLETE"
    )

    print(
        "Do not build embeddings yet."
    )

    print(
        "Use this output to decide how we construct "
        "the canonical semantic representation."
    )


if __name__ == "__main__":
    main()