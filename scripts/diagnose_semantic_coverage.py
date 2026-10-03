from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEMANTIC_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_books.csv"
)

ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhood_assignments.csv"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_embeddings.npy"
)


def main() -> None:
    print("Loading semantic corpus...")
    books = pd.read_csv(SEMANTIC_BOOKS_PATH)

    print("Loading neighborhood assignments...")
    assignments = pd.read_csv(ASSIGNMENTS_PATH)

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)

    print()
    print("=" * 70)
    print("SEMANTIC COVERAGE DIAGNOSTIC")
    print("=" * 70)

    print(f"Books: {len(books):,}")
    print(f"Embeddings: {embeddings.shape}")

    # ------------------------------------------------------------------
    # Description coverage
    # ------------------------------------------------------------------

    semantic_text = books["semantic_text"].fillna("").astype(str)
    subjects = books["cleaned_subjects"].fillna("").astype(str)

    books["has_description_text"] = semantic_text.str.strip().ne("")
    books["has_subjects"] = subjects.str.strip().ne("[]")

    # Because semantic_text contains title + author + description + subjects,
    # use word count as a general content-length diagnostic.
    books["semantic_word_count"] = (
        books["semantic_text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print()
    print("CONTENT COVERAGE")
    print("-" * 70)

    print(
        f"Books with subjects: "
        f"{books['has_subjects'].sum():,} "
        f"({books['has_subjects'].mean():.1%})"
    )

    print(
        f"Books without subjects: "
        f"{(~books['has_subjects']).sum():,} "
        f"({(~books['has_subjects']).mean():.1%})"
    )

    print(
        f"Books with semantic text: "
        f"{books['has_description_text'].sum():,} "
        f"({books['has_description_text'].mean():.1%})"
    )

    print(
        f"Books without semantic text: "
        f"{(~books['has_description_text']).sum():,} "
        f"({(~books['has_description_text']).mean():.1%})"
    )

    # ------------------------------------------------------------------
    # Description/content length
    # ------------------------------------------------------------------

    print()
    print("SEMANTIC TEXT LENGTH")
    print("-" * 70)

    print(
        books.groupby("has_subjects")["semantic_word_count"]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string()
    )

    # ------------------------------------------------------------------
    # Attach neighborhood assignments
    # ------------------------------------------------------------------

    assignment_columns = [
        "canonical_book_id",
        "neighborhood_id",
        "neighborhood_probability",
        "is_noise",
    ]

    assignments = assignments[assignment_columns]

    books = books.merge(
        assignments,
        on="canonical_book_id",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Neighborhood assignment by subject availability
    # ------------------------------------------------------------------

    print()
    print("NEIGHBORHOOD ASSIGNMENT BY SUBJECT AVAILABILITY")
    print("-" * 70)

    neighborhood_summary = (
        books.groupby("has_subjects")
        .agg(
            books=("canonical_book_id", "count"),
            assigned_books=(
                "is_noise",
                lambda x: (~x).sum(),
            ),
            noise_books=("is_noise", "sum"),
            mean_probability=("neighborhood_probability", "mean"),
        )
    )

    neighborhood_summary["assigned_pct"] = (
        neighborhood_summary["assigned_books"]
        / neighborhood_summary["books"]
    )

    neighborhood_summary["noise_pct"] = (
        neighborhood_summary["noise_books"]
        / neighborhood_summary["books"]
    )

    print(neighborhood_summary.to_string())

    # ------------------------------------------------------------------
    # Empty-subject books specifically
    # ------------------------------------------------------------------

    no_subjects = books.loc[
        ~books["has_subjects"]
    ].copy()

    print()
    print("BOOKS WITHOUT SUBJECTS")
    print("-" * 70)

    print(f"Count: {len(no_subjects):,}")

    if len(no_subjects) > 0:
        print()
        print(
            no_subjects[
                [
                    "title",
                    "author",
                    "semantic_word_count",
                    "neighborhood_id",
                    "neighborhood_probability",
                    "is_noise",
                ]
            ]
            .sort_values(
                ["is_noise", "neighborhood_probability"],
                ascending=[True, False],
            )
            .head(30)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Empty-subject books that ARE assigned
    # ------------------------------------------------------------------

    assigned_no_subjects = no_subjects.loc[
        ~no_subjects["is_noise"]
    ]

    noise_no_subjects = no_subjects.loc[
        no_subjects["is_noise"]
    ]

    print()
    print("EMPTY-SUBJECT BOOKS: ASSIGNED VS NOISE")
    print("-" * 70)

    print(f"Assigned to a neighborhood: {len(assigned_no_subjects):,}")
    print(f"Classified as noise:        {len(noise_no_subjects):,}")

    if len(assigned_no_subjects) > 0:
        print()
        print("Assigned empty-subject books:")
        print(
            assigned_no_subjects[
                [
                    "title",
                    "author",
                    "semantic_word_count",
                    "neighborhood_id",
                    "neighborhood_probability",
                ]
            ]
            .sort_values(
                "neighborhood_probability",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Save diagnostic
    # ------------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "canonical"
        / "semantic_coverage_diagnostic.csv"
    )

    books.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print("Saved diagnostic:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()