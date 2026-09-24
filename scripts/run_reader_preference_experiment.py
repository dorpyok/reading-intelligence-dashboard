from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.analytics.metadata_normalization import (
    MODEL_NAME,
    build_book_text,
)
from src.analytics.preference_signals import derive_preference_signal
from src.analytics.reader_preference import build_reader_representation
from src.analytics.reading_behavior import derive_reading_behavior


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENRICHED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "book_clusters_enriched.csv"
)

READER_FILES = {
    "you": PROJECT_ROOT / "data" / "raw" / "goodreads_books.csv",
    "sarah": PROJECT_ROOT / "data" / "raw" / "sarah_books.csv",
    "shannon": PROJECT_ROOT / "data" / "raw" / "shannon_books.csv",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reader Representation experiment."
    )

    parser.add_argument(
        "--reader",
        choices=READER_FILES.keys(),
        required=True,
        help="Reader to evaluate.",
    )

    return parser.parse_args()


def parse_subjects(value: object) -> list[str]:
    """Parse Open Library subjects stored as a string representation of a list."""
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, list):
        return [str(item) for item in parsed]

    return []


def derive_reading_status(row: pd.Series) -> str:
    """
    Reconstruct canonical reading status from Goodreads evidence.

    Evidence hierarchy:
        1. did-not-finish
        2. currently-reading
        3. to-read
        4. date_read
        5. rating > 0
        6. unknown
    """
    shelves = str(row.get("shelves", "")).strip().lower()

    if "did-not-finish" in shelves:
        return "did_not_finish"

    if "currently-reading" in shelves:
        return "currently_reading"

    if "to-read" in shelves:
        return "to_read"

    date_read = row.get("date_read")

    if pd.notna(date_read) and str(date_read).strip():
        return "read"

    rating = row.get("rating")

    if pd.notna(rating):
        try:
            if float(rating) > 0:
                return "read"
        except (TypeError, ValueError):
            pass

    return "unknown"

def load_experiment_data(reader: str) -> pd.DataFrame:
    """Load enriched books and join them to one reader's Goodreads data."""
    enriched = pd.read_csv(ENRICHED_FILE)
    reader_file = READER_FILES[reader]
    goodreads = pd.read_csv(reader_file)

    print(f"Reader: {reader}")
    print(f"Enriched books loaded: {len(enriched)}")
    print(f"Reader books loaded: {len(goodreads)}")

    # Book IDs are identifiers, not numeric measures.
    # Normalize them to strings before joining so that Goodreads
    # exports with numeric-looking IDs join consistently with the
    # enriched dataset.
    enriched["source_book_id"] = (
        enriched["source_book_id"]
        .astype("string")
        .str.strip()
    )

    goodreads["source_book_id"] = (
        goodreads["source_book_id"]
        .astype("string")
        .str.strip()
    )

    goodreads["rating"] = pd.to_numeric(
        goodreads["user_rating"],
        errors="coerce",
    )

    # Goodreads uses 0 to mean unrated.
    goodreads["rating"] = goodreads["rating"].replace(0, np.nan)

    goodreads["reading_status"] = goodreads.apply(
        derive_reading_status,
        axis=1,
    )

    merged = enriched.merge(
        goodreads[
            [
                "source_book_id",
                "rating",
                "reading_status",
                "shelves",
                "date_read",
            ]
        ],
        on="source_book_id",
        how="inner",
        suffixes=("", "_goodreads"),
    )

    print(f"Books available for experiment: {len(merged)}")

    return merged


def build_semantic_representations(
    df: pd.DataFrame,
) -> list[str]:
    """Build semantic text representations for every book."""
    representations = []

    for _, row in df.iterrows():
        subjects = parse_subjects(row.get("subjects"))

        text = build_book_text(
            title=str(row.get("title", "")),
            author=str(row.get("author", "")),
            subjects=subjects,
            description=str(row.get("description", "")),
        )

        representations.append(text)

    return representations


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Generate normalized semantic embeddings."""
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings)


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add preference and reading-behavior evidence."""
    result = df.copy()

    result["preference_signal"] = result.apply(
        derive_preference_signal,
        axis=1,
    )

    result["reading_behavior"] = result.apply(
        derive_reading_behavior,
        axis=1,
    )

    return result


def calculate_similarities(
    embeddings: np.ndarray,
    reader_vector: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity between books and a reader vector."""
    return cosine_similarity(
        embeddings,
        reader_vector.reshape(1, -1),
    ).ravel()


def summarize(
    df: pd.DataFrame,
    similarities: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Return descriptive similarity statistics by evidence group."""
    result = df.copy()
    result["similarity"] = similarities

    groups = {
        "positive": result["preference_signal"].eq("positive"),
        "negative": result["preference_signal"].eq("negative"),
        "read_unrated": (
            result["reading_behavior"].eq("read")
            & result["rating"].isna()
        ),
        "three_star": result["rating"].eq(3),
    }

    summary = {}

    for label, mask in groups.items():
        values = result.loc[mask, "similarity"]

        if values.empty:
            summary[label] = {
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
            }
        else:
            summary[label] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
            }

    return summary


def print_top_books(
    df: pd.DataFrame,
    similarities: np.ndarray,
    label: str,
    top_n: int = 10,
) -> None:
    """Print the top books for one reader representation."""
    result = df.copy()
    result["similarity"] = similarities

    result = result.sort_values(
        "similarity",
        ascending=False,
    )

    print()
    print(label)
    print("-" * 70)

    for _, row in result.head(top_n).iterrows():
        rating = row["rating"]

        if pd.isna(rating):
            rating_display = "unrated"
        else:
            rating_display = str(int(rating))

        print(
            f"{row['title']} | "
            f"rating={rating_display} | "
            f"status={row['reading_status']} | "
            f"similarity={row['similarity']:.4f}"
        )


def main() -> None:
    args = parse_arguments()

    reader = args.reader

    print()
    print("Reader Representation Experiment")
    print("=" * 70)

    df = load_experiment_data(reader)

    print()
    print("Evidence counts")
    print("-" * 70)
    print(df["preference_signal"].value_counts())
    print()
    print(df["reading_behavior"].value_counts())

    print()
    print("Building semantic representations...")

    texts = build_semantic_representations(df)

    print(f"Semantic representations created: {len(texts)}")

    print()
    print("Generating embeddings...")
    print(f"Embedding model: {MODEL_NAME}")

    embeddings = generate_embeddings(texts)

    print(f"Embedding matrix shape: {embeddings.shape}")

    print()
    print("Building reader representations...")

    representations = build_reader_representation(
        embeddings=embeddings,
        preference_signals=df["preference_signal"],
        reading_behavior=df["reading_behavior"],
    )

    preference_vector = representations["preference_vector"]
    exposure_vector = representations["exposure_vector"]

    print(f"Preference vector shape: {preference_vector.shape}")
    print(f"Exposure vector shape: {exposure_vector.shape}")

    preference_similarity = calculate_similarities(
        embeddings,
        preference_vector,
    )

    exposure_similarity = calculate_similarities(
        embeddings,
        exposure_vector,
    )

    preference_summary = summarize(
        df,
        preference_similarity,
    )

    exposure_summary = summarize(
        df,
        exposure_similarity,
    )

    print_top_books(
        df,
        preference_similarity,
        "EXPLICIT PREFERENCE",
    )

    print_top_books(
        df,
        exposure_similarity,
        "READING EXPOSURE",
    )

    print()
    print("SIMILARITY SUMMARY")
    print("=" * 70)

    for label in (
        "positive",
        "read_unrated",
        "three_star",
        "negative",
    ):
        pref = preference_summary[label]
        exposure = exposure_summary[label]

        print()
        print(label)

        print(
            f"  n={pref['n']}"
        )

        print(
            f"  Preference: "
            f"mean={pref['mean']:.4f}, "
            f"median={pref['median']:.4f}"
        )

        print(
            f"  Exposure:   "
            f"mean={exposure['mean']:.4f}, "
            f"median={exposure['median']:.4f}"
        )

        if not np.isnan(pref["mean"]):
            print(
                f"  Difference: "
                f"{exposure['mean'] - pref['mean']:+.4f}"
            )

    print()
    print("=" * 70)
    print(f"Experiment complete for {reader}.")
    print()
    print(
        "This is an exploratory in-sample comparison."
    )
    print(
        "No exposure weight has been assigned and no "
        "held-out performance has been evaluated."
    )


if __name__ == "__main__":
    main()