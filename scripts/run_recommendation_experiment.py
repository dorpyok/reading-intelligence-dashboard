import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity

from src.analytics.metadata_normalization import build_book_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

POSITIVE_RATING_MIN = 4


def parse_subjects(value) -> list[str]:
    """
    Convert the stored Open Library subjects value into a list.

    Enriched CSVs may contain subjects stored as a string
    representation of a Python list.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    except Exception:
        pass

    return [value]


def load_reader_data(
    reader_file: Path,
) -> pd.DataFrame:
    """Load one reader's enriched book data."""

    if not reader_file.exists():
        raise FileNotFoundError(
            f"Reader file not found: {reader_file}"
        )

    dataframe = pd.read_csv(reader_file)

    required_columns = {
        "source_book_id",
        "title",
        "author",
        "description",
        "subjects",
        "user_rating",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return dataframe


def prepare_book_text(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Build the semantic text representation used by the project."""

    texts = []

    for _, row in dataframe.iterrows():

        subjects = parse_subjects(row.get("subjects"))

        text = build_book_text(
            title=row.get("title"),
            author=row.get("author"),
            subjects=subjects,
            description=row.get("description"),
        )

        texts.append(text)

    return texts


def generate_embeddings(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
    """Generate normalized semantic embeddings."""

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings)


def normalize_ratings(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Convert Goodreads ratings into numeric values."""

    ratings = pd.to_numeric(
        dataframe["user_rating"],
        errors="coerce",
    )

    ratings = ratings.replace(0, np.nan)

    return ratings


def get_positive_indices(
    ratings: pd.Series,
) -> np.ndarray:
    """Return row indices for explicitly positive books."""

    positive_mask = ratings >= POSITIVE_RATING_MIN

    return np.flatnonzero(
        positive_mask.to_numpy()
    )


def rank_candidates(
    preference_vector: np.ndarray,
    embeddings: np.ndarray,
    excluded_indices: set[int],
) -> np.ndarray:
    """
    Rank candidate books by cosine similarity to the
    reader's preference vector.

    Books used to construct the preference vector are
    excluded from the recommendation ranking.
    """

    scores = cosine_similarity(
        embeddings,
        preference_vector.reshape(1, -1),
    ).ravel()

    if excluded_indices:
        scores[list(excluded_indices)] = -np.inf

    return np.argsort(scores)[::-1]


def calculate_recall_at_k(
    ranked_indices: np.ndarray,
    held_out_indices: set[int],
    k: int,
) -> float:
    """Calculate Recall@K for held-out positive books."""

    if not held_out_indices:
        return np.nan

    top_k = set(ranked_indices[:k])

    hits = len(
        top_k.intersection(held_out_indices)
    )

    return hits / len(held_out_indices)


def calculate_held_out_similarities(
    preference_vector: np.ndarray,
    embeddings: np.ndarray,
    held_out_indices: np.ndarray,
) -> list[float]:
    """Calculate similarity between preference vector and held-out positives."""

    if len(held_out_indices) == 0:
        return []

    held_out_embeddings = embeddings[
        held_out_indices
    ]

    similarities = cosine_similarity(
        held_out_embeddings,
        preference_vector.reshape(1, -1),
    ).ravel()

    return similarities.tolist()


def calculate_rank_percentiles(
    ranked_indices: np.ndarray,
    held_out_indices: np.ndarray,
) -> list[float]:
    """
    Calculate the percentile position of each held-out
    positive within the candidate ranking.

    Higher percentile means the book appeared closer
    to the top of the recommendation list.
    """

    if len(held_out_indices) == 0:
        return []

    rank_lookup = {
        int(book_index): rank
        for rank, book_index in enumerate(ranked_indices)
    }

    total_candidates = len(ranked_indices)

    percentiles = []

    for book_index in held_out_indices:

        rank = rank_lookup[int(book_index)]

        percentile = 1 - (
            rank / total_candidates
        )

        percentiles.append(percentile)

    return percentiles


def run_model_a(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    reader_name: str,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run Model A:

        Explicit preference only.

    Positive preference evidence is defined as
    Goodreads rating >= 4.
    """

    ratings = normalize_ratings(dataframe)

    positive_indices = get_positive_indices(
        ratings
    )

    if len(positive_indices) < n_splits:
        raise ValueError(
            f"{reader_name} has only "
            f"{len(positive_indices)} positive books. "
            f"At least {n_splits} are required."
        )

    print()
    print("=" * 70)
    print(f"MODEL A — {reader_name}")
    print("=" * 70)
    print(
        f"Books: {len(dataframe):,}"
    )
    print(
        f"Positive books (rating >= 4): "
        f"{len(positive_indices):,}"
    )
    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )
    print(
        f"Cross-validation folds: {n_splits}"
    )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    for fold_number, (
        train_positions,
        test_positions,
    ) in enumerate(
        kfold.split(positive_indices),
        start=1,
    ):

        training_positive_indices = (
            positive_indices[train_positions]
        )

        held_out_positive_indices = (
            positive_indices[test_positions]
        )

        training_positive_embeddings = embeddings[
            training_positive_indices
        ]

        preference_vector = np.mean(training_positive_embeddings, axis=0)

        norm = np.linalg.norm(preference_vector)

        if norm == 0:
            raise ValueError("Preference vector has zero magnitude.")

        preference_vector = preference_vector / norm

        excluded_indices = set(
            training_positive_indices.tolist()
        )

        ranked_indices = rank_candidates(
            preference_vector=preference_vector,
            embeddings=embeddings,
            excluded_indices=excluded_indices,
        )

        held_out_set = set(
            held_out_positive_indices.tolist()
        )

        recall_at_10 = calculate_recall_at_k(
            ranked_indices,
            held_out_set,
            10,
        )

        recall_at_25 = calculate_recall_at_k(
            ranked_indices,
            held_out_set,
            25,
        )

        recall_at_50 = calculate_recall_at_k(
            ranked_indices,
            held_out_set,
            50,
        )

        similarities = (
            calculate_held_out_similarities(
                preference_vector,
                embeddings,
                held_out_positive_indices,
            )
        )

        rank_percentiles = (
            calculate_rank_percentiles(
                ranked_indices,
                held_out_positive_indices,
            )
        )

        result = {
            "reader": reader_name,
            "fold": fold_number,
            "training_positive_count": len(
                training_positive_indices
            ),
            "held_out_positive_count": len(
                held_out_positive_indices
            ),
            "recall_at_10": recall_at_10,
            "recall_at_25": recall_at_25,
            "recall_at_50": recall_at_50,
            "mean_held_out_similarity": np.mean(
                similarities
            ),
            "median_held_out_similarity": np.median(
                similarities
            ),
            "mean_rank_percentile": np.mean(
                rank_percentiles
            ),
            "median_rank_percentile": np.median(
                rank_percentiles
            ),
        }

        results.append(result)

        print(
            f"Fold {fold_number}: "
            f"Recall@10={recall_at_10:.3f}, "
            f"Recall@25={recall_at_25:.3f}, "
            f"Recall@50={recall_at_50:.3f}, "
            f"Mean similarity="
            f"{result['mean_held_out_similarity']:.3f}"
        )

    return pd.DataFrame(results)


def summarize_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold-level results."""

    numeric_columns = [
        "recall_at_10",
        "recall_at_25",
        "recall_at_50",
        "mean_held_out_similarity",
        "median_held_out_similarity",
        "mean_rank_percentile",
        "median_rank_percentile",
    ]

    summary = (
        fold_results
        .groupby("reader")[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print a readable experiment summary."""

    print()
    print("=" * 70)
    print("MODEL A — SUMMARY")
    print("=" * 70)

    for _, row in summary.iterrows():

        reader = row["reader"]

        print()
        print(reader)

        print(
            f"  Recall@10: "
            f"{row[('recall_at_10', 'mean')]:.3f} "
            f"± "
            f"{row[('recall_at_10', 'std')]:.3f}"
        )

        print(
            f"  Recall@25: "
            f"{row[('recall_at_25', 'mean')]:.3f} "
            f"± "
            f"{row[('recall_at_25', 'std')]:.3f}"
        )

        print(
            f"  Recall@50: "
            f"{row[('recall_at_50', 'mean')]:.3f} "
            f"± "
            f"{row[('recall_at_50', 'std')]:.3f}"
        )

        print(
            f"  Mean held-out similarity: "
            f"{row[('mean_held_out_similarity', 'mean')]:.3f}"
        )

        print(
            f"  Mean rank percentile: "
            f"{row[('mean_rank_percentile', 'mean')]:.3f}"
        )


def parse_reader_argument(
    value: str,
) -> tuple[str, Path]:
    """Parse reader name and enriched CSV path."""

    reader_paths = {
        "you": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "you_books_enriched.csv"
        ),
        "sarah": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "sarah_books_enriched.csv"
        ),
        "shannon": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "shannon_books_enriched.csv"
        ),
    }

    if value not in reader_paths:
        raise ValueError(
            f"Unknown reader '{value}'. "
            f"Choose from: "
            f"{', '.join(reader_paths)}"
        )

    return value, reader_paths[value]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Model A recommendation evaluation "
            "using explicit positive preference."
        )
    )

    parser.add_argument(
        "--reader",
        default="you",
        choices=[
            "you",
            "sarah",
            "shannon",
        ],
        help="Reader to evaluate.",
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Sentence-Transformer model.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of cross-validation folds.",
    )

    args = parser.parse_args()

    reader_name, reader_file = (
        parse_reader_argument(
            args.reader
        )
    )

    print(
        f"Loading {reader_name} data from:"
    )
    print(reader_file)

    dataframe = load_reader_data(
        reader_file
    )

    print(
        f"Loaded {len(dataframe):,} books."
    )

    print()
    print(
        f"Building semantic representations "
        f"with {args.model_name}..."
    )

    texts = prepare_book_text(
        dataframe
    )

    embeddings = generate_embeddings(
        texts=texts,
        model_name=args.model_name,
    )

    print(
        f"Generated embeddings: "
        f"{embeddings.shape}"
    )

    fold_results = run_model_a(
        dataframe=dataframe,
        embeddings=embeddings,
        reader_name=reader_name,
        n_splits=args.n_splits,
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "recommendation_experiments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_output = (
        output_dir
        / f"{reader_name}_model_a_folds.csv"
    )

    summary = summarize_results(
        fold_results
    )

    summary_output = (
        output_dir
        / f"{reader_name}_model_a_summary.csv"
    )

    fold_results.to_csv(
        fold_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print_summary(summary)

    print()
    print("Saved:")
    print(f"  {fold_output}")
    print(f"  {summary_output}")


if __name__ == "__main__":
    main()