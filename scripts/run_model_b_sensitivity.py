from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.metadata_normalization import build_book_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

MIN_K = 2
MAX_K = 10

N_SPLITS = 5
RANDOM_STATE = 42

POSITIVE_RATING_THRESHOLD = 4

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_experiments"
    / "model_b_sensitivity"
)

READERS = {
    "you": PROJECT_ROOT / "data" / "raw" / "you_books_enriched.csv",
    "sarah": PROJECT_ROOT / "data" / "raw" / "sarah_books_enriched.csv",
    "shannon": PROJECT_ROOT / "data" / "raw" / "shannon_books_enriched.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_subjects(value) -> list[str]:
    """
    Convert the stored Open Library subjects representation into a list.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    text = str(value).strip()

    if not text:
        return []

    try:
        import ast

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    except (ValueError, SyntaxError):
        pass

    return [text]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    """

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_similarity_matrix(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between every embedding and every centroid.

    Returns:
        Array with shape:
        (number_of_books, number_of_centroids)
    """

    embeddings_norm = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    centroids_norm = centroids / np.linalg.norm(
        centroids,
        axis=1,
        keepdims=True,
    )

    return embeddings_norm @ centroids_norm.T


def calculate_recall(
    ranked_indices: np.ndarray,
    held_out_indices: set[int],
    k: int,
) -> float:
    """
    Calculate Recall@K for held-out positive books.
    """

    if not held_out_indices:
        return np.nan

    top_k = ranked_indices[:k]

    hits = sum(
        1
        for index in top_k
        if index in held_out_indices
    )

    return hits / len(held_out_indices)


def calculate_rank_percentiles(
    ranked_indices: np.ndarray,
    held_out_indices: set[int],
) -> list[float]:
    """
    Calculate rank percentile for each held-out positive book.

    A value closer to 1 means the book appeared closer to the top
    of the recommendation ranking.
    """

    if not held_out_indices:
        return []

    total_candidates = len(ranked_indices)

    rank_lookup = {
        book_index: rank
        for rank, book_index in enumerate(ranked_indices, start=1)
    }

    percentiles = []

    for book_index in held_out_indices:
        rank = rank_lookup[book_index]

        percentile = 1 - ((rank - 1) / total_candidates)

        percentiles.append(percentile)

    return percentiles


# ---------------------------------------------------------------------------
# Book text
# ---------------------------------------------------------------------------

def build_reader_texts(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Build the canonical semantic representation used by Model B.

    This intentionally uses the same metadata_normalization function
    as the production recommendation model.
    """

    texts = []

    for _, row in dataframe.iterrows():

        title = row.get("title", "")
        author = row.get("author", "")

        subjects = parse_subjects(
            row.get("subjects", "")
        )

        description = row.get(
            "description",
            "",
        )

        text = build_book_text(
            title=title,
            author=author,
            subjects=subjects,
            description=description,
        )

        texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def cluster_positive_books(
    embeddings: np.ndarray,
    positive_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster positive books and return:

        labels
        centroids
        silhouette score
    """

    positive_embeddings = embeddings[positive_indices]

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(
        positive_embeddings
    )

    centroids = model.cluster_centers_

    # Normalize centroids for cosine-based recommendation.
    centroids = np.array(
        [
            normalize_vector(centroid)
            for centroid in centroids
        ]
    )

    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(
            positive_embeddings,
            labels,
            metric="cosine",
        )
    else:
        silhouette = np.nan

    return labels, centroids, silhouette


# ---------------------------------------------------------------------------
# Recommendation evaluation
# ---------------------------------------------------------------------------

def evaluate_k(
    embeddings: np.ndarray,
    positive_indices: np.ndarray,
    train_indices: np.ndarray,
    held_out_indices: np.ndarray,
    k: int,
) -> dict:
    """
    Train a multi-interest representation using exactly k clusters
    and evaluate retrieval of held-out positive books.
    """

    train_positive_indices = positive_indices[
        np.isin(
            positive_indices,
            train_indices,
        )
    ]

    held_out_positive_indices = positive_indices[
        np.isin(
            positive_indices,
            held_out_indices,
        )
    ]

    if len(train_positive_indices) < k:
        return {}

    # ---------------------------------------------------------------
    # Cluster training positives
    # ---------------------------------------------------------------

    labels, centroids, silhouette = cluster_positive_books(
        embeddings=embeddings,
        positive_indices=train_positive_indices,
        k=k,
    )

    # ---------------------------------------------------------------
    # Cluster size statistics
    # ---------------------------------------------------------------

    cluster_sizes = np.bincount(
        labels,
        minlength=k,
    )

    # ---------------------------------------------------------------
    # Score every book against every interest centroid
    # ---------------------------------------------------------------

    similarities = cosine_similarity_matrix(
        embeddings,
        centroids,
    )

    # Each book receives the similarity of its closest interest.
    candidate_scores = similarities.max(
        axis=1
    )

    # ---------------------------------------------------------------
    # Exclude books used to train the interest representation
    # ---------------------------------------------------------------

    ranking_mask = np.ones(
        len(embeddings),
        dtype=bool,
    )

    ranking_mask[
        train_positive_indices
    ] = False

    candidate_indices = np.where(
        ranking_mask
    )[0]

    ranked_indices = candidate_indices[
        np.argsort(
            candidate_scores[candidate_indices]
        )[::-1]
    ]

    held_out_set = set(
        held_out_positive_indices.tolist()
    )

    # ---------------------------------------------------------------
    # Recall
    # ---------------------------------------------------------------

    recall_10 = calculate_recall(
        ranked_indices,
        held_out_set,
        10,
    )

    recall_25 = calculate_recall(
        ranked_indices,
        held_out_set,
        25,
    )

    recall_50 = calculate_recall(
        ranked_indices,
        held_out_set,
        50,
    )

    # ---------------------------------------------------------------
    # Held-out similarity
    # ---------------------------------------------------------------

    held_out_similarities = (
        candidate_scores[
            held_out_positive_indices
        ]
    )

    if len(held_out_similarities):

        mean_similarity = float(
            np.mean(
                held_out_similarities
            )
        )

        median_similarity = float(
            np.median(
                held_out_similarities
            )
        )

    else:

        mean_similarity = np.nan
        median_similarity = np.nan

    # ---------------------------------------------------------------
    # Rank percentile
    # ---------------------------------------------------------------

    rank_percentiles = calculate_rank_percentiles(
        ranked_indices,
        held_out_set,
    )

    if rank_percentiles:

        mean_rank_percentile = float(
            np.mean(
                rank_percentiles
            )
        )

        median_rank_percentile = float(
            np.median(
                rank_percentiles
            )
        )

    else:

        mean_rank_percentile = np.nan
        median_rank_percentile = np.nan

    # ---------------------------------------------------------------
    # Centroid separation
    # ---------------------------------------------------------------

    centroid_similarity_matrix = (
        centroids @ centroids.T
    )

    upper_triangle = centroid_similarity_matrix[
        np.triu_indices(
            k,
            k=1,
        )
    ]

    if len(upper_triangle):

        mean_centroid_similarity = float(
            np.mean(
                upper_triangle
            )
        )

        max_centroid_similarity = float(
            np.max(
                upper_triangle
            )
        )

    else:

        mean_centroid_similarity = np.nan
        max_centroid_similarity = np.nan

    # ---------------------------------------------------------------
    # Return results
    # ---------------------------------------------------------------

    return {
        "k": k,

        "silhouette": silhouette,

        "min_cluster_size": int(
            cluster_sizes.min()
        ),

        "max_cluster_size": int(
            cluster_sizes.max()
        ),

        "mean_cluster_size": float(
            cluster_sizes.mean()
        ),

        "cluster_size_std": float(
            cluster_sizes.std()
        ),

        "mean_centroid_similarity":
            mean_centroid_similarity,

        "max_centroid_similarity":
            max_centroid_similarity,

        "recall_at_10":
            recall_10,

        "recall_at_25":
            recall_25,

        "recall_at_50":
            recall_50,

        "mean_held_out_similarity":
            mean_similarity,

        "median_held_out_similarity":
            median_similarity,

        "mean_rank_percentile":
            mean_rank_percentile,

        "median_rank_percentile":
            median_rank_percentile,

        "training_positive_books":
            len(train_positive_indices),

        "held_out_positive_books":
            len(held_out_positive_indices),
    }


# ---------------------------------------------------------------------------
# Reader experiment
# ---------------------------------------------------------------------------

def run_reader_experiment(
    reader_name: str,
    input_path: Path,
    model: SentenceTransformer,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print("=" * 70)
    print(reader_name.upper())
    print("=" * 70)

    print(f"Reading: {input_path}")

    dataframe = pd.read_csv(
        input_path
    )

    dataframe["user_rating"] = pd.to_numeric(
        dataframe["user_rating"],
        errors="coerce",
    )

    positive_mask = (
        dataframe["user_rating"]
        >= POSITIVE_RATING_THRESHOLD
    )

    positive_indices = np.where(
        positive_mask
    )[0]

    print(
        f"Books: {len(dataframe):,}"
    )

    print(
        f"Positive books: "
        f"{len(positive_indices):,}"
    )

    if len(positive_indices) < MIN_K:

        raise ValueError(
            f"{reader_name} has only "
            f"{len(positive_indices)} positive books. "
            f"Cannot evaluate k={MIN_K}."
        )

    # ---------------------------------------------------------------
    # Build semantic representation
    # ---------------------------------------------------------------

    print()
    print("Building canonical book representations...")

    texts = build_reader_texts(
        dataframe
    )

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings
    )

    # ---------------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------------

    n_splits = min(
        N_SPLITS,
        len(positive_indices),
    )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold_number, (
        train_position_indices,
        held_out_position_indices,
    ) in enumerate(
        kfold.split(positive_indices),
        start=1,
    ):

        train_indices = positive_indices[
            train_position_indices
        ]

        held_out_indices = positive_indices[
            held_out_position_indices
        ]

        print()
        print(
            f"Fold {fold_number}/{n_splits}"
        )

        print(
            f"  Training positives: "
            f"{len(train_indices)}"
        )

        print(
            f"  Held-out positives: "
            f"{len(held_out_indices)}"
        )

        for k in range(
            MIN_K,
            MAX_K + 1,
        ):

            if len(train_indices) < k:
                continue

            print(
                f"  Evaluating k={k}..."
            )

            result = evaluate_k(
                embeddings=embeddings,
                positive_indices=positive_indices,
                train_indices=train_indices,
                held_out_indices=held_out_indices,
                k=k,
            )

            if not result:
                continue

            result["reader"] = reader_name
            result["fold"] = fold_number

            fold_results.append(
                result
            )

    fold_dataframe = pd.DataFrame(
        fold_results
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_dataframe = (
        fold_dataframe
        .groupby(
            ["reader", "k"],
            as_index=False,
        )
        .agg(
            silhouette_mean=(
                "silhouette",
                "mean",
            ),
            silhouette_std=(
                "silhouette",
                "std",
            ),
            min_cluster_size_mean=(
                "min_cluster_size",
                "mean",
            ),
            max_cluster_size_mean=(
                "max_cluster_size",
                "mean",
            ),
            mean_cluster_size=(
                "mean_cluster_size",
                "mean",
            ),
            cluster_size_std_mean=(
                "cluster_size_std",
                "mean",
            ),
            mean_centroid_similarity=(
                "mean_centroid_similarity",
                "mean",
            ),
            max_centroid_similarity=(
                "max_centroid_similarity",
                "mean",
            ),
            recall_at_10_mean=(
                "recall_at_10",
                "mean",
            ),
            recall_at_10_std=(
                "recall_at_10",
                "std",
            ),
            recall_at_25_mean=(
                "recall_at_25",
                "mean",
            ),
            recall_at_25_std=(
                "recall_at_25",
                "std",
            ),
            recall_at_50_mean=(
                "recall_at_50",
                "mean",
            ),
            recall_at_50_std=(
                "recall_at_50",
                "std",
            ),
            held_out_similarity_mean=(
                "mean_held_out_similarity",
                "mean",
            ),
            held_out_similarity_std=(
                "mean_held_out_similarity",
                "std",
            ),
            median_held_out_similarity=(
                "median_held_out_similarity",
                "mean",
            ),
            rank_percentile_mean=(
                "mean_rank_percentile",
                "mean",
            ),
            rank_percentile_std=(
                "mean_rank_percentile",
                "std",
            ),
            median_rank_percentile=(
                "median_rank_percentile",
                "mean",
            ),
        )
    )

    return (
        fold_dataframe,
        summary_dataframe,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("MODEL B SENSITIVITY EXPERIMENT")
    print("=" * 70)

    print()
    print(
        f"Testing k={MIN_K} through k={MAX_K}"
    )

    print(
        "Production Model B will NOT be modified."
    )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    # ---------------------------------------------------------------
    # Load model once
    # ---------------------------------------------------------------

    print()
    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    all_fold_results = []
    all_summary_results = []

    # ---------------------------------------------------------------
    # Run each reader
    # ---------------------------------------------------------------

    for reader_name, input_path in READERS.items():

        if not input_path.exists():

            print()
            print(
                f"WARNING: Missing file for "
                f"{reader_name}:"
            )

            print(input_path)

            continue

        fold_dataframe, summary_dataframe = (
            run_reader_experiment(
                reader_name=reader_name,
                input_path=input_path,
                model=model,
            )
        )

        # Save reader-specific outputs.

        reader_fold_path = (
            OUTPUT_DIR
            / f"{reader_name}_fold_results.csv"
        )

        reader_summary_path = (
            OUTPUT_DIR
            / f"{reader_name}_summary.csv"
        )

        fold_dataframe.to_csv(
            reader_fold_path,
            index=False,
        )

        summary_dataframe.to_csv(
            reader_summary_path,
            index=False,
        )

        print()
        print(
            f"Saved: {reader_fold_path}"
        )

        print(
            f"Saved: {reader_summary_path}"
        )

        all_fold_results.append(
            fold_dataframe
        )

        all_summary_results.append(
            summary_dataframe
        )

    # ---------------------------------------------------------------
    # Combined outputs
    # ---------------------------------------------------------------

    if all_fold_results:

        combined_fold = pd.concat(
            all_fold_results,
            ignore_index=True,
        )

        combined_fold_path = (
            OUTPUT_DIR
            / "all_readers_fold_results.csv"
        )

        combined_fold.to_csv(
            combined_fold_path,
            index=False,
        )

        print()
        print(
            f"Saved: {combined_fold_path}"
        )

    if all_summary_results:

        combined_summary = pd.concat(
            all_summary_results,
            ignore_index=True,
        )

        combined_summary_path = (
            OUTPUT_DIR
            / "all_readers_summary.csv"
        )

        combined_summary.to_csv(
            combined_summary_path,
            index=False,
        )

        print(
            f"Saved: {combined_summary_path}"
        )

        # -----------------------------------------------------------
        # Console summary
        # -----------------------------------------------------------

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        display_columns = [
            "reader",
            "k",
            "silhouette_mean",
            "min_cluster_size_mean",
            "max_cluster_size_mean",
            "recall_at_10_mean",
            "recall_at_25_mean",
            "recall_at_50_mean",
            "held_out_similarity_mean",
            "rank_percentile_mean",
        ]

        print()

        print(
            combined_summary[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}"
            )
        )

    print()
    print("=" * 70)
    print("SENSITIVITY EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results are in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()