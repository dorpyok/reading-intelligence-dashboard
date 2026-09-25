import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metadata_normalization import build_book_text


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

POSITIVE_RATING_MIN = 4

MIN_K = 2
MAX_K = 6


READER_PATHS = {
    "you": PROJECT_ROOT / "data" / "raw" / "you_books_enriched.csv",
    "sarah": PROJECT_ROOT / "data" / "raw" / "sarah_books_enriched.csv",
    "shannon": PROJECT_ROOT / "data" / "raw" / "shannon_books_enriched.csv",
}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def parse_subjects(value) -> list[str]:
    """Safely parse Open Library subjects stored as strings/lists."""

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

    except (ValueError, SyntaxError):
        pass

    return [value]


def load_reader_data(reader: str) -> pd.DataFrame:
    """Load one reader's enriched Goodreads dataset."""

    path = READER_PATHS[reader]

    if not path.exists():
        raise FileNotFoundError(
            f"Reader file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required = {
        "source_book_id",
        "title",
        "author",
        "description",
        "subjects",
        "user_rating",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    ratings = (
        pd.to_numeric(
            dataframe["user_rating"],
            errors="coerce",
        )
        .replace(0, np.nan)
    )

    dataframe["rating_numeric"] = ratings

    dataframe["is_positive"] = (
        ratings >= POSITIVE_RATING_MIN
    )

    return dataframe


# ---------------------------------------------------------------------------
# SEMANTIC REPRESENTATION
# ---------------------------------------------------------------------------

def build_book_texts(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Build semantic text representations for each book."""

    texts = []

    for _, row in dataframe.iterrows():

        texts.append(
            build_book_text(
                title=row.get("title"),
                author=row.get("author"),
                subjects=parse_subjects(
                    row.get("subjects")
                ),
                description=row.get("description"),
            )
        )

    return texts


def generate_embeddings(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
    """Generate normalized sentence-transformer embeddings."""

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings)


# ---------------------------------------------------------------------------
# MULTI-INTEREST MODEL
# ---------------------------------------------------------------------------

def choose_k(
    positive_embeddings: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[int, pd.DataFrame]:
    """
    Select the number of interest clusters using silhouette score.

    K is evaluated only on the training positive books during CV.
    """

    n_samples = len(positive_embeddings)

    upper_k = min(
        max_k,
        n_samples - 1,
    )

    if upper_k < min_k:
        raise ValueError(
            f"Need at least {min_k + 1} positive books "
            "to evaluate multiple interests."
        )

    rows = []

    for k in range(min_k, upper_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )

        labels = model.fit_predict(
            positive_embeddings
        )

        score = silhouette_score(
            positive_embeddings,
            labels,
            metric="cosine",
        )

        rows.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

    scores = pd.DataFrame(rows)

    best_k = int(
        scores.loc[
            scores["silhouette_score"].idxmax(),
            "k",
        ]
    )

    return best_k, scores


def fit_interest_model(
    positive_embeddings: np.ndarray,
    k: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit KMeans and convert cluster centers into normalized
    interest vectors.
    """

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=20,
    )

    labels = model.fit_predict(
        positive_embeddings
    )

    centroids = model.cluster_centers_

    norms = np.linalg.norm(
        centroids,
        axis=1,
        keepdims=True,
    )

    centroids = centroids / np.clip(
        norms,
        1e-12,
        None,
    )

    return model, centroids


# ---------------------------------------------------------------------------
# RANKING
# ---------------------------------------------------------------------------

def rank_candidates_multi_interest(
    interest_centroids: np.ndarray,
    embeddings: np.ndarray,
    excluded_indices: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every candidate against every interest.

    Final score = similarity to the reader's closest interest.
    """

    similarities = cosine_similarity(
        embeddings,
        interest_centroids,
    )

    scores = similarities.max(axis=1)

    if excluded_indices:
        scores[list(excluded_indices)] = -np.inf

    ranked = np.argsort(scores)[::-1]

    return ranked, scores


def calculate_recall_at_k(
    ranked_indices: np.ndarray,
    held_out_indices: set[int],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if not held_out_indices:
        return np.nan

    return (
        len(
            set(
                ranked_indices[:k]
            ).intersection(
                held_out_indices
            )
        )
        / len(held_out_indices)
    )


def calculate_held_out_scores(
    interest_centroids: np.ndarray,
    embeddings: np.ndarray,
    held_out_indices: np.ndarray,
) -> list[float]:
    """Calculate each held-out book's best interest similarity."""

    if len(held_out_indices) == 0:
        return []

    similarities = cosine_similarity(
        embeddings[held_out_indices],
        interest_centroids,
    )

    return similarities.max(axis=1).tolist()


def calculate_rank_percentiles(
    ranked_indices: np.ndarray,
    held_out_indices: np.ndarray,
) -> list[float]:
    """
    Convert rank position into a percentile-like score.

    Higher = closer to the top of the ranking.
    """

    if len(held_out_indices) == 0:
        return []

    rank_lookup = {
        int(book_index): rank
        for rank, book_index in enumerate(
            ranked_indices
        )
    }

    total_candidates = len(ranked_indices)

    return [
        1
        - (
            rank_lookup[int(book_index)]
            / total_candidates
        )
        for book_index in held_out_indices
    ]


# ---------------------------------------------------------------------------
# CROSS-VALIDATED MODEL B
# ---------------------------------------------------------------------------

def run_model_b(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    reader_name: str,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the multi-interest recommendation model.

    Only positive books are used as preference examples.

    For each fold:

    1. Split positive books into train/test.
    2. Discover K interests from training positives.
    3. Build one centroid per interest.
    4. Rank every book against the closest interest.
    5. Measure retrieval of held-out positive books.
    """

    positive_indices = np.flatnonzero(
        dataframe["is_positive"].to_numpy()
    )

    if len(positive_indices) < n_splits + 1:
        raise ValueError(
            f"{reader_name} has only "
            f"{len(positive_indices)} positive books."
        )

    print()
    print("=" * 70)
    print(
        f"MODEL B — MULTI-INTEREST — {reader_name}"
    )
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
        f"Cross-validation folds: "
        f"{n_splits}"
    )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    k_selection_rows = []

    for fold_number, (
        train_positions,
        test_positions,
    ) in enumerate(
        kfold.split(positive_indices),
        start=1,
    ):

        training_indices = positive_indices[
            train_positions
        ]

        held_out_indices = positive_indices[
            test_positions
        ]

        training_embeddings = embeddings[
            training_indices
        ]

        # IMPORTANT:
        # K is selected only using the training positives.
        selected_k, k_scores = choose_k(
            training_embeddings,
            random_state=random_state,
        )

        k_scores["reader"] = reader_name
        k_scores["fold"] = fold_number

        k_selection_rows.extend(
            k_scores.to_dict("records")
        )

        # Fit the multi-interest representation.
        _, centroids = fit_interest_model(
            training_embeddings,
            selected_k,
            random_state=random_state,
        )

        # Rank all books against the closest interest.
        ranked_indices, _ = (
            rank_candidates_multi_interest(
                centroids,
                embeddings,
                set(
                    training_indices.tolist()
                ),
            )
        )

        held_out_set = set(
            held_out_indices.tolist()
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

        held_out_scores = (
            calculate_held_out_scores(
                centroids,
                embeddings,
                held_out_indices,
            )
        )

        rank_percentiles = (
            calculate_rank_percentiles(
                ranked_indices,
                held_out_indices,
            )
        )

        result = {
            "reader": reader_name,
            "fold": fold_number,
            "training_positive_count": len(
                training_indices
            ),
            "held_out_positive_count": len(
                held_out_indices
            ),
            "selected_k": selected_k,
            "recall_at_10": recall_at_10,
            "recall_at_25": recall_at_25,
            "recall_at_50": recall_at_50,
            "mean_held_out_similarity": np.mean(
                held_out_scores
            ),
            "median_held_out_similarity": np.median(
                held_out_scores
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
            f"k={selected_k}, "
            f"Recall@10={recall_at_10:.3f}, "
            f"Recall@25={recall_at_25:.3f}, "
            f"Recall@50={recall_at_50:.3f}, "
            f"Mean similarity="
            f"{result['mean_held_out_similarity']:.3f}"
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(k_selection_rows),
    )


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def summarize_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create mean/std summary statistics."""

    numeric_columns = [
        "selected_k",
        "recall_at_10",
        "recall_at_25",
        "recall_at_50",
        "mean_held_out_similarity",
        "median_held_out_similarity",
        "mean_rank_percentile",
        "median_rank_percentile",
    ]

    return (
        fold_results
        .groupby("reader")[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def make_reader_visual(
    reader: str,
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> int:
    """
    Create a 2D visualization of the reader's discovered
    positive preference interests.

    PCA is visualization only.

    The actual model operates in the full embedding space.
    """

    positive_indices = np.flatnonzero(
        dataframe["is_positive"].to_numpy()
    )

    positive_embeddings = embeddings[
        positive_indices
    ]

    selected_k, k_scores = choose_k(
        positive_embeddings,
        random_state=random_state,
    )

    _, centroids = fit_interest_model(
        positive_embeddings,
        selected_k,
        random_state=random_state,
    )

    labels = KMeans(
        n_clusters=selected_k,
        random_state=random_state,
        n_init=20,
    ).fit_predict(
        positive_embeddings
    )

    # PCA is used only for visualization.
    projection_model = PCA(
        n_components=2,
        random_state=random_state,
    )

    all_2d = projection_model.fit_transform(
        embeddings
    )

    positive_2d = all_2d[
        positive_indices
    ]

    centroid_2d = projection_model.transform(
        centroids
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    non_positive = np.ones(
        len(dataframe),
        dtype=bool,
    )

    non_positive[
        positive_indices
    ] = False

    ax.scatter(
        all_2d[non_positive, 0],
        all_2d[non_positive, 1],
        s=14,
        alpha=0.18,
        label="Other books",
    )

    for cluster_id in range(
        selected_k
    ):

        mask = (
            labels == cluster_id
        )

        ax.scatter(
            positive_2d[mask, 0],
            positive_2d[mask, 1],
            s=28,
            alpha=0.75,
            label=(
                f"Interest "
                f"{cluster_id + 1}"
            ),
        )

    ax.scatter(
        centroid_2d[:, 0],
        centroid_2d[:, 1],
        marker="X",
        s=180,
        edgecolors="black",
        linewidths=1.2,
        label="Interest centroids",
    )

    ax.set_title(
        f"{reader.title()} — positive "
        f"preference space "
        f"(multi-interest k={selected_k})"
    )

    ax.set_xlabel(
        "PCA dimension 1"
    )

    ax.set_ylabel(
        "PCA dimension 2"
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{reader}_positive_preference_space.png",
        dpi=180,
    )

    plt.close(fig)

    k_scores.to_csv(
        output_dir
        / f"{reader}_interest_k_selection.csv",
        index=False,
    )

    return selected_k


# ---------------------------------------------------------------------------
# MODEL A VS MODEL B COMPARISON
# ---------------------------------------------------------------------------
def make_model_comparison(output_dir: Path) -> None:
    """
    Compare Model A and Model B summary results.

    Model summary CSVs are written with pandas MultiIndex columns,
    such as:

        ("recall_at_10", "mean")
        ("recall_at_10", "std")

    This function explicitly reads both header rows so those
    columns are reconstructed correctly.
    """

    rows = []

    readers = [
        "you",
        "sarah",
        "shannon",
    ]

    for reader in readers:

        model_a_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "recommendation_experiments"
            / f"{reader}_model_a_summary.csv"
        )

        model_b_path = (
            output_dir
            / f"{reader}_model_b_summary.csv"
        )

        if not model_a_path.exists():
            print(
                f"Skipping {reader}: "
                f"Model A summary not found at "
                f"{model_a_path}"
            )
            continue

        if not model_b_path.exists():
            print(
                f"Skipping {reader}: "
                f"Model B summary not found at "
                f"{model_b_path}"
            )
            continue

        # IMPORTANT:
        # Both summary files have two header rows because the
        # aggregation creates MultiIndex columns.
        a = pd.read_csv(
            model_a_path,
            header=[0, 1],
        )

        b = pd.read_csv(
            model_b_path,
            header=[0, 1],
        )

        def mean_metric(
            frame: pd.DataFrame,
            metric: str,
        ) -> float:
            """
            Safely retrieve the mean value for a metric from
            a summary DataFrame with MultiIndex columns.
            """

            # Expected MultiIndex column.
            column = (
                metric,
                "mean",
            )

            if column in frame.columns:
                return float(
                    frame[column].iloc[0]
                )

            # Defensive fallback if the CSV was written with
            # flattened column names.
            flattened = (
                f"{metric}_mean"
            )

            if flattened in frame.columns:
                return float(
                    frame[flattened].iloc[0]
                )

            raise KeyError(
                f"Could not find mean value for "
                f"'{metric}'. Available columns: "
                f"{list(frame.columns)}"
            )

        metrics = [
            "recall_at_10",
            "recall_at_25",
            "recall_at_50",
            "mean_held_out_similarity",
            "mean_rank_percentile",
        ]

        row = {
            "reader": reader,
        }

        for metric in metrics:

            model_a_value = mean_metric(
                a,
                metric,
            )

            model_b_value = mean_metric(
                b,
                metric,
            )

            row[
                f"model_a_{metric}"
            ] = model_a_value

            row[
                f"model_b_{metric}"
            ] = model_b_value

            row[
                f"change_{metric}"
            ] = (
                model_b_value
                - model_a_value
            )

        rows.append(row)

    if not rows:
        print(
            "No Model A / Model B comparison "
            "could be generated."
        )
        return

    comparison = pd.DataFrame(rows)

    comparison_path = (
        output_dir
        / "model_a_vs_model_b_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL A VS MODEL B COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Recall@10 comparison
    # ---------------------------------------------------------------

    x = np.arange(
        len(comparison)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.bar(
        x - width / 2,
        comparison[
            "model_a_recall_at_10"
        ],
        width,
        label="Model A — single centroid",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "model_b_recall_at_10"
        ],
        width,
        label="Model B — multi-interest",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison["reader"].str.title()
    )

    ax.set_ylabel(
        "Recall@10"
    )

    ax.set_title(
        "Model A vs Model B — Recall@10"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "model_a_vs_model_b_recall_at_10.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Recall@25 comparison
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.bar(
        x - width / 2,
        comparison[
            "model_a_recall_at_25"
        ],
        width,
        label="Model A — single centroid",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "model_b_recall_at_25"
        ],
        width,
        label="Model B — multi-interest",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison["reader"].str.title()
    )

    ax.set_ylabel(
        "Recall@25"
    )

    ax.set_title(
        "Model A vs Model B — Recall@25"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "model_a_vs_model_b_recall_at_25.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Recall@50 comparison
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.bar(
        x - width / 2,
        comparison[
            "model_a_recall_at_50"
        ],
        width,
        label="Model A — single centroid",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "model_b_recall_at_50"
        ],
        width,
        label="Model B — multi-interest",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison["reader"].str.title()
    )

    ax.set_ylabel(
        "Recall@50"
    )

    ax.set_title(
        "Model A vs Model B — Recall@50"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "model_a_vs_model_b_recall_at_50.png",
        dpi=180,
    )

    plt.close(fig)

    print()
    print(
        f"Comparison saved to:"
    )
    print(
        comparison_path
    )
    
def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-interest "
            "Model B for book recommendations."
        )
    )

    parser.add_argument(
        "--reader",
        default="all",
        choices=[
            "you",
            "sarah",
            "shannon",
            "all",
        ],
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
    )

    args = parser.parse_args()

    readers = (
        [
            "you",
            "sarah",
            "shannon",
        ]
        if args.reader == "all"
        else [args.reader]
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "recommendation_experiments"
        / "model_b_cleaned"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []

    for reader in readers:

        dataframe = load_reader_data(
            reader
        )

        print()
        print(
            f"Loading {reader} data from:"
        )
        print(
            READER_PATHS[reader]
        )

        print(
            f"Loaded "
            f"{len(dataframe):,} books."
        )

        print()

        print(
            "Building semantic "
            f"representations with "
            f"{args.model_name}..."
        )

        texts = build_book_texts(
            dataframe
        )

        embeddings = generate_embeddings(
            texts,
            args.model_name,
        )

        print(
            f"Generated embeddings: "
            f"{embeddings.shape}"
        )

        fold_results, k_selection = (
            run_model_b(
                dataframe,
                embeddings,
                reader,
                n_splits=args.n_splits,
            )
        )

        summary = summarize_results(
            fold_results
        )

        all_summaries.append(
            summary
        )

        # -----------------------------------------------------------
        # Save fold-level results
        # -----------------------------------------------------------

        fold_results.to_csv(
            output_dir
            / f"{reader}_model_b_folds.csv",
            index=False,
        )

        # -----------------------------------------------------------
        # Save K-selection results
        # -----------------------------------------------------------

        k_selection.to_csv(
            output_dir
            / f"{reader}_model_b_k_selection.csv",
            index=False,
        )

        # -----------------------------------------------------------
        # Save summary
        # -----------------------------------------------------------

        summary.to_csv(
            output_dir
            / f"{reader}_model_b_summary.csv",
            index=False,
        )

        # -----------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------

        selected_k = make_reader_visual(
            reader,
            dataframe,
            embeddings,
            output_dir,
        )

        print()

        print(
            f"{reader.title()} "
            f"descriptive multi-interest "
            f"k: {selected_k}"
        )

    # ----------------------------------------------------------------
    # COMBINED SUMMARY
    # ----------------------------------------------------------------

    if all_summaries:

        combined = pd.concat(
            all_summaries,
            ignore_index=True,
        )

        combined.to_csv(
            output_dir
            / "model_b_all_readers_summary.csv",
            index=False,
        )

        print()

        print("=" * 70)
        print("MODEL B — SUMMARY")
        print("=" * 70)

        print(
            combined.to_string(
                index=False
            )
        )

    # ----------------------------------------------------------------
    # MODEL A VS MODEL B
    # ----------------------------------------------------------------

    make_model_comparison(
        output_dir
    )

    print()

    print(
        "Saved Model B outputs to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()