import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load Model B directly from its file path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_B_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_model_b_multi_interest.py"
)

spec = importlib.util.spec_from_file_location(
    "run_model_b_multi_interest",
    MODEL_B_PATH,
)

model_b = importlib.util.module_from_spec(spec)

spec.loader.exec_module(model_b)


# ---------------------------------------------------------------------------
# K SELECTION TESTS
# ---------------------------------------------------------------------------

def test_choose_k_returns_valid_k():
    """K selection should return a valid number of clusters."""

    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(30, 10)
    )

    best_k, scores = model_b.choose_k(
        embeddings,
        min_k=2,
        max_k=5,
        random_state=42,
    )

    assert 2 <= best_k <= 5

    assert isinstance(
        scores,
        pd.DataFrame,
    )

    assert "k" in scores.columns

    assert "silhouette_score" in scores.columns


def test_choose_k_scores_all_requested_values():
    """K selection should evaluate every requested K."""

    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(40, 10)
    )

    best_k, scores = model_b.choose_k(
        embeddings,
        min_k=2,
        max_k=5,
        random_state=42,
    )

    assert set(
        scores["k"]
    ) == {2, 3, 4, 5}

    assert (
        scores["silhouette_score"]
        .notna()
        .all()
    )

    assert 2 <= best_k <= 5


# ---------------------------------------------------------------------------
# INTEREST MODEL TESTS
# ---------------------------------------------------------------------------

def test_fit_interest_model_returns_normalized_centroids():
    """Interest centroids should be normalized."""

    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(30, 10)
    )

    _, centroids = model_b.fit_interest_model(
        embeddings,
        k=3,
        random_state=42,
    )

    assert centroids.shape == (
        3,
        10,
    )

    norms = np.linalg.norm(
        centroids,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-6,
    )


def test_multi_interest_representation_can_capture_multiple_interests():
    """
    A synthetic example showing that two distinct interest clusters
    can be represented separately.
    """

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.1, 0.9],
        ]
    )

    _, centroids = model_b.fit_interest_model(
        embeddings,
        k=2,
        random_state=42,
    )

    candidate_a = np.array(
        [
            [0.98, 0.02],
        ]
    )

    candidate_b = np.array(
        [
            [0.02, 0.98],
        ]
    )

    from sklearn.metrics.pairwise import cosine_similarity

    score_a = cosine_similarity(
        candidate_a,
        centroids,
    ).max()

    score_b = cosine_similarity(
        candidate_b,
        centroids,
    ).max()

    assert score_a > 0.9

    assert score_b > 0.9


# ---------------------------------------------------------------------------
# RANKING TESTS
# ---------------------------------------------------------------------------

def test_rank_candidates_multi_interest():
    """Candidates should be ranked by their closest interest."""

    interest_centroids = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )

    ranked, scores = (
        model_b.rank_candidates_multi_interest(
            interest_centroids,
            embeddings,
            excluded_indices=set(),
        )
    )

    assert len(ranked) == 4

    assert scores.shape == (
        4,
    )

    assert set(
        ranked[:2]
    ) == {0, 1}


def test_rank_candidates_excludes_training_books():
    """Training positives should not appear in recommendations."""

    interest_centroids = np.array(
        [
            [1.0, 0.0],
        ]
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ]
    )

    ranked, scores = (
        model_b.rank_candidates_multi_interest(
            interest_centroids,
            embeddings,
            excluded_indices={0},
        )
    )

    assert ranked[0] != 0

    assert np.isneginf(
        scores[0]
    )


# ---------------------------------------------------------------------------
# RECALL TESTS
# ---------------------------------------------------------------------------

def test_calculate_recall_at_k():
    """Recall@K should correctly identify held-out positives."""

    ranked_indices = np.array(
        [
            4,
            2,
            7,
            1,
            5,
            3,
            6,
            0,
        ]
    )

    held_out = {
        2,
        5,
    }

    recall_at_3 = (
        model_b.calculate_recall_at_k(
            ranked_indices,
            held_out,
            3,
        )
    )

    assert np.isclose(
        recall_at_3,
        0.5,
    )


def test_calculate_recall_at_k_full_recall():
    """Recall should reach 1.0 when all held-out books are retrieved."""

    ranked_indices = np.array(
        [
            1,
            2,
            3,
            4,
            5,
        ]
    )

    held_out = {
        1,
        2,
    }

    recall = model_b.calculate_recall_at_k(
        ranked_indices,
        held_out,
        5,
    )

    assert np.isclose(
        recall,
        1.0,
    )


def test_calculate_recall_at_k_empty_set():
    """Empty held-out sets should return NaN."""

    ranked_indices = np.array(
        [
            1,
            2,
            3,
        ]
    )

    recall = model_b.calculate_recall_at_k(
        ranked_indices,
        set(),
        3,
    )

    assert np.isnan(
        recall
    )


# ---------------------------------------------------------------------------
# HELD-OUT SIMILARITY TESTS
# ---------------------------------------------------------------------------

def test_calculate_held_out_scores():
    """Held-out books should receive their best interest similarity."""

    interest_centroids = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ]
    )

    held_out_indices = np.array(
        [
            0,
            1,
            2,
        ]
    )

    scores = model_b.calculate_held_out_scores(
        interest_centroids,
        embeddings,
        held_out_indices,
    )

    assert len(scores) == 3

    assert np.isclose(
        scores[0],
        1.0,
    )

    assert np.isclose(
        scores[1],
        1.0,
    )

    assert scores[2] > 0.8


# ---------------------------------------------------------------------------
# RANK PERCENTILE TESTS
# ---------------------------------------------------------------------------

def test_calculate_rank_percentiles():
    """Higher-ranked books should have higher percentile scores."""

    ranked_indices = np.array(
        [
            10,
            20,
            30,
            40,
        ]
    )

    held_out_indices = np.array(
        [
            10,
            40,
        ]
    )

    percentiles = (
        model_b.calculate_rank_percentiles(
            ranked_indices,
            held_out_indices,
        )
    )

    assert len(percentiles) == 2

    assert percentiles[0] > percentiles[1]

    assert np.isclose(
        percentiles[0],
        1.0,
    )


def test_calculate_rank_percentiles_empty():
    """Empty held-out sets should return an empty list."""

    ranked_indices = np.array(
        [
            1,
            2,
            3,
        ]
    )

    percentiles = (
        model_b.calculate_rank_percentiles(
            ranked_indices,
            np.array([]),
        )
    )

    assert percentiles == []


# ---------------------------------------------------------------------------
# SUMMARY TESTS
# ---------------------------------------------------------------------------

def test_summarize_results():
    """Summary should calculate mean and standard deviation across folds."""

    fold_results = pd.DataFrame(
        {
            "reader": [
                "you",
                "you",
                "you",
            ],
            "fold": [
                1,
                2,
                3,
            ],
            "training_positive_count": [
                70,
                70,
                70,
            ],
            "held_out_positive_count": [
                20,
                20,
                20,
            ],
            "selected_k": [
                2,
                3,
                4,
            ],
            "recall_at_10": [
                0.1,
                0.2,
                0.3,
            ],
            "recall_at_25": [
                0.2,
                0.3,
                0.4,
            ],
            "recall_at_50": [
                0.4,
                0.5,
                0.6,
            ],
            "mean_held_out_similarity": [
                0.6,
                0.7,
                0.8,
            ],
            "median_held_out_similarity": [
                0.6,
                0.7,
                0.8,
            ],
            "mean_rank_percentile": [
                0.7,
                0.8,
                0.9,
            ],
            "median_rank_percentile": [
                0.7,
                0.8,
                0.9,
            ],
        }
    )

    summary = model_b.summarize_results(
        fold_results
    )

    assert isinstance(
        summary,
        pd.DataFrame,
    )

    assert len(summary) == 1

    # Use np.isclose because floating-point arithmetic can produce
    # 0.19999999999999998 instead of exactly 0.2.
    assert np.isclose(
        summary[
            (
                "recall_at_10",
                "mean",
            )
        ].iloc[0],
        0.2,
    )

    assert np.isclose(
        summary[
            (
                "selected_k",
                "mean",
            )
        ].iloc[0],
        3.0,
    )