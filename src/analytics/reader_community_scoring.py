from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_PRIOR_STRENGTH = 5.0
DEFAULT_PRIOR_PREFERENCE = 0.25
DEFAULT_MIN_AVOIDANCE_NEGATIVE_BOOKS = 2
DEFAULT_TOP_N = 10


# ============================================================================
# Validation
# ============================================================================

REQUIRED_COLUMNS = {
    "reader_id",
    "community_id",
    "book_count",
    "positive_book_count",
    "negative_book_count",
    "exposure_only_book_count",
}


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Validate required dataframe columns."""

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {sorted(missing)}"
        )


# ============================================================================
# Scoring
# ============================================================================


def score_reader_communities(
    dataframe: pd.DataFrame,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    prior_preference: float = DEFAULT_PRIOR_PREFERENCE,
    min_avoidance_negative_books: int = (
        DEFAULT_MIN_AVOIDANCE_NEGATIVE_BOOKS
    ),
) -> pd.DataFrame:
    """
    Score Reader × Community evidence.

    This is the public compatibility entry point used by
    scripts/run_reader_community_scoring.py.

    Reading DNA intentionally consists of separate signals:

    1. preference_strength
       Evidence that the reader likes the semantic community.

    2. avoidance_strength
       Evidence that negative preference outweighs positive preference.

    3. exploration_strength
       Evidence that the reader is actively exploring the community
       without established preference.

    4. evidence_strength
       Overall amount of actionable evidence.

    These signals are NOT combined into one opaque score.
    """

    require_columns(
        dataframe,
        REQUIRED_COLUMNS,
        "reader community evidence",
    )

    if prior_strength <= 0:
        raise ValueError(
            "prior_strength must be greater than zero."
        )

    if not 0 <= prior_preference <= 1:
        raise ValueError(
            "prior_preference must be between 0 and 1."
        )

    if min_avoidance_negative_books < 1:
        raise ValueError(
            "min_avoidance_negative_books must be at least 1."
        )

    result = dataframe.copy()

    # ------------------------------------------------------------------------
    # Normalize numeric inputs
    # ------------------------------------------------------------------------

    numeric_columns = [
        "book_count",
        "positive_book_count",
        "negative_book_count",
        "exposure_only_book_count",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0)

        result[column] = result[column].clip(
            lower=0
        )

    # ------------------------------------------------------------------------
    # Evidence counts
    # ------------------------------------------------------------------------

    result["preference_evidence_count"] = (
        result["positive_book_count"]
        + result["negative_book_count"]
    )

    result["actionable_evidence_count"] = (
        result["preference_evidence_count"]
        + result["exposure_only_book_count"]
    )

    # ------------------------------------------------------------------------
    # Descriptive rates
    # ------------------------------------------------------------------------

    preference_evidence = result[
        "preference_evidence_count"
    ]

    positive = result[
        "positive_book_count"
    ]

    negative = result[
        "negative_book_count"
    ]

    exposure_only = result[
        "exposure_only_book_count"
    ]

    actionable = result[
        "actionable_evidence_count"
    ]

    result["positive_rate"] = np.where(
        preference_evidence > 0,
        positive / preference_evidence,
        0.0,
    )

    result["negative_rate"] = np.where(
        preference_evidence > 0,
        negative / preference_evidence,
        0.0,
    )

    # ------------------------------------------------------------------------
    # Generic evidence strength
    #
    # Answers:
    # "How much actionable evidence do we have?"
    #
    # It does NOT describe whether the evidence is positive or negative.
    # ------------------------------------------------------------------------

    result["evidence_strength"] = (
        1.0
        - np.exp(
            -actionable / prior_strength
        )
    )

    # =========================================================================
    # PREFERENCE
    # =========================================================================

    result["preference_evidence_strength"] = (
        1.0
        - np.exp(
            -preference_evidence / prior_strength
        )
    )

    # Bayesian-style smoothing prevents tiny communities from receiving
    # extreme scores simply because their first observed book was positive.

    result["smoothed_positive_rate"] = (
        positive
        + (
            prior_strength
            * prior_preference
        )
    ) / (
        preference_evidence
        + prior_strength
    )

    result["preference_strength"] = (
        result["smoothed_positive_rate"]
        * result["preference_evidence_strength"]
    )

    # =========================================================================
    # AVOIDANCE
    # =========================================================================
    #
    # Avoidance is deliberately NOT just negative_rate.
    #
    # Example:
    #
    #   37 positive / 4 negative
    #
    # contains negative evidence, but does NOT mean the reader avoids
    # that community.
    #
    # We therefore require:
    #
    #   negative_rate > positive_rate
    #
    # and at least two negative books.
    # =========================================================================

    result["negative_evidence_strength"] = (
        1.0
        - np.exp(
            -negative / prior_strength
        )
    )

    result["avoidance_balance"] = (
        result["negative_rate"]
        - result["positive_rate"]
    ).clip(lower=0.0)

    result["avoidance_strength"] = (
        result["avoidance_balance"]
        * result["negative_evidence_strength"]
    )

    # Do not call one negative book "avoidance."
    result.loc[
        negative < min_avoidance_negative_books,
        "avoidance_strength",
    ] = 0.0

    # =========================================================================
    # EXPLORATION
    # =========================================================================
    #
    # Exploration is the proportion of actionable evidence represented by
    # exposure-only books.
    #
    # This distinguishes:
    #
    #   "I have lots of books here because I love this community"
    #
    # from:
    #
    #   "I have lots of books here that I haven't established a preference
    #    for yet."
    # =========================================================================

    result["exploration_rate"] = np.where(
        actionable > 0,
        exposure_only / actionable,
        0.0,
    )

    result["exploration_evidence_strength"] = (
        1.0
        - np.exp(
            -exposure_only / prior_strength
        )
    )

    result["exploration_strength"] = (
        result["exploration_rate"]
        * result["exploration_evidence_strength"]
    )

    # =========================================================================
    # Signal flags
    # =========================================================================

    result["has_preference_evidence"] = (
        positive > 0
    )

    result["has_avoidance_evidence"] = (
        (
            negative
            >= min_avoidance_negative_books
        )
        & (
            result["avoidance_balance"]
            > 0
        )
    )

    result["has_exploration_evidence"] = (
        exposure_only > 0
    )

    # =========================================================================
    # Clean numeric output
    # =========================================================================

    score_columns = [
        "positive_rate",
        "negative_rate",
        "evidence_strength",
        "preference_evidence_strength",
        "smoothed_positive_rate",
        "preference_strength",
        "negative_evidence_strength",
        "avoidance_balance",
        "avoidance_strength",
        "exploration_rate",
        "exploration_evidence_strength",
        "exploration_strength",
    ]

    for column in score_columns:
        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

    return result


# ============================================================================
# New descriptive alias
# ============================================================================


def score_reader_community_evidence(
    dataframe: pd.DataFrame,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    prior_preference: float = DEFAULT_PRIOR_PREFERENCE,
    min_avoidance_negative_books: int = (
        DEFAULT_MIN_AVOIDANCE_NEGATIVE_BOOKS
    ),
) -> pd.DataFrame:
    """
    Descriptive alias for score_reader_communities().

    Kept separate semantically so newer code can use the more explicit
    function name while the existing experiment runner remains compatible.
    """

    return score_reader_communities(
        dataframe=dataframe,
        prior_strength=prior_strength,
        prior_preference=prior_preference,
        min_avoidance_negative_books=(
            min_avoidance_negative_books
        ),
    )


# ============================================================================
# Rankings
# ============================================================================


def _build_signal_ranking(
    dataframe: pd.DataFrame,
    signal: str,
    score_column: str,
    evidence_column: str,
    eligibility_mask: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    """Build a ranking for one Reading DNA signal."""

    candidates = dataframe.loc[
        eligibility_mask
    ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "reader_id",
                "signal",
                "community_id",
                "score",
                "signal_evidence_strength",
                "evidence_strength",
                "preference_evidence_count",
                "exposure_only_book_count",
                "positive_book_count",
                "negative_book_count",
            ]
        )

    candidates = candidates.sort_values(
        [
            score_column,
            evidence_column,
            "preference_evidence_count",
            "exposure_only_book_count",
            "community_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).head(top_n)

    candidates = candidates.copy()

    candidates.insert(
        0,
        "rank",
        range(1, len(candidates) + 1),
    )

    candidates.insert(
        2,
        "signal",
        signal,
    )

    candidates["score"] = candidates[
        score_column
    ]

    candidates["signal_evidence_strength"] = candidates[
        evidence_column
    ]

    return candidates[
        [
            "rank",
            "reader_id",
            "signal",
            "community_id",
            "score",
            "signal_evidence_strength",
            "evidence_strength",
            "preference_evidence_count",
            "exposure_only_book_count",
            "positive_book_count",
            "negative_book_count",
        ]
    ].reset_index(drop=True)


def build_reading_dna_rankings(
    scored_data: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    min_avoidance_negative_books: int = (
        DEFAULT_MIN_AVOIDANCE_NEGATIVE_BOOKS
    ),
) -> pd.DataFrame:
    """
    Build separate Preference, Avoidance, and Exploration rankings.
    """

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    require_columns(
        scored_data,
        {
            "reader_id",
            "community_id",
            "positive_book_count",
            "negative_book_count",
            "exposure_only_book_count",
            "preference_strength",
            "avoidance_strength",
            "exploration_strength",
            "preference_evidence_strength",
            "negative_evidence_strength",
            "exploration_evidence_strength",
            "evidence_strength",
            "preference_evidence_count",
        },
        "scored Reading DNA data",
    )

    ranking_frames = []

    # ------------------------------------------------------------------------
    # Preference
    # ------------------------------------------------------------------------

    preference_mask = (
        scored_data["positive_book_count"]
        > 0
    )

    for reader_id in scored_data[
        "reader_id"
    ].drop_duplicates().sort_values():

        reader_mask = (
            preference_mask
            & scored_data["reader_id"].eq(reader_id)
        )

        ranking_frames.append(
            _build_signal_ranking(
                dataframe=scored_data,
                signal="preference",
                score_column="preference_strength",
                evidence_column=(
                    "preference_evidence_strength"
                ),
                eligibility_mask=reader_mask,
                top_n=top_n,
            )
        )

    # ------------------------------------------------------------------------
    # Avoidance
    # ------------------------------------------------------------------------

    avoidance_mask = (
        scored_data["negative_book_count"]
        >= min_avoidance_negative_books
    ) & (
        scored_data["avoidance_strength"]
        > 0
    )

    for reader_id in scored_data[
        "reader_id"
    ].drop_duplicates().sort_values():

        reader_mask = (
            avoidance_mask
            & scored_data["reader_id"].eq(reader_id)
        )

        ranking_frames.append(
            _build_signal_ranking(
                dataframe=scored_data,
                signal="avoidance",
                score_column="avoidance_strength",
                evidence_column=(
                    "negative_evidence_strength"
                ),
                eligibility_mask=reader_mask,
                top_n=top_n,
            )
        )

    # ------------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------------

    exploration_mask = (
        scored_data["exposure_only_book_count"]
        > 0
    )

    for reader_id in scored_data[
        "reader_id"
    ].drop_duplicates().sort_values():

        reader_mask = (
            exploration_mask
            & scored_data["reader_id"].eq(reader_id)
        )

        ranking_frames.append(
            _build_signal_ranking(
                dataframe=scored_data,
                signal="exploration",
                score_column="exploration_strength",
                evidence_column=(
                    "exploration_evidence_strength"
                ),
                eligibility_mask=reader_mask,
                top_n=top_n,
            )
        )

    non_empty = [
        frame
        for frame in ranking_frames
        if not frame.empty
    ]

    if not non_empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "reader_id",
                "signal",
                "community_id",
                "score",
                "signal_evidence_strength",
                "evidence_strength",
                "preference_evidence_count",
                "exposure_only_book_count",
                "positive_book_count",
                "negative_book_count",
            ]
        )

    return pd.concat(
        non_empty,
        ignore_index=True,
    ).sort_values(
        [
            "reader_id",
            "signal",
            "rank",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)


# ============================================================================
# Validation
# ============================================================================


def validate_avoidance_rankings(
    scored_data: pd.DataFrame,
    rankings: pd.DataFrame,
    min_avoidance_negative_books: int = (
        DEFAULT_MIN_AVOIDANCE_NEGATIVE_BOOKS
    ),
) -> dict[str, object]:
    """
    Validate avoidance rules.

    Rules:
        1. Zero-negative communities cannot have positive avoidance.
        2. Communities below the negative threshold cannot rank.
        3. Avoidance requires negative rate > positive rate.
    """

    zero_negative_positive_score = int(
        (
            (
                scored_data["negative_book_count"]
                == 0
            )
            & (
                scored_data["avoidance_strength"]
                > 0
            )
        ).sum()
    )

    if rankings.empty:
        avoidance_rankings = rankings
    else:
        avoidance_rankings = rankings[
            rankings["signal"] == "avoidance"
        ]

    under_threshold_ranked = int(
        (
            avoidance_rankings[
                "negative_book_count"
            ]
            < min_avoidance_negative_books
        ).sum()
    )

    balance_violations = int(
        (
            (
                scored_data["avoidance_strength"]
                > 0
            )
            & (
                scored_data["negative_rate"]
                <= scored_data["positive_rate"]
            )
        ).sum()
    )

    passed = (
        zero_negative_positive_score == 0
        and under_threshold_ranked == 0
        and balance_violations == 0
    )

    return {
        "passed": passed,
        "zero_negative_positive_score": (
            zero_negative_positive_score
        ),
        "under_threshold_ranked": (
            under_threshold_ranked
        ),
        "avoidance_balance_violations": (
            balance_violations
        ),
    }


# ============================================================================
# Convenience API
# ============================================================================


def run_reading_dna_scoring(
    evidence: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score Reader × Community evidence and build rankings.
    """

    scored = score_reader_communities(
        evidence
    )

    rankings = build_reading_dna_rankings(
        scored,
        top_n=top_n,
    )

    return scored, rankings