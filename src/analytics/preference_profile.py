from __future__ import annotations

import pandas as pd


def build_preference_profile(df: pd.DataFrame) -> dict:
    """
    Summarize the preference evidence available for a reader.

    This function measures the available evidence but does not
    determine whether the reader has "enough" data for modeling.

    Preference signals:
        - Positive: rating >= 4
        - Negative: rating <= 2
        - DNF: negative preference signal
        - Neutral: rating == 3
        - Unrated: no explicit rating

    Args:
        df: Reader reading-history dataframe.

    Returns:
        Dictionary containing counts and basic preference statistics.
    """

    result = df.copy()

    # Normalize rating values.
    if "rating" in result.columns:
        ratings = pd.to_numeric(
            result["rating"],
            errors="coerce",
        )
    elif "user_rating" in result.columns:
        ratings = pd.to_numeric(
            result["user_rating"],
            errors="coerce",
        )
        ratings = ratings.replace(0, pd.NA)
    else:
        ratings = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Float64",
        )

    result["_rating"] = ratings

    # Identify DNF books.
    if "reading_status" in result.columns:
        dnf = (
            result["reading_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("did_not_finish")
        )
    elif "shelves" in result.columns:
        dnf = (
            result["shelves"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "did-not-finish",
                regex=False,
            )
        )
    else:
        dnf = pd.Series(
            False,
            index=result.index,
        )

    positive = result["_rating"] >= 4
    negative_rating = result["_rating"] <= 2
    neutral = result["_rating"] == 3

    negative = negative_rating | dnf

    rated = result["_rating"].notna()
    unrated = ~rated

    total_books = len(result)
    positive_count = int(positive.sum())
    negative_rating_count = int(negative_rating.sum())
    dnf_count = int(dnf.sum())
    negative_count = int(negative.sum())
    neutral_count = int(neutral.sum())
    rated_count = int(rated.sum())
    unrated_count = int(unrated.sum())

    profile = {
        "total_books": total_books,
        "rated_books": rated_count,
        "unrated_books": unrated_count,
        "positive_books": positive_count,
        "negative_books": negative_count,
        "negative_rating_books": negative_rating_count,
        "dnf_books": dnf_count,
        "neutral_books": neutral_count,
    }

    if total_books > 0:
        profile["rated_pct"] = rated_count / total_books
        profile["positive_pct"] = positive_count / total_books
        profile["negative_pct"] = negative_count / total_books
        profile["unrated_pct"] = unrated_count / total_books
    else:
        profile["rated_pct"] = 0.0
        profile["positive_pct"] = 0.0
        profile["negative_pct"] = 0.0
        profile["unrated_pct"] = 0.0

    return profile