from __future__ import annotations

import pandas as pd


POSITIVE_RATING_MIN = 4
NEGATIVE_RATING_MAX = 2


def derive_preference_signal(row: pd.Series) -> str:
    """
    Derive a preference signal from canonical reading behavior.

    Positive:
        - Rating >= 4

    Negative:
        - Did not finish
        - Rating <= 2

    Neutral:
        - Rating == 3
        - No rating
        - Currently reading
        - To-read
        - Unknown

    DNF is treated as negative preference even when no rating exists.
    """

    status = str(row.get("reading_status", "")).strip().lower()

    if status == "did_not_finish":
        return "negative"

    rating = row.get("rating")

    if pd.notna(rating):
        try:
            rating = float(rating)

            if rating >= POSITIVE_RATING_MIN:
                return "positive"

            if rating <= NEGATIVE_RATING_MAX:
                return "negative"

        except (TypeError, ValueError):
            pass

    return "neutral"


def add_preference_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add preference_signal to a dataframe containing canonical reading data.

    The original reading data is preserved.
    """

    result = df.copy()

    result["preference_signal"] = result.apply(
        derive_preference_signal,
        axis=1,
    )

    return result