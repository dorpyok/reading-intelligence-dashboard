from __future__ import annotations

import pandas as pd

from src.analytics.preference_signals import (
    add_preference_signals,
)
from src.analytics.reading_behavior import (
    add_reading_behavior,
)
from src.analytics.reading_exposure import (
    add_reading_exposure,
)


def build_preference_evidence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a unified evidence table for a reader's books.

    Adds:
        - preference_signal
        - reading_behavior
        - reading_exposure

    This function does not assign weights or calculate a
    preference score. It combines independently derived
    evidence so that later modeling experiments can determine
    how different evidence types should contribute.
    """

    result = df.copy()

    result = add_preference_signals(result)
    result = add_reading_behavior(result)
    result = add_reading_exposure(result)

    return result