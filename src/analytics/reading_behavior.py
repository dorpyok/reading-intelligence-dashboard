from __future__ import annotations

import pandas as pd


def derive_reading_behavior(row: pd.Series) -> str:
    """
    Derive a reading-behavior signal from canonical reading status.

    Signals:
        - read
        - did_not_finish
        - currently_reading
        - to_read
        - unknown

    An unrated book that is known to have been read is still
    considered reading behavior. It is not treated as a
    positive preference signal.
    """

    status = str(
        row.get("reading_status", "")
    ).strip().lower()

    if status == "read":
        return "read"

    if status == "did_not_finish":
        return "did_not_finish"

    if status == "currently_reading":
        return "currently_reading"

    if status == "to_read":
        return "to_read"

    return "unknown"


def add_reading_behavior(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a reading_behavior column to a dataframe.

    Returns a copy of the input dataframe.
    """

    result = df.copy()

    result["reading_behavior"] = result.apply(
        derive_reading_behavior,
        axis=1,
    )

    return result