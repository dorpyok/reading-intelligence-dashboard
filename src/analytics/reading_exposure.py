from __future__ import annotations

import pandas as pd


OBSERVED_STATUSES = {
    "read",
    "did_not_finish",
    "currently_reading",
}


def derive_reading_exposure(row: pd.Series) -> str:
    """
    Determine whether a book represents observed reading exposure.

    Observed:
        - read
        - did_not_finish
        - currently_reading

    Not observed:
        - to_read
        - unknown

    Reading exposure is intentionally separate from preference.
    A reader can be exposed to a book without explicitly liking it.
    """

    status = str(
        row.get("reading_status", "")
    ).strip().lower()

    if status in OBSERVED_STATUSES:
        return "observed"

    return "not_observed"


def add_reading_exposure(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a reading_exposure column to a dataframe.

    Returns a copy of the input dataframe.
    """

    result = df.copy()

    result["reading_exposure"] = result.apply(
        derive_reading_exposure,
        axis=1,
    )

    return result