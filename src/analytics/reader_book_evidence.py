"""
Reader/book-level aggregation of reading evidence.

This module converts record-level evidence into one analytical
relationship per reader + canonical book.

Important:
- reader_evidence.csv remains record-level and is not modified.
- Multiple Goodreads records for the same reader/book are legitimate.
- This layer aggregates those records rather than dropping them.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "reader_id",
    "canonical_book_id",
    "canonical_work_id",
    "reading_status",
    "user_rating",
    "date_read",
    "is_read",
    "is_positive",
    "is_negative",
    "is_exposure",
    "is_tbr",
    "is_currently_reading",
    "is_dnf",
    "evidence_strength",
}


STATUS_PRECEDENCE = {
    "did_not_finish": 5,
    "read": 4,
    "currently_reading": 3,
    "to_read": 2,
    "unknown": 1,
}


EVIDENCE_STRENGTH_PRECEDENCE = {
    "unknown": 1,
    "exposure": 2,
    "moderate": 3,
    "strong": 4,
    "negative": 5,
}


def validate_reader_evidence(df: pd.DataFrame) -> None:
    """
    Validate the record-level evidence table before aggregation.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"reader_evidence is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError("reader_evidence is empty.")

    key_columns = ["reader_id", "canonical_book_id"]

    if df[key_columns].isna().any().any():
        raise ValueError(
            "reader_evidence contains missing reader_id or "
            "canonical_book_id values."
        )


def _as_bool(series: pd.Series) -> pd.Series:
    """
    Normalize boolean-like values safely.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
    )


def _select_status(group: pd.DataFrame) -> str:
    """
    Select the strongest observed reading status.

    DNF takes precedence over completed reading, followed by read,
    currently reading, TBR, and unknown.

    The individual boolean evidence fields are retained separately,
    so this field is a summary rather than a replacement for them.
    """
    statuses = (
        group["reading_status"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ranked = statuses.map(STATUS_PRECEDENCE).fillna(
        STATUS_PRECEDENCE["unknown"]
    )

    return statuses.loc[ranked.idxmax()]


def _select_evidence_strength(group: pd.DataFrame) -> str:
    """
    Summarize the strongest overall evidence.

    If both positive and negative evidence exist for the same
    reader/book relationship, classify the relationship as conflicted
    rather than silently choosing one side.
    """
    has_positive = bool(group["is_positive"].any())
    has_negative = bool(group["is_negative"].any())

    if has_positive and has_negative:
        return "conflicted"

    strengths = (
        group["evidence_strength"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ranked = strengths.map(EVIDENCE_STRENGTH_PRECEDENCE).fillna(1)

    return strengths.loc[ranked.idxmax()]


def _select_preference_signal(group: pd.DataFrame) -> str:
    """
    Summarize the reader's preference signal for a book.

    This deliberately distinguishes preference from exposure and
    reading behavior.
    """
    has_positive = bool(group["is_positive"].any())
    has_negative = bool(group["is_negative"].any())

    if has_positive and has_negative:
        return "conflicted"

    if has_positive:
        return "positive"

    if has_negative:
        return "negative"

    if bool(group["is_read"].any()):
        return "read_without_preference"

    if bool(group["is_exposure"].any()):
        return "exposure_only"

    return "unknown"


def _latest_date(series: pd.Series) -> str | None:
    """
    Return the latest non-null reading date as a normalized string.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)

    if parsed.notna().sum() == 0:
        return None

    latest = parsed.max()

    return latest.isoformat()


def aggregate_reader_book_evidence(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate record-level evidence to one row per reader + book.

    All original evidence records remain untouched in the source
    dataframe. This function creates a separate analytical view.

    Returns
    -------
    pd.DataFrame
        One row per reader_id + canonical_book_id.
    """
    validate_reader_evidence(evidence)

    df = evidence.copy()

    boolean_columns = [
        "is_read",
        "is_positive",
        "is_negative",
        "is_exposure",
        "is_tbr",
        "is_currently_reading",
        "is_dnf",
    ]

    for column in boolean_columns:
        df[column] = _as_bool(df[column])

    df["user_rating"] = pd.to_numeric(
        df["user_rating"],
        errors="coerce",
    )

    grouped_rows: list[dict] = []

    group_columns = [
        "reader_id",
        "canonical_book_id",
    ]

    for (reader_id, canonical_book_id), group in df.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        work_ids = (
            group["canonical_work_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        work_ids = work_ids[work_ids != ""]

        if work_ids.empty:
            raise ValueError(
                f"No canonical_work_id found for "
                f"{reader_id} / {canonical_book_id}."
            )

        if work_ids.nunique() > 1:
            raise ValueError(
                "A reader/book relationship maps to multiple "
                f"canonical works: {reader_id} / {canonical_book_id}"
            )

        row = {
            "reader_id": reader_id,
            "canonical_book_id": canonical_book_id,
            "canonical_work_id": work_ids.iloc[0],

            # How many underlying records contributed to this relationship?
            "source_record_count": len(group),

            # Overall relationship summary.
            "reading_status": _select_status(group),
            "preference_signal": _select_preference_signal(group),
            "evidence_strength": _select_evidence_strength(group),

            # Preserve the strongest/simple boolean evidence.
            "is_read": bool(group["is_read"].any()),
            "is_positive": bool(group["is_positive"].any()),
            "is_negative": bool(group["is_negative"].any()),
            "is_exposure": bool(group["is_exposure"].any()),
            "is_tbr": bool(group["is_tbr"].any()),
            "is_currently_reading": bool(
                group["is_currently_reading"].any()
            ),
            "is_dnf": bool(group["is_dnf"].any()),

            # Useful counts for explaining the relationship.
            "read_record_count": int(group["is_read"].sum()),
            "positive_record_count": int(
                group["is_positive"].sum()
            ),
            "negative_record_count": int(
                group["is_negative"].sum()
            ),
            "exposure_record_count": int(
                group["is_exposure"].sum()
            ),
            "tbr_record_count": int(group["is_tbr"].sum()),
            "currently_reading_record_count": int(
                group["is_currently_reading"].sum()
            ),
            "dnf_record_count": int(group["is_dnf"].sum()),

            # Rating is descriptive only; preference is determined by
            # the evidence flags above.
            "max_user_rating": (
                float(group["user_rating"].max())
                if group["user_rating"].notna().any()
                else None
            ),

            # Retain the latest known reading date.
            "latest_date_read": _latest_date(group["date_read"]),
        }

        grouped_rows.append(row)

    result = pd.DataFrame(grouped_rows)

    if result.empty:
        raise ValueError(
            "Reader/book evidence aggregation produced no rows."
        )

    if result.duplicated(
        ["reader_id", "canonical_book_id"]
    ).any():
        raise ValueError(
            "Aggregated reader/book evidence contains duplicate "
            "reader/book relationships."
        )

    return result.reset_index(drop=True)