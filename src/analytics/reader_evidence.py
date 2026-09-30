from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "reader_id",
    "canonical_work_id",
    "canonical_book_id",
    "reading_status",
    "user_rating",
    "date_read",
}


VALID_STATUSES = {
    "to_read",
    "currently_reading",
    "read",
    "did_not_finish",
    "unknown",
}


def validate_reader_books(
    reader_books: pd.DataFrame,
) -> None:
    """
    Validate the reader-level canonical book records.
    """
    missing = REQUIRED_COLUMNS - set(reader_books.columns)

    if missing:
        raise ValueError(
            "Reader books are missing required columns: "
            + ", ".join(sorted(missing))
        )

    if reader_books.empty:
        return

    if reader_books["reader_id"].isna().any():
        raise ValueError(
            "Reader books contain missing reader_id values."
        )

    if reader_books["canonical_work_id"].isna().any():
        raise ValueError(
            "Reader books contain missing canonical_work_id values."
        )

    if reader_books["canonical_book_id"].isna().any():
        raise ValueError(
            "Reader books contain missing canonical_book_id values."
        )

    invalid_statuses = (
        set(
            reader_books["reading_status"]
            .dropna()
            .astype(str)
        )
        - VALID_STATUSES
    )

    if invalid_statuses:
        raise ValueError(
            "Reader books contain invalid reading_status values: "
            + ", ".join(sorted(invalid_statuses))
        )


def _is_positive_rating(
    rating: Any,
) -> bool:
    """Return True for a 3-5 star rating."""
    if pd.isna(rating):
        return False

    try:
        value = float(rating)
    except (TypeError, ValueError):
        return False

    return value >= 3.0


def _is_negative_rating(
    rating: Any,
) -> bool:
    """Return True for a 1-2 star rating."""
    if pd.isna(rating):
        return False

    try:
        value = float(rating)
    except (TypeError, ValueError):
        return False

    return 0 < value <= 2.0


def _has_read_date(
    value: Any,
) -> bool:
    """Return True when a meaningful read date is present."""
    if pd.isna(value):
        return False

    return bool(str(value).strip())


def build_reader_evidence(
    reader_books: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build deterministic reader/book evidence records.

    Important distinction:

    is_read
        Indicates reading behavior.

    is_positive
        Indicates explicit positive preference evidence.
        Currently this means a rating of 3-5.

    is_negative
        Indicates negative preference evidence.
        Currently this means a rating of 1-2 or DNF.

    is_exposure
        Indicates TBR or currently-reading exposure.

    A read date or read status by itself does NOT imply that
    the reader liked the book.
    """
    validate_reader_books(reader_books)

    output_columns = [
        "reader_id",
        "canonical_work_id",
        "canonical_book_id",
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
    ]

    if reader_books.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    evidence = reader_books[
        [
            "reader_id",
            "canonical_work_id",
            "canonical_book_id",
            "reading_status",
            "user_rating",
            "date_read",
        ]
    ].copy()

    evidence["reading_status"] = (
        evidence["reading_status"]
        .fillna("unknown")
        .astype(str)
    )

    evidence["is_tbr"] = (
        evidence["reading_status"]
        == "to_read"
    )

    evidence["is_currently_reading"] = (
        evidence["reading_status"]
        == "currently_reading"
    )

    evidence["is_dnf"] = (
        evidence["reading_status"]
        == "did_not_finish"
    )

    evidence["has_read_date"] = evidence[
        "date_read"
    ].apply(_has_read_date)

    evidence["rating_is_positive"] = evidence[
        "user_rating"
    ].apply(_is_positive_rating)

    evidence["rating_is_negative"] = evidence[
        "user_rating"
    ].apply(_is_negative_rating)

    # Reading behavior.
    #
    # A book is read if:
    #   - it has explicit read status
    #   - it has a read date
    #   - it has a positive rating
    evidence["is_read"] = (
        evidence["reading_status"].eq("read")
        | evidence["has_read_date"]
        | evidence["rating_is_positive"]
    )

    # Negative preference evidence.
    #
    # DNF is treated as negative regardless of rating.
    evidence["is_negative"] = (
        evidence["is_dnf"]
        | evidence["rating_is_negative"]
    )

    # Positive preference evidence.
    #
    # Explicit positive rating is required.
    #
    # A DNF overrides a positive rating because the behavioral
    # evidence says the reader did not finish the book.
    evidence["is_positive"] = (
        evidence["rating_is_positive"]
        & ~evidence["is_dnf"]
        & ~evidence["rating_is_negative"]
    )

    # Exposure is distinct from preference.
    evidence["is_exposure"] = (
        evidence["is_tbr"]
        | evidence["is_currently_reading"]
    )

    def assign_evidence_strength(
        row: pd.Series,
    ) -> str:
        if row["is_dnf"]:
            return "negative"

        if row["rating_is_negative"]:
            return "negative"

        if row["rating_is_positive"]:
            return "strong"

        if row["reading_status"] == "read":
            return "moderate"

        if row["has_read_date"]:
            return "moderate"

        if row["is_currently_reading"]:
            return "exposure"

        if row["is_tbr"]:
            return "exposure"

        return "unknown"

    evidence["evidence_strength"] = evidence.apply(
        assign_evidence_strength,
        axis=1,
    )

    return evidence[
        output_columns
    ].reset_index(drop=True)