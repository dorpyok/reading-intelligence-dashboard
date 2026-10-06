from __future__ import annotations

from pathlib import Path

import pandas as pd


READER_BOOK_REQUIRED_COLUMNS = {
    "reader_id",
    "canonical_book_id",
    "reading_status",
    "preference_signal",
    "is_read",
    "is_positive",
    "is_negative",
    "is_exposure",
    "is_tbr",
    "is_currently_reading",
    "is_dnf",
}

LEIDEN_REQUIRED_COLUMNS = {
    "canonical_book_id",
    "community_id",
}


EVIDENCE_COLUMNS = [
    "is_read",
    "is_positive",
    "is_negative",
    "is_exposure",
    "is_tbr",
    "is_currently_reading",
    "is_dnf",
]


PREFERENCE_SIGNAL_VALUES = {
    "positive",
    "negative",
    "exposure_only",
    "read_without_preference",
    "unknown",
    "conflicted",
}


def validate_reader_book_evidence(df: pd.DataFrame) -> None:
    """Validate the reader/book evidence input."""
    missing = READER_BOOK_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "reader_book_evidence is missing required columns: "
            + ", ".join(sorted(missing))
        )

    duplicate_count = df.duplicated(
        subset=["reader_id", "canonical_book_id"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "reader_book_evidence must contain one row per "
            f"(reader_id, canonical_book_id). Found {duplicate_count} duplicates."
        )

    unexpected_signals = (
        set(df["preference_signal"].dropna().unique())
        - PREFERENCE_SIGNAL_VALUES
    )

    if unexpected_signals:
        raise ValueError(
            "Unexpected preference_signal values: "
            + ", ".join(sorted(unexpected_signals))
        )


def validate_leiden_assignments(df: pd.DataFrame) -> None:
    """Validate the Leiden community assignment input."""
    missing = LEIDEN_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "leiden_assignments is missing required columns: "
            + ", ".join(sorted(missing))
        )

    duplicate_count = df.duplicated(
        subset=["canonical_book_id"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "leiden_assignments must contain one row per canonical_book_id. "
            f"Found {duplicate_count} duplicates."
        )


def _ensure_bool_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Normalize boolean evidence columns."""
    result = df.copy()

    for column in columns:
        result[column] = result[column].fillna(False).astype(bool)

    return result


def build_reader_community_evidence(
    reader_book_evidence: pd.DataFrame,
    leiden_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one descriptive evidence row per reader/community pair.

    The evidence categories come from the already-aggregated
    reader/book evidence layer. Unknown books are retained as
    evidence records but do not count as preference or exploration.
    """
    validate_reader_book_evidence(reader_book_evidence)
    validate_leiden_assignments(leiden_assignments)

    reader_books = _ensure_bool_columns(
        reader_book_evidence,
        EVIDENCE_COLUMNS,
    )

    assignments = leiden_assignments[
        ["canonical_book_id", "community_id"]
    ].copy()

    merged = reader_books.merge(
        assignments,
        on="canonical_book_id",
        how="left",
        validate="many_to_one",
    )

    merged["is_unassigned"] = merged["community_id"].isna()

    assigned = merged.loc[~merged["is_unassigned"]].copy()

    if assigned.empty:
        return pd.DataFrame(
            columns=[
                "reader_id",
                "community_id",
                "book_count",
                "read_book_count",
                "positive_book_count",
                "negative_book_count",
                "exposure_book_count",
                "exposure_only_book_count",
                "read_without_preference_book_count",
                "unknown_book_count",
                "conflicted_book_count",
                "tbr_book_count",
                "currently_reading_book_count",
                "dnf_book_count",
                "preference_evidence_count",
                "actionable_evidence_count",
                "positive_rate",
                "negative_rate",
                "exposure_only_rate",
            ]
        )

    # Explicit evidence categories.
    assigned["is_positive_signal"] = (
        assigned["preference_signal"] == "positive"
    )

    assigned["is_negative_signal"] = (
        assigned["preference_signal"] == "negative"
    )

    assigned["is_exposure_only_signal"] = (
        assigned["preference_signal"] == "exposure_only"
    )

    assigned["is_read_without_preference_signal"] = (
        assigned["preference_signal"] == "read_without_preference"
    )

    assigned["is_unknown_signal"] = (
        assigned["preference_signal"] == "unknown"
    )

    assigned["is_conflicted_signal"] = (
        assigned["preference_signal"] == "conflicted"
    )

    grouped = (
        assigned.groupby(
            ["reader_id", "community_id"],
            as_index=False,
        )
        .agg(
            book_count=("canonical_book_id", "nunique"),
            read_book_count=("is_read", "sum"),
            positive_book_count=("is_positive_signal", "sum"),
            negative_book_count=("is_negative_signal", "sum"),
            exposure_book_count=("is_exposure", "sum"),
            exposure_only_book_count=(
                "is_exposure_only_signal",
                "sum",
            ),
            read_without_preference_book_count=(
                "is_read_without_preference_signal",
                "sum",
            ),
            unknown_book_count=(
                "is_unknown_signal",
                "sum",
            ),
            conflicted_book_count=(
                "is_conflicted_signal",
                "sum",
            ),
            tbr_book_count=("is_tbr", "sum"),
            currently_reading_book_count=(
                "is_currently_reading",
                "sum",
            ),
            dnf_book_count=("is_dnf", "sum"),
        )
    )

    grouped["preference_evidence_count"] = (
        grouped["positive_book_count"]
        + grouped["negative_book_count"]
    )

    grouped["actionable_evidence_count"] = (
        grouped["preference_evidence_count"]
        + grouped["exposure_only_book_count"]
    )

    grouped["positive_rate"] = (
        grouped["positive_book_count"]
        / grouped["preference_evidence_count"]
    ).fillna(0.0)

    grouped["negative_rate"] = (
        grouped["negative_book_count"]
        / grouped["preference_evidence_count"]
    ).fillna(0.0)

    grouped["exposure_only_rate"] = (
        grouped["exposure_only_book_count"]
        / grouped["actionable_evidence_count"]
    ).fillna(0.0)

    grouped["community_id"] = grouped["community_id"].astype(int)

    return grouped.sort_values(
        ["reader_id", "community_id"]
    ).reset_index(drop=True)


def build_reader_unassigned_evidence(
    reader_book_evidence: pd.DataFrame,
    leiden_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize books without a Leiden community assignment.
    """
    validate_reader_book_evidence(reader_book_evidence)
    validate_leiden_assignments(leiden_assignments)

    reader_books = _ensure_bool_columns(
        reader_book_evidence,
        EVIDENCE_COLUMNS,
    )

    assigned_ids = set(
        leiden_assignments["canonical_book_id"]
    )

    unassigned = reader_books.loc[
        ~reader_books["canonical_book_id"].isin(assigned_ids)
    ].copy()

    if unassigned.empty:
        return pd.DataFrame(
            columns=[
                "reader_id",
                "unassigned_book_count",
                "read_book_count",
                "positive_book_count",
                "negative_book_count",
                "exposure_book_count",
                "tbr_book_count",
                "currently_reading_book_count",
                "dnf_book_count",
            ]
        )

    result = (
        unassigned.groupby(
            "reader_id",
            as_index=False,
        )
        .agg(
            unassigned_book_count=(
                "canonical_book_id",
                "nunique",
            ),
            read_book_count=("is_read", "sum"),
            positive_book_count=("is_positive", "sum"),
            negative_book_count=("is_negative", "sum"),
            exposure_book_count=("is_exposure", "sum"),
            tbr_book_count=("is_tbr", "sum"),
            currently_reading_book_count=(
                "is_currently_reading",
                "sum",
            ),
            dnf_book_count=("is_dnf", "sum"),
        )
    )

    return result.sort_values(
        "reader_id"
    ).reset_index(drop=True)


def build_reader_community_evidence_from_files(
    reader_book_path: str | Path,
    leiden_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inputs and build community evidence tables."""
    reader_book_evidence = pd.read_csv(
        reader_book_path
    )

    leiden_assignments = pd.read_csv(
        leiden_path
    )

    community_evidence = build_reader_community_evidence(
        reader_book_evidence,
        leiden_assignments,
    )

    unassigned_evidence = build_reader_unassigned_evidence(
        reader_book_evidence,
        leiden_assignments,
    )

    return (
        community_evidence,
        unassigned_evidence,
    )