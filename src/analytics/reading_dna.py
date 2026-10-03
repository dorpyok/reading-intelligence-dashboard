from __future__ import annotations

import ast
import re
from itertools import combinations

import numpy as np
import pandas as pd


POSITIVE_RATING_MIN = 3
NEGATIVE_RATING_MAX = 2


GENERIC_ATTRIBUTE_TERMS = {
    "book",
    "books",
    "fiction",
    "literature",
    "literary",
    "novel",
    "novels",
    "story",
    "stories",
    "reading",
    "readers",
    "reader",
    "author",
    "authors",
    "writing",
    "written",
    "publication",
    "publications",
    "general",
    "subjects",
    "subject",
    "new",
}


METADATA_ATTRIBUTE_TERMS = {
    "goodreads",
    "isbn",
    "rating",
    "ratings",
    "review",
    "reviews",
    "http",
    "https",
    "www",
    "amp",
}


ENTITY_ATTRIBUTE_TERMS = {
    "new york",
    "new york city",
    "united states",
    "usa",
    "england",
    "america",
}


def parse_list_value(
    value: object,
) -> list[str]:
    """Safely parse list-like values."""

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(parsed, list):
            return [
                str(item)
                for item in parsed
                if str(item).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [text]


def normalize_attribute(
    value: object,
) -> str:
    """
    Normalize a multi-label attribute.

    This does not invent a genre taxonomy.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = (
        str(value)
        .lower()
        .strip()
    )

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"[/_]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9'\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return ""

    if text in GENERIC_ATTRIBUTE_TERMS:
        return ""

    if text in METADATA_ATTRIBUTE_TERMS:
        return ""

    if text in ENTITY_ATTRIBUTE_TERMS:
        return ""

    return text


def extract_book_attributes(
    row: pd.Series,
) -> list[str]:
    """
    Extract overlapping multi-label attributes
    from curated Open Library subjects.
    """

    values = parse_list_value(
        row.get("subjects")
    )

    attributes = {
        normalized
        for value in values
        if (
            normalized :=
            normalize_attribute(value)
        )
    }

    return sorted(
        attributes
    )


def add_book_attributes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["attributes"] = (
        result.apply(
            extract_book_attributes,
            axis=1,
        )
    )

    result["attribute_count"] = (
        result["attributes"]
        .str.len()
    )

    return result


def derive_reader_evidence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert aggregated reader/book evidence
    into the three Reading DNA evidence streams.

    Preference:
        positive
        negative
        neutral

    Exposure:
        observed
        not_observed

    Intent:
        intent
        no_intent

    The aggregated reader/book evidence layer is
    authoritative when present.
    """

    result = df.copy()

    canonical_columns = {
        "is_positive",
        "is_negative",
        "is_exposure",
        "is_tbr",
        "is_currently_reading",
        "is_dnf",
    }

    if canonical_columns.issubset(
        result.columns
    ):

        positive = (
            result["is_positive"]
            .fillna(False)
            .astype(bool)
        )

        negative = (
            result["is_negative"]
            .fillna(False)
            .astype(bool)
        )

        exposure = (
            result["is_exposure"]
            .fillna(False)
            .astype(bool)
        )

        tbr = (
            result["is_tbr"]
            .fillna(False)
            .astype(bool)
        )

        # If both positive and negative evidence
        # exist for a relationship, preserve that
        # conflict rather than silently choosing one.
        conflicted = (
            positive & negative
        )

        result["preference_signal"] = np.select(
            [
                conflicted,
                negative,
                positive,
            ],
            [
                "conflicted",
                "negative",
                "positive",
            ],
            default="neutral",
        )

        result["exposure_signal"] = np.where(
            exposure,
            "observed",
            "not_observed",
        )

        result["intent_signal"] = np.where(
            tbr,
            "intent",
            "no_intent",
        )

        return result

    # ---------------------------------------------------------------
    # Small-test fallback
    # ---------------------------------------------------------------

    ratings = pd.to_numeric(
        result.get(
            "user_rating",
            pd.Series(
                index=result.index
            ),
        ),
        errors="coerce",
    )

    ratings = ratings.replace(
        0,
        np.nan,
    )

    status = (
        result.get(
            "reading_status",
            pd.Series(
                index=result.index
            ),
        )
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    result["preference_signal"] = np.select(
        [
            status.eq(
                "did_not_finish"
            ),
            ratings >= POSITIVE_RATING_MIN,
            ratings <= NEGATIVE_RATING_MAX,
        ],
        [
            "negative",
            "positive",
            "negative",
        ],
        default="neutral",
    )

    result["exposure_signal"] = np.where(
        status.isin(
            {
                "read",
                "did_not_finish",
                "currently_reading",
            }
        ),
        "observed",
        "not_observed",
    )

    result["intent_signal"] = np.where(
        status.eq("to_read"),
        "intent",
        "no_intent",
    )

    return result


def aggregate_reader_attributes(
    df: pd.DataFrame,
    reader: str,
) -> pd.DataFrame:
    """
    Aggregate overlapping attributes into
    reader-level Reading DNA evidence.
    """

    evidence = derive_reader_evidence(
        df
    )

    rows: list[dict] = []

    for _, row in evidence.iterrows():

        attributes = row.get(
            "attributes",
            [],
        )

        if not isinstance(
            attributes,
            list,
        ):
            attributes = parse_list_value(
                attributes
            )

        for attribute in attributes:

            rows.append(
                {
                    "reader": reader,
                    "attribute": attribute,
                    "preference_signal": (
                        row[
                            "preference_signal"
                        ]
                    ),
                    "exposure_signal": (
                        row[
                            "exposure_signal"
                        ]
                    ),
                    "intent_signal": (
                        row[
                            "intent_signal"
                        ]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "reader",
                "attribute",
                "book_count",
                "positive_count",
                "negative_count",
                "conflicted_count",
                "neutral_count",
                "observed_count",
                "intent_count",
                "preference_rate",
                "exposure_rate",
                "intent_rate",
            ]
        )

    long = pd.DataFrame(
        rows
    )

    grouped = (
        long.groupby(
            [
                "reader",
                "attribute",
            ]
        )
        .agg(
            book_count=(
                "attribute",
                "size",
            ),
            positive_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "positive"
                    ).sum()
                ),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "negative"
                    ).sum()
                ),
            ),
            conflicted_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "conflicted"
                    ).sum()
                ),
            ),
            neutral_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "neutral"
                    ).sum()
                ),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int(
                    (
                        x == "observed"
                    ).sum()
                ),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int(
                    (
                        x == "intent"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    explicit = (
        grouped["positive_count"]
        + grouped["negative_count"]
    )

    grouped["preference_rate"] = np.where(
        explicit > 0,
        (
            grouped["positive_count"]
            / explicit
        ),
        np.nan,
    )

    grouped["exposure_rate"] = (
        grouped["observed_count"]
        / grouped["book_count"]
    )

    grouped["intent_rate"] = (
        grouped["intent_count"]
        / grouped["book_count"]
    )

    return (
        grouped.sort_values(
            [
                "reader",
                "book_count",
                "attribute",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_attribute_combinations(
    df: pd.DataFrame,
    reader: str,
) -> pd.DataFrame:
    """
    Build reader-level attribute co-occurrence.

    Attributes remain independent and overlapping.
    """

    rows: list[dict] = []

    evidence = derive_reader_evidence(
        df
    )

    for _, row in evidence.iterrows():

        attributes = sorted(
            {
                attribute
                for attribute in row.get(
                    "attributes",
                    [],
                )
                if attribute
            }
        )

        for first, second in combinations(
            attributes,
            2,
        ):

            rows.append(
                {
                    "reader": reader,
                    "attribute_1": first,
                    "attribute_2": second,
                    "preference_signal": (
                        row[
                            "preference_signal"
                        ]
                    ),
                    "exposure_signal": (
                        row[
                            "exposure_signal"
                        ]
                    ),
                    "intent_signal": (
                        row[
                            "intent_signal"
                        ]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "reader",
                "attribute_1",
                "attribute_2",
                "book_count",
                "positive_count",
                "negative_count",
                "conflicted_count",
                "observed_count",
                "intent_count",
            ]
        )

    long = pd.DataFrame(
        rows
    )

    grouped = (
        long.groupby(
            [
                "reader",
                "attribute_1",
                "attribute_2",
            ]
        )
        .agg(
            book_count=(
                "reader",
                "size",
            ),
            positive_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "positive"
                    ).sum()
                ),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "negative"
                    ).sum()
                ),
            ),
            conflicted_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "conflicted"
                    ).sum()
                ),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int(
                    (
                        x == "observed"
                    ).sum()
                ),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int(
                    (
                        x == "intent"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    return (
        grouped.sort_values(
            [
                "reader",
                "book_count",
                "attribute_1",
                "attribute_2",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_neighborhood_descriptions(
    df: pd.DataFrame,
    top_terms: int = 10,
) -> pd.DataFrame:
    """
    Describe semantic neighborhoods using
    their observed multi-label attributes.

    Noise (-1) is retained as a special neighborhood
    rather than being silently discarded.
    """

    work = df.copy()

    if "neighborhood_id" not in work.columns:
        raise ValueError(
            "Dataframe must contain "
            "neighborhood_id."
        )

    rows: list[dict] = []

    for neighborhood_id, group in (
        work.groupby(
            "neighborhood_id"
        )
    ):

        attribute_counts: dict[
            str,
            int,
        ] = {}

        for attributes in group[
            "attributes"
        ]:

            if not isinstance(
                attributes,
                list,
            ):
                attributes = parse_list_value(
                    attributes
                )

            for attribute in attributes:
                attribute_counts[
                    attribute
                ] = (
                    attribute_counts.get(
                        attribute,
                        0,
                    )
                    + 1
                )

        top_attributes = [
            attribute
            for attribute, _ in sorted(
                attribute_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )[:top_terms]
        ]

        rows.append(
            {
                "neighborhood_id": int(
                    neighborhood_id
                ),
                "book_count": int(
                    len(group)
                ),
                "top_attributes": (
                    top_attributes
                ),
                "is_noise": (
                    int(neighborhood_id)
                    == -1
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "neighborhood_id"
        )
        .reset_index(drop=True)
    )


def build_reader_neighborhood_profile(
    df: pd.DataFrame,
    reader: str,
) -> pd.DataFrame:
    """
    Summarize a reader's evidence across
    semantic neighborhoods.
    """

    if "neighborhood_id" not in df.columns:
        raise ValueError(
            "Dataframe must contain "
            "neighborhood_id."
        )

    evidence = derive_reader_evidence(
        df
    )

    evidence["neighborhood_id"] = (
        df["neighborhood_id"]
        .to_numpy()
    )

    grouped = (
        evidence.groupby(
            "neighborhood_id"
        )
        .agg(
            book_count=(
                "neighborhood_id",
                "size",
            ),
            positive_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "positive"
                    ).sum()
                ),
            ),
            negative_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "negative"
                    ).sum()
                ),
            ),
            conflicted_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "conflicted"
                    ).sum()
                ),
            ),
            neutral_count=(
                "preference_signal",
                lambda x: int(
                    (
                        x == "neutral"
                    ).sum()
                ),
            ),
            observed_count=(
                "exposure_signal",
                lambda x: int(
                    (
                        x == "observed"
                    ).sum()
                ),
            ),
            intent_count=(
                "intent_signal",
                lambda x: int(
                    (
                        x == "intent"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    grouped.insert(
        0,
        "reader",
        reader,
    )

    explicit = (
        grouped["positive_count"]
        + grouped["negative_count"]
    )

    grouped["preference_rate"] = np.where(
        explicit > 0,
        (
            grouped["positive_count"]
            / explicit
        ),
        np.nan,
    )

    grouped["exposure_rate"] = (
        grouped["observed_count"]
        / grouped["book_count"]
    )

    grouped["intent_rate"] = (
        grouped["intent_count"]
        / grouped["book_count"]
    )

    grouped["is_noise"] = (
        grouped["neighborhood_id"]
        == -1
    )

    return (
        grouped.sort_values(
            [
                "reader",
                "book_count",
                "neighborhood_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def score_reader_attribute_strength(
    reader_attributes: pd.DataFrame,
    min_book_count: int = 3,
) -> pd.DataFrame:
    """
    Add descriptive evidence-strength categories.

    These are not an opaque preference score.
    """

    result = (
        reader_attributes.copy()
    )

    result["evidence_level"] = np.select(
        [
            (
                result["book_count"]
                < min_book_count
            ),
            (
                result[
                    "conflicted_count"
                ]
                > 0
            ),
            (
                result[
                    "positive_count"
                ]
                > 0
            ),
            (
                result[
                    "observed_count"
                ]
                > 0
            ),
        ],
        [
            "sparse",
            "conflicted",
            "preference_evidence",
            "exposure_only",
        ],
        default="intent_or_neutral",
    )

    return result