import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.reading_dna import (
    add_book_attributes,
    aggregate_reader_attributes,
    build_attribute_combinations,
    build_neighborhood_descriptions,
    build_reader_neighborhood_profile,
    extract_book_attributes,
    normalize_attribute,
    score_reader_attribute_strength,
)


def test_normalize_attribute_removes_metadata_noise():
    assert normalize_attribute("Goodreads") == ""
    assert normalize_attribute("  Feminist Fiction  ") == "feminist fiction"
    assert normalize_attribute("New York") == ""


def test_extract_book_attributes_is_multilabel():
    row = pd.Series(
        {
            "subjects": [
                "Horror",
                "Feminist fiction",
                "Speculative fiction",
                "fiction",
                "Goodreads",
            ]
        }
    )

    assert extract_book_attributes(row) == [
        "feminist fiction",
        "horror",
        "speculative fiction",
    ]


def test_add_book_attributes():
    df = pd.DataFrame(
        [
            {"subjects": ["Horror", "Feminism"]},
            {"subjects": ["Romance"]},
        ]
    )

    result = add_book_attributes(df)

    assert result["attribute_count"].tolist() == [2, 1]


def test_reader_attribute_aggregation_keeps_evidence_streams_separate():
    df = pd.DataFrame(
        [
            {
                "attributes": ["horror", "feminism"],
                "user_rating": 5,
                "reading_status": "read",
            },
            {
                "attributes": ["horror"],
                "user_rating": 2,
                "reading_status": "read",
            },
            {
                "attributes": ["horror", "feminism"],
                "user_rating": 0,
                "reading_status": "to_read",
            },
        ]
    )

    result = aggregate_reader_attributes(df, "you")

    horror = result[result["attribute"] == "horror"].iloc[0]

    assert horror["book_count"] == 3
    assert horror["positive_count"] == 1
    assert horror["negative_count"] == 1
    assert horror["observed_count"] == 2
    assert horror["intent_count"] == 1
    assert horror["preference_rate"] == 0.5


def test_attribute_combinations_are_pairwise():
    df = pd.DataFrame(
        [
            {
                "attributes": ["horror", "feminism", "speculative"],
                "user_rating": 5,
                "reading_status": "read",
            }
        ]
    )

    result = build_attribute_combinations(df, "you")

    assert len(result) == 3

    pairs = {
        tuple(sorted([row.attribute_1, row.attribute_2]))
        for row in result.itertuples()
    }

    assert pairs == {
        ("feminism", "horror"),
        ("feminism", "speculative"),
        ("horror", "speculative"),
    }


def test_neighborhood_descriptions_include_noise():
    df = pd.DataFrame(
        [
            {
                "neighborhood_id": 0,
                "attributes": ["horror", "feminist fiction"],
            },
            {
                "neighborhood_id": 0,
                "attributes": ["horror", "speculative fiction"],
            },
            {
                "neighborhood_id": -1,
                "attributes": ["romance"],
            },
        ]
    )

    result = build_neighborhood_descriptions(
        df,
        top_terms=2,
    )

    assert set(result["neighborhood_id"]) == {0, -1}

    community = result[result["neighborhood_id"] == 0].iloc[0]

    assert community["book_count"] == 2
    assert "horror" in community["top_attributes"]
    assert bool(community["is_noise"]) is False

    noise = result[result["neighborhood_id"] == -1].iloc[0]

    assert noise["book_count"] == 1
    assert bool(noise["is_noise"]) is True


def test_reader_neighborhood_profile_preserves_evidence():
    df = pd.DataFrame(
        [
            {
                "neighborhood_id": 0,
                "attributes": ["horror"],
                "user_rating": 5,
                "reading_status": "read",
            },
            {
                "neighborhood_id": 0,
                "attributes": ["horror"],
                "user_rating": 2,
                "reading_status": "read",
            },
            {
                "neighborhood_id": 1,
                "attributes": ["romance"],
                "user_rating": 0,
                "reading_status": "to_read",
            },
        ]
    )

    result = build_reader_neighborhood_profile(
        df,
        "you",
    )

    neighborhood_zero = result[
        result["neighborhood_id"] == 0
    ].iloc[0]

    assert neighborhood_zero["book_count"] == 2
    assert neighborhood_zero["positive_count"] == 1
    assert neighborhood_zero["negative_count"] == 1
    assert neighborhood_zero["observed_count"] == 2
    assert neighborhood_zero["preference_rate"] == 0.5

    neighborhood_one = result[
        result["neighborhood_id"] == 1
    ].iloc[0]

    assert neighborhood_one["book_count"] == 1
    assert neighborhood_one["positive_count"] == 0
    assert neighborhood_one["negative_count"] == 0
    assert neighborhood_one["observed_count"] == 0
    assert neighborhood_one["intent_count"] == 1


def test_score_reader_attribute_strength_marks_sparse_evidence():
    df = pd.DataFrame(
        [
            {
                "reader": "you",
                "attribute": "horror",
                "book_count": 2,
                "positive_count": 1,
                "negative_count": 0,
                "conflicted_count": 0,
                "observed_count": 2,
                "intent_count": 0,
                "preference_rate": 1.0,
                "exposure_rate": 1.0,
                "intent_rate": 0.0,
            },
            {
                "reader": "you",
                "attribute": "romance",
                "book_count": 4,
                "positive_count": 3,
                "negative_count": 0,
                "conflicted_count": 0,
                "observed_count": 4,
                "intent_count": 0,
                "preference_rate": 1.0,
                "exposure_rate": 1.0,
                "intent_rate": 0.0,
            },
        ]
    )

    result = score_reader_attribute_strength(
        df,
        min_book_count=3,
    )

    horror = result[
        result["attribute"] == "horror"
    ].iloc[0]

    romance = result[
        result["attribute"] == "romance"
    ].iloc[0]

    assert horror["evidence_level"] == "sparse"
    assert romance["evidence_level"] == "preference_evidence"


def test_score_reader_attribute_strength_marks_conflicted_evidence():
    df = pd.DataFrame(
        [
            {
                "reader": "you",
                "attribute": "fantasy",
                "book_count": 5,
                "positive_count": 2,
                "negative_count": 1,
                "conflicted_count": 1,
                "observed_count": 5,
                "intent_count": 0,
                "preference_rate": 2 / 3,
                "exposure_rate": 1.0,
                "intent_rate": 0.0,
            },
        ]
    )

    result = score_reader_attribute_strength(df)

    assert result.iloc[0]["evidence_level"] == "conflicted"