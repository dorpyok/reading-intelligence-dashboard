import pandas as pd

from src.analytics.metadata_normalization import (
    build_semantic_text,
    clean_list,
    clean_subjects,
    clean_text,
    is_catalog_noise_subject,
    normalize_subject,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_removes_html():
    assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_clean_text_decodes_html_entities():
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"


def test_clean_text_repairs_common_mojibake():
    value = "cafÃ©"
    assert clean_text(value) == "café"


# ---------------------------------------------------------------------------
# clean_list
# ---------------------------------------------------------------------------

def test_clean_list_handles_lists():
    assert clean_list(["Fantasy", " Romance "]) == [
        "Fantasy",
        "Romance",
    ]


def test_clean_list_handles_delimited_strings():
    assert clean_list("Fantasy, Romance, Mystery") == [
        "Fantasy",
        "Romance",
        "Mystery",
    ]


# ---------------------------------------------------------------------------
# subject normalization
# ---------------------------------------------------------------------------

def test_normalize_subject():
    assert normalize_subject("  Fantasy  ") == "Fantasy"


def test_catalog_noise_detection():
    assert is_catalog_noise_subject(
        "collectionID:EanesChallenge"
    )

    assert is_catalog_noise_subject(
        "[series:SJM_Crescent:City]"
    )

    assert is_catalog_noise_subject(
        "nyt:trade-fiction-paperback=2012-03-25"
    )

    assert is_catalog_noise_subject(
        "Translations into Hindi"
    )

    assert is_catalog_noise_subject(
        "Vietnamese language materials"
    )

    assert is_catalog_noise_subject(
        "Large type books"
    )


# ---------------------------------------------------------------------------
# IMPORTANT: useful semantic subjects must survive
# ---------------------------------------------------------------------------

def test_useful_subjects_are_not_catalog_noise():
    useful_subjects = [
        "Fantasy",
        "Urban",
        "Romance",
        "Paranormal",
        "Epic",
        "Angels",
        "Demonology",
        "Fairies",
        "Murder",
        "Sexual attraction",
        "Adultery",
        "College students",
        "Businessmen",
        "Self-realization",
    ]

    for subject in useful_subjects:
        assert not is_catalog_noise_subject(subject), subject


def test_clean_subjects_preserves_useful_subjects():
    subjects = [
        "Fantasy",
        "Romance",
        "Paranormal",
        "Murder",
        "collectionID:EanesChallenge",
        "[series:SJM_Crescent:City]",
        "nyt:trade-fiction-paperback=2012-03-25",
        "Large type books",
        "Translations into Hindi",
    ]

    cleaned = clean_subjects(subjects)

    assert cleaned == [
        "Fantasy",
        "Romance",
        "Paranormal",
        "Murder",
    ]


def test_clean_subjects_deduplicates_case_insensitively():
    subjects = [
        "Fantasy",
        "fantasy",
        "FANTASY",
        "Romance",
        "romance",
    ]

    cleaned = clean_subjects(subjects)

    assert cleaned == [
        "Fantasy",
        "Romance",
    ]


# ---------------------------------------------------------------------------
# semantic text
# ---------------------------------------------------------------------------

def test_build_semantic_text_includes_core_fields():
    text = build_semantic_text(
        title="Test Book",
        author="Jane Doe",
        description="A story about friendship.",
    )

    assert "Title: Test Book" in text
    assert "Author: Jane Doe" in text
    assert "Description: A story about friendship." in text


def test_build_semantic_text_includes_cleaned_subjects():
    text = build_semantic_text(
        title="House of Earth and Blood",
        author="Sarah J. Maas",
        description="A fantasy story.",
        subjects=[
            "Fantasy",
            "Romance",
            "collectionID:EanesChallenge",
            "[series:SJM_Crescent:City]",
        ],
    )

    assert "Subjects: Fantasy, Romance" in text
    assert "collectionID:EanesChallenge" not in text
    assert "[series:SJM_Crescent:City]" not in text


def test_build_semantic_text_does_not_add_people_places_times_yet():
    text = build_semantic_text(
        title="Test Book",
        author="Jane Doe",
        description="A story.",
        subjects=["Fantasy"],
        people=["Character One"],
        places=["Crescent City"],
        times=["2011"],
    )

    assert "Subjects: Fantasy" in text
    assert "Character One" not in text
    assert "Crescent City" not in text
    assert "2011" not in text


# ---------------------------------------------------------------------------
# dataframe compatibility
# ---------------------------------------------------------------------------

def test_semantic_metadata_can_be_built_from_dataframe():
    df = pd.DataFrame(
        [
            {
                "canonical_book_id": "book_123",
                "title": "Test Book",
                "author": "Jane Doe",
                "description": "A fantasy story.",
                "subjects": [
                    "Fantasy",
                    "Romance",
                    "collectionID:EanesChallenge",
                ],
                "people": ["Character One"],
                "places": ["Crescent City"],
                "times": ["2011"],
            }
        ]
    )

    from src.analytics.metadata_normalization import build_semantic_metadata

    result = build_semantic_metadata(df)

    assert len(result) == 1
    assert result.loc[0, "canonical_book_id"] == "book_123"
    assert result.loc[0, "subject_count"] == 2
    assert "Fantasy" in result.loc[0, "semantic_text"]
    assert "Romance" in result.loc[0, "semantic_text"]
    assert "collectionID:EanesChallenge" not in result.loc[0, "semantic_text"]