from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# General text cleaning
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """
    Clean source text while preserving its semantic meaning.

    This function intentionally does not rewrite or summarize source text.
    It only removes obvious formatting/encoding artifacts.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    # Repair common mojibake when possible.
    try:
        repaired = text.encode("latin1").decode("utf-8")
        text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    try:
        repaired = text.encode("cp1252").decode("utf-8")
        text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Decode HTML entities.
    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# List cleaning
# ---------------------------------------------------------------------------

def clean_list(values: Any) -> list[str]:
    """
    Convert a source value into a cleaned list of strings.

    Supports:
      - Python lists
      - tuples
      - sets
      - pipe/comma/semicolon-delimited strings
      - missing values
    """
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return []

    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        text = clean_text(values)

        if not text:
            return []

        # Open Library subject fields sometimes arrive serialized as strings.
        if "|" in text:
            raw_values = text.split("|")
        elif ";" in text:
            raw_values = text.split(";")
        elif "," in text:
            raw_values = text.split(",")
        else:
            raw_values = [text]

    cleaned: list[str] = []

    for value in raw_values:
        value = clean_text(value)

        if value:
            cleaned.append(value)

    return cleaned


# ---------------------------------------------------------------------------
# Subject normalization
# ---------------------------------------------------------------------------

def normalize_subject(subject: Any) -> str:
    """
    Normalize one Open Library subject without changing its meaning.
    """
    return clean_text(subject)


def is_catalog_noise_subject(subject: str) -> bool:
    """
    Identify subjects that are clearly bibliographic/catalog metadata rather
    than useful semantic concepts.

    This intentionally uses a conservative approach.

    We REMOVE:
      - collection IDs
      - series IDs
      - NYT internal ranking fields
      - ISBN-only values
      - obvious translation/language-material catalog descriptors
      - a small number of obvious physical-format descriptors

    We KEEP ordinary literary, topical, thematic, demographic, and
    relationship-oriented subjects even when they are broad.

    The goal is not to create a genre taxonomy. It is simply to remove
    metadata artifacts while preserving semantic signal.
    """
    if not subject:
        return True

    normalized = subject.strip()
    lowered = normalized.lower()

    # ------------------------------------------------------------------
    # Open Library / catalog identifiers
    # ------------------------------------------------------------------

    if lowered.startswith("collectionid:"):
        return True

    if re.match(r"^\[series:.*\]$", lowered):
        return True

    if lowered.startswith("nyt:"):
        return True

    # ISBN appearing as the entire subject.
    if re.fullmatch(r"isbn(?:[-\s:]*)?\d[\dXx\-\s]*", lowered):
        return True

    # ------------------------------------------------------------------
    # Translation / language-material catalog descriptors
    # ------------------------------------------------------------------

    if re.match(r"^translations?\s+into\b", lowered):
        return True

    if re.search(r"\blanguage materials?\b", lowered):
        return True

    # Examples such as:
    #   "Vietnamese language materials"
    #   "Hindi language materials"
    # are covered above.
    #
    # We deliberately do NOT remove ordinary subjects merely because they
    # happen to be multilingual.

    # ------------------------------------------------------------------
    # Obvious physical/catalog format descriptors
    # ------------------------------------------------------------------

    physical_format_patterns = [
        r"^large type books?$",
        r"^large print books?$",
        r"^easy readers?$",
        r"^audio books?$",
        r"^audio recordings?$",
    ]

    for pattern in physical_format_patterns:
        if re.fullmatch(pattern, lowered):
            return True

    return False


def clean_subjects(values: Any) -> list[str]:
    """
    Clean Open Library subjects conservatively.

    Important:
    This function should preserve useful semantic subjects such as:
        fantasy
        romance
        paranormal
        murder
        angels
        demonology
        college students
        self-realization

    It should remove only clearly identifiable catalog/administrative noise.
    """
    subjects = clean_list(values)

    cleaned: list[str] = []
    seen: set[str] = set()

    for subject in subjects:
        subject = normalize_subject(subject)

        if not subject:
            continue

        if is_catalog_noise_subject(subject):
            continue

        # Case-insensitive deduplication while preserving original wording.
        dedupe_key = subject.casefold()

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        cleaned.append(subject)

    return cleaned


# ---------------------------------------------------------------------------
# Semantic representation
# ---------------------------------------------------------------------------

def build_semantic_text(
    title: Any,
    author: Any,
    description: Any,
    subjects: Any = None,
    people: Any = None,
    places: Any = None,
    times: Any = None,
) -> str:
    """
    Build the semantic representation used for embedding.

    Current intended representation:

        Title
        Author
        Description
        Curated Open Library Subjects

    People, places, and times are accepted as arguments for future
    experimentation but are intentionally NOT included in the semantic
    text yet.

    Keeping them out of the embedding for now prevents noisy extracted
    entities from influencing the semantic representation before they have
    been validated.
    """
    title_text = clean_text(title)
    author_text = clean_text(author)
    description_text = clean_text(description)
    subject_values = clean_subjects(subjects)

    parts: list[str] = []

    if title_text:
        parts.append(f"Title: {title_text}")

    if author_text:
        parts.append(f"Author: {author_text}")

    if description_text:
        parts.append(f"Description: {description_text}")

    if subject_values:
        parts.append(f"Subjects: {', '.join(subject_values)}")

    return "\n".join(parts)


def build_semantic_metadata_record(row: pd.Series) -> dict[str, Any]:
    """
    Build one normalized semantic metadata record from a canonical-book row.
    """
    subjects = clean_subjects(row.get("subjects"))

    people = clean_list(row.get("people"))
    places = clean_list(row.get("places"))
    times = clean_list(row.get("times"))

    semantic_text = build_semantic_text(
        title=row.get("title"),
        author=row.get("author"),
        description=row.get("description"),
        subjects=subjects,
        people=people,
        places=places,
        times=times,
    )

    return {
        "canonical_book_id": row.get("canonical_book_id"),
        "title": clean_text(row.get("title")),
        "author": clean_text(row.get("author")),
        "semantic_text": semantic_text,
        "cleaned_subjects": subjects,
        "subject_count": len(subjects),
        "subject_people_count": len(people),
        "subject_places_count": len(places),
        "subject_times_count": len(times),
    }


def build_semantic_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the normalized semantic metadata dataframe.
    """
    records = [
        build_semantic_metadata_record(row)
        for _, row in df.iterrows()
    ]

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
):
    """
    Generate sentence-transformer embeddings.

    This is intentionally kept separate from metadata normalization so that
    the representation can be validated before expensive embedding work.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    return model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )