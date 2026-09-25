from __future__ import annotations

import re
from html import unescape

import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(value: object) -> str:
    """Clean HTML, repair common encoding corruption, and normalize whitespace."""
    if pd.isna(value):
        return ""

    text = str(value)

    # Repair common UTF-8/Windows-1252 mojibake.
    # Goodreads descriptions can occasionally arrive with UTF-8 bytes
    # decoded using the wrong character encoding.
    mojibake_markers = (
        "Ã",
        "Â",
        "â€",
        "â€™",
        "â€œ",
        "â€”",
        "â€“",
        "â€¦",
    )

    if any(marker in text for marker in mojibake_markers):
        try:
            text = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Keep the original text if it cannot be safely repaired.
            pass

    # Decode HTML entities such as &amp;, &nbsp;, and &quot;.
    text = unescape(text)

    # Remove HTML tags such as <b>, <i>, and <br />.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def build_book_text(
    title: str,
    author: str,
    subjects: list[str] | None = None,
    description: str = "",
) -> str:
    """
    Build a semantic text representation of a book.

    The representation intentionally combines:
    - title
    - author
    - Open Library subjects
    - book description
    """

    subjects = subjects or []

    cleaned_subjects = [
        clean_text(subject)
        for subject in subjects
    ]

    subject_text = ", ".join(
        subject
        for subject in cleaned_subjects
        if subject
    )

    parts = [
        f"Title: {clean_text(title)}",
        f"Author: {clean_text(author)}",
    ]

    if subject_text:
        parts.append(f"Subjects: {subject_text}")

    if description:
        cleaned_description = clean_text(description)

        if cleaned_description:
            parts.append(
                f"Description: {cleaned_description}"
            )

    return "\n".join(parts)


def generate_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
):
    """
    Generate semantic embeddings for a collection of book texts.
    """

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings