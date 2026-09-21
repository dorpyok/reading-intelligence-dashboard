from __future__ import annotations

import re
from html import unescape

import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(value: object) -> str:
    """Clean HTML and normalize whitespace."""
    if pd.isna(value):
        return ""

    text = unescape(str(value))

    # Remove basic HTML tags.
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

    subject_text = ", ".join(
        clean_text(subject)
        for subject in subjects
        if clean_text(subject)
    )

    parts = [
        f"Title: {clean_text(title)}",
        f"Author: {clean_text(author)}",
    ]

    if subject_text:
        parts.append(f"Subjects: {subject_text}")

    if description:
        parts.append(f"Description: {clean_text(description)}")

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