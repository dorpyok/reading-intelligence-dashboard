"""
Generate interpretable reader profiles from Model B.

This script creates the human-readable interpretation layer for Model B.

Important:
    This script does NOT modify the recommendation embeddings or clustering.

The interpretation layer:
    1. Uses the same cleaned semantic representation as Model B.
    2. Extracts candidate terms from interest clusters.
    3. Removes generic book/metadata vocabulary.
    4. Suppresses likely author/entity names.
    5. Favors terms that distinguish an interest from the reader's
       other interests.
    6. Returns representative books alongside the keywords.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_B = (
    ROOT
    / "scripts"
    / "run_model_b_multi_interest.py"
)

OUT = (
    ROOT
    / "data"
    / "processed"
    / "recommendation_experiments"
    / "model_b_cleaned"
    / "reader_profiles"
)

READERS = [
    "you",
    "sarah",
    "shannon",
]

SEED = 42

TOP_BOOKS = 10
TOP_KEYWORDS = 8


# ---------------------------------------------------------------------------
# INTERPRETATION FILTERS
# ---------------------------------------------------------------------------

# Generic words that describe books or the reading process rather than
# the reader's interests.
GENERIC_BOOK_TERMS = {
    "book",
    "books",
    "fiction",
    "author",
    "authors",
    "title",
    "novel",
    "novels",
    "story",
    "stories",
    "reader",
    "readers",
    "read",
    "reading",
    "written",
    "writing",
    "published",
    "publication",
    "page",
    "pages",
    "chapter",
    "chapters",
    "new",
    "life",
    "literature",
    "literary",
}


# Goodreads / web / metadata artifacts.
METADATA_NOISE_TERMS = {
    "goodreads",
    "rating",
    "ratings",
    "review",
    "reviews",
    "isbn",
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "br",
    "amp",
}


# Educational metadata is particularly common in Open Library subjects.
EDUCATIONAL_METADATA_TERMS = {
    "grade",
    "grades",
    "level",
    "levels",
    "reading level",
    "grade level",
    "reading grade",
    "lexile",
    "accelerated",
    "reading age",
}


# Terms that are usually not useful as thematic explanations.
GENERIC_RELATIONSHIP_TERMS = {
    "man",
    "woman",
    "men",
    "women",
    "boy",
    "girl",
    "person",
    "people",
}


# ---------------------------------------------------------------------------
# TERM NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_keyword_term(
    term: str,
) -> str:
    """Normalize a candidate keyword."""

    term = str(term).lower().strip()

    term = re.sub(
        r"\s+",
        " ",
        term,
    )

    return term


def is_noise_term(
    term: str,
) -> bool:
    """
    Remove obvious metadata and generic vocabulary.
    """

    term = normalize_keyword_term(
        term
    )

    if not term:
        return True

    # URLs and URL fragments.
    if re.search(
        r"(https?://|www\.|\.com\b|\.org\b|\.net\b)",
        term,
    ):
        return True

    words = set(
        term.split()
    )

    # Explicit metadata artifacts.
    if words.intersection(
        METADATA_NOISE_TERMS
    ):
        return True

    # Exact generic terms.
    if term in GENERIC_BOOK_TERMS:
        return True

    # Educational metadata.
    if term in EDUCATIONAL_METADATA_TERMS:
        return True

    # Generic phrases composed entirely of noise.
    if words and words.issubset(
        GENERIC_BOOK_TERMS
        | METADATA_NOISE_TERMS
        | EDUCATIONAL_METADATA_TERMS
    ):
        return True

    # Very short terms are usually fragments or noise.
    if len(term) <= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# ENTITY / AUTHOR FILTERING
# ---------------------------------------------------------------------------

def tokenize_term(
    term: str,
) -> list[str]:
    """Return normalized word tokens."""

    return re.findall(
        r"[a-z]+",
        term.lower(),
    )


def build_entity_vocabulary(
    dataframe: pd.DataFrame,
) -> set[str]:
    """
    Build a vocabulary of likely author/entity names.

    We use author fields as an explicit source of names rather than trying
    to infer whether every word is a proper noun from the description.

    This allows us to suppress:
        maas
        sarah
        sanderson
        hoover

    when they occur as keyword candidates.

    This affects interpretation only.
    """

    entity_terms = set()

    if "author" not in dataframe.columns:
        return entity_terms

    for value in dataframe[
        "author"
    ].dropna():

        author = str(
            value
        ).strip()

        for token in tokenize_term(
            author
        ):

            if len(token) >= 3:
                entity_terms.add(
                    token
                )

    return entity_terms


def is_likely_entity_term(
    term: str,
    entity_vocabulary: set[str],
) -> bool:
    """
    Suppress candidate terms that are likely author/entity names.

    A phrase is removed when one of its meaningful tokens is an author
    token and the phrase is not otherwise clearly thematic.

    Examples:
        "sarah" -> removed
        "maas" -> removed
        "sarah maas" -> removed

    We deliberately do NOT remove words such as "court" or "kingdom"
    simply because they appear in titles.
    """

    tokens = tokenize_term(
        term
    )

    if not tokens:
        return True

    entity_overlap = (
        set(tokens)
        .intersection(
            entity_vocabulary
        )
    )

    if not entity_overlap:
        return False

    # Single-token author/entity matches should be removed.
    if len(tokens) == 1:
        return True

    # If most of the phrase consists of entity tokens, remove it.
    entity_fraction = (
        len(entity_overlap)
        / len(tokens)
    )

    if entity_fraction >= 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# MODEL B LOADING
# ---------------------------------------------------------------------------

def load_model_b():
    """Load the existing Model B module."""

    spec = importlib.util.spec_from_file_location(
        "model_b",
        MODEL_B,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {MODEL_B}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ---------------------------------------------------------------------------
# SUBJECT PARSING
# ---------------------------------------------------------------------------

def parse_subjects(
    value,
) -> list[str]:
    """Safely parse Open Library subjects."""

    if pd.isna(value):
        return []

    if isinstance(
        value,
        list,
    ):
        return [
            str(item)
            for item in value
        ]

    value = str(
        value
    ).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(
            value
        )

        if isinstance(
            parsed,
            list,
        ):
            return [
                str(item)
                for item in parsed
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [value]


# ---------------------------------------------------------------------------
# SEMANTIC TEXT
# ---------------------------------------------------------------------------

def descriptive_text(
    row,
) -> str:
    """
    Build the same cleaned semantic representation used by Model B.

    This keeps the interpretation layer aligned with the ML representation.
    """

    from src.analytics.metadata_normalization import (
        build_book_text,
    )

    return build_book_text(
        title=row.get(
            "title"
        ),
        author=row.get(
            "author"
        ),
        subjects=parse_subjects(
            row.get(
                "subjects"
            )
        ),
        description=row.get(
            "description"
        ),
    )


# ---------------------------------------------------------------------------
# KEYWORD EXTRACTION
# ---------------------------------------------------------------------------

def keyword_candidates(
    cluster_texts: list[str],
    all_texts: list[str],
    entity_vocabulary: set[str],
) -> list[tuple[str, float]]:
    """
    Find terms that distinguish one interest from the reader's overall
    positive-book corpus.

    This is intentionally an interpretation-only representation.
    """

    if not cluster_texts:
        return []

    if not all_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        all_texts
    )

    terms = vectorizer.get_feature_names_out()

    cluster_matrix = vectorizer.transform(
        cluster_texts
    )

    cluster_scores = np.asarray(
        cluster_matrix.mean(
            axis=0
        )
    ).ravel()

    overall_scores = np.asarray(
        matrix.mean(
            axis=0
        )
    ).ravel()

    epsilon = 1e-6

    # How disproportionately important is a term inside this interest?
    distinctiveness = (
        cluster_scores
        / (
            overall_scores
            + epsilon
        )
    )

    # Avoid terms that barely occur.
    meaningful = (
        cluster_scores
        > 0.02
    )

    candidates = []

    for index, term in enumerate(
        terms
    ):

        normalized = normalize_keyword_term(
            term
        )

        if is_noise_term(
            normalized
        ):
            continue

        if is_likely_entity_term(
            normalized,
            entity_vocabulary,
        ):
            continue

        if not meaningful[
            index
        ]:
            continue

        candidates.append(
            (
                normalized,
                float(
                    distinctiveness[
                        index
                    ]
                ),
            )
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates


def keywords(
    cluster_texts: list[str],
    all_texts: list[str],
    entity_vocabulary: set[str],
) -> list[str]:
    """
    Return cleaned, distinctive keywords for an interest.
    """

    candidates = keyword_candidates(
        cluster_texts=cluster_texts,
        all_texts=all_texts,
        entity_vocabulary=entity_vocabulary,
    )

    selected = []

    for term, _ in candidates:

        # Avoid redundant phrases.
        #
        # Example:
        #     "court"
        #     "court politics"
        #
        # We keep the more informative phrase only when appropriate.
        redundant = False

        for existing in selected:

            if term == existing:
                redundant = True
                break

            if (
                term in existing
                and len(term)
                < len(existing)
            ):
                redundant = True
                break

            if (
                existing in term
                and len(existing)
                < len(term)
            ):
                selected.remove(
                    existing
                )
                break

        if redundant:
            continue

        selected.append(
            term
        )

        if len(selected) >= TOP_KEYWORDS:
            break

    return selected


# ---------------------------------------------------------------------------
# PROFILE GENERATION
# ---------------------------------------------------------------------------

def make_profile(
    reader: str,
    mb,
):
    """Build one reader's interpretable Model B profile."""

    dataframe = mb.load_reader_data(
        reader
    )

    positive_idx = np.flatnonzero(
        dataframe[
            "is_positive"
        ].to_numpy()
    )

    texts = mb.build_book_texts(
        dataframe
    )

    embeddings = mb.generate_embeddings(
        texts,
        mb.DEFAULT_MODEL_NAME,
    )

    positive_embeddings = embeddings[
        positive_idx
    ]

    selected_k, k_scores = mb.choose_k(
        positive_embeddings,
        random_state=SEED,
    )

    model, centroids = (
        mb.fit_interest_model(
            positive_embeddings,
            selected_k,
            random_state=SEED,
        )
    )

    labels = model.labels_

    positive_texts = [
        texts[index]
        for index in positive_idx
    ]

    entity_vocabulary = (
        build_entity_vocabulary(
            dataframe
        )
    )

    interest_rows = []
    book_rows = []

    for cluster_id in range(
        selected_k
    ):

        positions = np.flatnonzero(
            labels == cluster_id
        )

        book_indices = positive_idx[
            positions
        ]

        cluster_embeddings = (
            positive_embeddings[
                positions
            ]
        )

        similarities = cosine_similarity(
            cluster_embeddings,
            centroids[
                cluster_id
            ].reshape(
                1,
                -1,
            ),
        ).ravel()

        order = np.argsort(
            similarities
        )[::-1]

        cluster_text = [
            texts[index]
            for index in book_indices
        ]

        cluster_keywords = keywords(
            cluster_texts=cluster_text,
            all_texts=positive_texts,
            entity_vocabulary=entity_vocabulary,
        )

        representative_books = []

        for rank, local_index in enumerate(
            order[
                :TOP_BOOKS
            ],
            start=1,
        ):

            dataframe_index = int(
                book_indices[
                    local_index
                ]
            )

            row = dataframe.iloc[
                dataframe_index
            ]

            title = str(
                row.get(
                    "title"
                )
                or ""
            )

            representative_books.append(
                title
            )

            book_rows.append(
                {
                    "reader": reader,
                    "selected_k": selected_k,
                    "interest_number": (
                        cluster_id + 1
                    ),
                    "interest_rank": rank,
                    "book_index": (
                        dataframe_index
                    ),
                    "source_book_id": row.get(
                        "source_book_id"
                    ),
                    "title": title,
                    "author": row.get(
                        "author"
                    ),
                    "user_rating": row.get(
                        "user_rating"
                    ),
                    "interest_similarity": float(
                        similarities[
                            local_index
                        ]
                    ),
                }
            )

        interest_rows.append(
            {
                "reader": reader,
                "interest_number": (
                    cluster_id + 1
                ),
                "selected_k": selected_k,
                "book_count": len(
                    book_indices
                ),
                "percent_of_positive_books": (
                    len(book_indices)
                    / len(positive_idx)
                    * 100
                ),
                "keywords": ", ".join(
                    cluster_keywords
                ),
                "representative_books": (
                    " | ".join(
                        representative_books
                    )
                ),
            }
        )

    k_scores.to_csv(
        OUT
        / f"{reader}_interest_k_selection.csv",
        index=False,
    )

    make_map(
        reader,
        positive_embeddings,
        labels,
        centroids,
    )

    make_sizes(
        reader,
        interest_rows,
    )

    make_heatmap(
        reader,
        centroids,
    )

    return (
        interest_rows,
        book_rows,
    )


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------

def make_map(
    reader,
    embeddings,
    labels,
    centroids,
):
    """Create 2D visualization of reader interests."""

    pca = PCA(
        n_components=2,
        random_state=SEED,
    )

    xy = pca.fit_transform(
        embeddings
    )

    centroid_xy = pca.transform(
        centroids
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    for cluster_id in range(
        len(centroids)
    ):

        mask = (
            labels
            == cluster_id
        )

        ax.scatter(
            xy[
                mask,
                0,
            ],
            xy[
                mask,
                1,
            ],
            s=42,
            alpha=0.72,
            label=(
                f"Interest "
                f"{cluster_id + 1}"
            ),
        )

    ax.scatter(
        centroid_xy[
            :,
            0,
        ],
        centroid_xy[
            :,
            1,
        ],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.2,
        label="Interest centroid",
    )

    ax.set_title(
        f"{reader.title()} — "
        "Model B interest profile"
    )

    ax.set_xlabel(
        "PCA dimension 1"
    )

    ax.set_ylabel(
        "PCA dimension 2"
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{reader}_interest_map.png",
        dpi=180,
    )

    plt.close(fig)


def make_sizes(
    reader,
    rows,
):
    """Create interest-size visualization."""

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    labels = [
        f"Interest {row['interest_number']}"
        for row in rows
    ]

    counts = [
        row["book_count"]
        for row in rows
    ]

    ax.bar(
        labels,
        counts,
    )

    ax.set_title(
        f"{reader.title()} — "
        "size of discovered interests"
    )

    ax.set_xlabel(
        "Discovered interest"
    )

    ax.set_ylabel(
        "Positive books"
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{reader}_interest_sizes.png",
        dpi=180,
    )

    plt.close(fig)


def make_heatmap(
    reader,
    centroids,
):
    """Create interest-centroid similarity heatmap."""

    similarity = cosine_similarity(
        centroids
    )

    labels = [
        f"Interest {index + 1}"
        for index in range(
            len(centroids)
        )
    ]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        similarity,
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    ax.set_xticks(
        range(
            len(labels)
        )
    )

    ax.set_yticks(
        range(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_title(
        f"{reader.title()} — "
        "similarity between interests"
    )

    for row in range(
        len(labels)
    ):

        for column in range(
            len(labels)
        ):

            ax.text(
                column,
                row,
                f"{similarity[row, column]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Cosine similarity",
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{reader}_interest_separation.png",
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_b = load_model_b()

    all_interests = []
    all_books = []

    for reader in READERS:

        print(
            f"\n{'=' * 70}"
        )

        print(
            reader.upper()
        )

        print(
            f"{'=' * 70}"
        )

        interest_rows, book_rows = (
            make_profile(
                reader,
                model_b,
            )
        )

        all_interests.extend(
            interest_rows
        )

        all_books.extend(
            book_rows
        )

        for row in interest_rows:

            print(
                f"\nInterest "
                f"{row['interest_number']} "
                f"— "
                f"{row['book_count']} books "
                f"("
                f"{row['percent_of_positive_books']:.1f}"
                f"%)"
            )

            print(
                "Keywords: "
                f"{row['keywords']}"
            )

            print(
                "Representative books:"
            )

            for title in (
                row[
                    "representative_books"
                ].split(" | ")
            ):

                print(
                    f"  - {title}"
                )

    pd.DataFrame(
        all_interests
    ).to_csv(
        OUT
        / "model_b_interest_profiles.csv",
        index=False,
    )

    pd.DataFrame(
        all_books
    ).to_csv(
        OUT
        / "model_b_interest_books.csv",
        index=False,
    )

    print(
        f"\nDone. Outputs: {OUT}"
    )


if __name__ == "__main__":
    main()