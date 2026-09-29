from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.analytics.semantic_embeddings import (
    EMBEDDING_DIMENSIONS,
    generate_embeddings,
    load_embedding_model,
    load_semantic_books,
    save_embeddings,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SEMANTIC_BOOKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_books.csv"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_embeddings_metadata.csv"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Generating semantic embeddings")
    print("=" * 70)

    print("\nSemantic books:")
    print(f"  {SEMANTIC_BOOKS_PATH}")

    print("\nEmbeddings:")
    print(f"  {EMBEDDINGS_PATH}")

    print("\nMetadata:")
    print(f"  {METADATA_PATH}")

    if not SEMANTIC_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Semantic books file not found:\n{SEMANTIC_BOOKS_PATH}"
        )

    # -----------------------------------------------------------------------
    # Load semantic corpus
    # -----------------------------------------------------------------------

    semantic_books = load_semantic_books(
        SEMANTIC_BOOKS_PATH
    )

    print(
        f"\nLoaded {len(semantic_books):,} semantic book records."
    )

    if "semantic_text" not in semantic_books.columns:
        raise ValueError(
            "Semantic corpus is missing the required "
            "'semantic_text' column."
        )

    if "canonical_book_id" not in semantic_books.columns:
        raise ValueError(
            "Semantic corpus is missing the required "
            "'canonical_book_id' column."
        )

    # -----------------------------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------------------------

    print("\nLoading embedding model...")

    model = load_embedding_model()

    print("Embedding model loaded.")

    # -----------------------------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------------------------

    print(
        f"\nGenerating embeddings "
        f"(batch size={BATCH_SIZE})..."
    )

    embeddings = generate_embeddings(
        semantic_books["semantic_text"],
        model=model,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    print(
        f"\nGenerated embedding matrix: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------------------------
    # Validate generated embeddings
    # -----------------------------------------------------------------------

    expected_shape = (
        len(semantic_books),
        EMBEDDING_DIMENSIONS,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrix contains NaN or infinite values."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

    # -----------------------------------------------------------------------
    # Save embeddings
    # -----------------------------------------------------------------------

    save_embeddings(
        embeddings=embeddings,
        dataframe=semantic_books,
        embedding_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    )

    print("\nSaved embedding artifacts.")

    # -----------------------------------------------------------------------
    # Final verification
    # -----------------------------------------------------------------------

    saved_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    saved_metadata = pd.read_csv(
        METADATA_PATH
    )

    if saved_embeddings.shape != expected_shape:
        raise ValueError(
            "Saved embedding shape does not match "
            f"expected shape {expected_shape}: "
            f"{saved_embeddings.shape}"
        )

    if len(saved_metadata) != len(semantic_books):
        raise ValueError(
            "Saved metadata row count does not match "
            f"semantic corpus: "
            f"{len(saved_metadata)}"
        )

    if (
        saved_metadata["canonical_book_id"].nunique()
        != len(saved_metadata)
    ):
        raise ValueError(
            "Saved embedding metadata contains "
            "duplicate canonical book IDs."
        )

    if (
        saved_metadata["canonical_book_id"].tolist()
        != semantic_books["canonical_book_id"].tolist()
    ):
        raise ValueError(
            "Embedding metadata order does not match "
            "the semantic corpus order."
        )

    print("\nFinal validation:")
    print(
        f"  Embedding shape: "
        f"{saved_embeddings.shape}"
    )
    print(
        f"  Metadata rows: "
        f"{len(saved_metadata):,}"
    )
    print(
        f"  Unique canonical IDs: "
        f"{saved_metadata['canonical_book_id'].nunique():,}"
    )
    print(
        f"  Finite values: "
        f"{np.isfinite(saved_embeddings).all()}"
    )
    print(
        f"  Duplicate IDs: "
        f"{saved_metadata['canonical_book_id'].duplicated().sum()}"
    )
    print(
        f"  Dimensions: "
        f"{saved_embeddings.shape[1]}"
    )

    print("\n" + "=" * 70)
    print("Semantic embedding generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()