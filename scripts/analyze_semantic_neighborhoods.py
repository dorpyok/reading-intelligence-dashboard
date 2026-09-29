from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.analytics.semantic_neighborhoods import (
    DEFAULT_K,
    find_nearest_neighbors,
    load_embedding_artifacts,
    summarize_neighborhoods,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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

NEIGHBORHOODS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhoods.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhood_summary.csv"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = DEFAULT_K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Semantic Neighborhood Analysis")
    print("=" * 70)

    print("\nLoading embedding artifacts...")

    embeddings, metadata = load_embedding_artifacts(
        embedding_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    )

    print(
        f"Loaded {len(metadata):,} books."
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"\nFinding top {K} semantic neighbors "
        f"for every book..."
    )

    neighborhoods = find_nearest_neighbors(
        embeddings=embeddings,
        metadata=metadata,
        k=K,
    )

    print(
        f"Generated {len(neighborhoods):,} "
        f"book-neighbor relationships."
    )

    # -----------------------------------------------------------------------
    # Save detailed neighborhoods
    # -----------------------------------------------------------------------

    NEIGHBORHOODS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    neighborhoods.to_csv(
        NEIGHBORHOODS_PATH,
        index=False,
    )

    print(
        f"\nSaved neighborhoods:"
        f"\n  {NEIGHBORHOODS_PATH}"
    )

    # -----------------------------------------------------------------------
    # Summarize neighborhood structure
    # -----------------------------------------------------------------------

    summary = summarize_neighborhoods(
        neighborhoods
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        f"\nSaved neighborhood summary:"
        f"\n  {SUMMARY_PATH}"
    )

    # -----------------------------------------------------------------------
    # Overall diagnostics
    # -----------------------------------------------------------------------

    print("\nNeighborhood diagnostics:")

    print(
        f"  Books: "
        f"{summary['canonical_book_id'].nunique():,}"
    )

    print(
        f"  Neighbors per book: "
        f"{summary['neighbor_count'].median():.0f}"
    )

    print(
        f"  Mean similarity: "
        f"{summary['mean_similarity'].mean():.4f}"
    )

    print(
        f"  Median book mean similarity: "
        f"{summary['mean_similarity'].median():.4f}"
    )

    print(
        f"  Minimum similarity observed: "
        f"{summary['min_similarity'].min():.4f}"
    )

    print(
        f"  Maximum similarity observed: "
        f"{summary['max_similarity'].max():.4f}"
    )

    # -----------------------------------------------------------------------
    # Example neighborhoods
    # -----------------------------------------------------------------------

    print("\nExample semantic neighborhoods:")

    example_books = (
        metadata
        .sort_values("title")
        .head(5)
    )

    for _, book in example_books.iterrows():

        book_id = book["canonical_book_id"]

        book_neighbors = neighborhoods[
            neighborhoods["canonical_book_id"]
            == book_id
        ].sort_values("neighbor_rank")

        print(
            f"\n{book['title']} "
            f"by {book['author']}"
        )

        for _, neighbor in book_neighbors.head(5).iterrows():
            print(
                f"  {neighbor['neighbor_rank']:>2}. "
                f"{neighbor['neighbor_title']} "
                f"— "
                f"{neighbor['cosine_similarity']:.3f}"
            )

    print("\n" + "=" * 70)
    print("Semantic neighborhood analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()