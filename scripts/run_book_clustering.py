from pathlib import Path

import pandas as pd

from src.analytics.metadata_normalization import (
    build_book_text,
    generate_embeddings,
)
from src.analytics.book_clustering import (
    cluster_books,
    get_cluster_keywords,
    reduce_for_clustering,
    reduce_for_visualization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "goodreads_books.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "book_clusters.csv"


def main():
    print("Reading Intelligence — Book Clustering")
    print("=" * 50)

    df = pd.read_csv(INPUT_FILE)

    print(f"Books loaded: {len(df)}")

    # Build semantic text representation.
    texts = []

    for _, row in df.iterrows():
        text = build_book_text(
            title=row.get("title", ""),
            author=row.get("author", ""),
            description=row.get("description", ""),
        )

        texts.append(text)

    print("Generating semantic embeddings...")

    embeddings = generate_embeddings(texts)

    print(f"Embedding shape: {embeddings.shape}")

    print("Reducing embeddings for clustering...")

    clustering_embeddings = reduce_for_clustering(embeddings)

    print("Running HDBSCAN...")

    clusterer = cluster_books(clustering_embeddings)

    df["cluster_id"] = clusterer.labels_
    df["cluster_probability"] = clusterer.probabilities_

    print()
    print("Cluster summary")
    print("-" * 50)

    print(df["cluster_id"].value_counts().sort_index())

    print()
    print("Generating visualization coordinates...")

    visualization_embeddings = reduce_for_visualization(embeddings)

    df["umap_x"] = visualization_embeddings[:, 0]
    df["umap_y"] = visualization_embeddings[:, 1]

    print()
    print("Generating cluster keywords...")

    keywords = get_cluster_keywords(
        texts,
        clusterer.labels_,
    )

    for cluster_id, terms in keywords.items():
        print()
        print(f"Cluster {cluster_id}:")
        print("  " + ", ".join(terms))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()