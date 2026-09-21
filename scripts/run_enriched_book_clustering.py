from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.enrichment.openlibrary import OpenLibraryClient
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

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "goodreads_books.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ENRICHED_FILE = (
    OUTPUT_DIR
    / "openlibrary_enriched_books.csv"
)

CLUSTERS_FILE = (
    OUTPUT_DIR
    / "book_clusters_enriched.csv"
)

UMAP_PLOT = (
    OUTPUT_DIR
    / "book_clusters_umap.png"
)

CLUSTER_SIZE_PLOT = (
    OUTPUT_DIR
    / "book_cluster_sizes.png"
)


def enrich_books(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich Goodreads books with Open Library metadata.

    Goodreads remains the source of truth for the
    original book record. Open Library fields are
    added with explicit prefixes where needed to
    prevent duplicate column names.
    """

    client = OpenLibraryClient()

    enrichment_records = []

    total = len(df)

    for index, row in df.iterrows():

        if (
            (index + 1) % 25 == 0
            or index == 0
        ):

            print(
                f"Enriching book "
                f"{index + 1}/{total}..."
            )

        isbn = row.get("isbn")

        if pd.isna(isbn):
            isbn = None

        enrichment = client.match_book(
            title=str(
                row.get(
                    "title",
                    "",
                )
            ),
            author=str(
                row.get(
                    "author",
                    "",
                )
            ),
            isbn=isbn,
        )

        record = asdict(
            enrichment
        )

        # ----------------------------------------------------------
        # Prevent duplicate column names.
        #
        # Goodreads already owns "title".
        # Open Library's title becomes "openlibrary_title".
        # ----------------------------------------------------------

        if "title" in record:

            record["openlibrary_title"] = (
                record.pop("title")
            )

        enrichment_records.append(
            record
        )

    enrichment_df = pd.DataFrame(
        enrichment_records
    )

    # --------------------------------------------------------------
    # Prefix Open Library fields that could otherwise conflict
    # with source-level fields.
    # --------------------------------------------------------------

    rename_columns = {
        "source": "openlibrary_source",
    }

    enrichment_df = (
        enrichment_df.rename(
            columns=rename_columns
        )
    )

    # --------------------------------------------------------------
    # Combine Goodreads + Open Library
    # --------------------------------------------------------------

    combined_df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            enrichment_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return combined_df


def build_enriched_text(
    df: pd.DataFrame,
) -> list[str]:
    """
    Build semantic representations using
    Goodreads + Open Library metadata.
    """

    texts = []

    for _, row in df.iterrows():

        subjects = row.get(
            "subjects",
            [],
        )

        if not isinstance(
            subjects,
            list,
        ):
            subjects = []

        text = build_book_text(
            title=row.get(
                "title",
                "",
            ),
            author=row.get(
                "author",
                "",
            ),
            subjects=subjects,
            description=row.get(
                "description",
                "",
            ),
        )

        texts.append(
            text
        )

    return texts


def save_umap_plot(
    df: pd.DataFrame,
) -> None:
    """
    Save a temporary 2D visualization
    of the semantic book space.
    """

    plt.figure(
        figsize=(12, 8)
    )

    scatter = plt.scatter(
        df["umap_x"],
        df["umap_y"],
        c=df["cluster_id"],
        alpha=0.75,
        s=35,
    )

    plt.title(
        "Reading Intelligence — "
        "Semantic Book Clusters"
    )

    plt.xlabel(
        "UMAP Dimension 1"
    )

    plt.ylabel(
        "UMAP Dimension 2"
    )

    plt.colorbar(
        scatter,
        label="Cluster ID (-1 = outlier)",
    )

    plt.tight_layout()

    plt.savefig(
        UMAP_PLOT,
        dpi=150,
    )

    plt.close()


def save_cluster_size_plot(
    df: pd.DataFrame,
) -> None:
    """
    Save a temporary chart showing
    cluster sizes.
    """

    counts = (
        df["cluster_id"]
        .value_counts()
        .sort_index()
    )

    labels = [
        (
            "Outliers"
            if cluster_id == -1
            else str(cluster_id)
        )
        for cluster_id in counts.index
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        labels,
        counts.values,
    )

    plt.title(
        "Reading Intelligence — "
        "Books per Cluster"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Number of Books"
    )

    plt.tight_layout()

    plt.savefig(
        CLUSTER_SIZE_PLOT,
        dpi=150,
    )

    plt.close()


def main():

    print(
        "Reading Intelligence — "
        "Enriched Book Clustering"
    )

    print(
        "=" * 60
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # LOAD GOODREADS DATA
    # ==============================================================

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Books loaded: {len(df)}"
    )

    # ==============================================================
    # STEP 1 — OPEN LIBRARY ENRICHMENT
    # ==============================================================

    print()

    print(
        "Step 1: Open Library enrichment"
    )

    print(
        "-" * 60
    )

    enriched_df = enrich_books(
        df
    )

    enriched_df.to_csv(
        ENRICHED_FILE,
        index=False,
    )

    print()

    print(
        f"Enriched data saved to: "
        f"{ENRICHED_FILE}"
    )

    # ==============================================================
    # STEP 2 — BUILD SEMANTIC REPRESENTATIONS
    # ==============================================================

    print()

    print(
        "Step 2: Building semantic representations"
    )

    print(
        "-" * 60
    )

    texts = build_enriched_text(
        enriched_df
    )

    print(
        "Generating semantic embeddings..."
    )

    embeddings = generate_embeddings(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ==============================================================
    # STEP 3 — UMAP + HDBSCAN
    # ==============================================================

    print()

    print(
        "Step 3: UMAP + HDBSCAN"
    )

    print(
        "-" * 60
    )

    clustering_embeddings = (
        reduce_for_clustering(
            embeddings
        )
    )

    clusterer = cluster_books(
        clustering_embeddings
    )

    enriched_df[
        "cluster_id"
    ] = clusterer.labels_

    enriched_df[
        "cluster_probability"
    ] = (
        clusterer.probabilities_
    )

    # ==============================================================
    # STEP 4 — 2D VISUALIZATION
    # ==============================================================

    print()

    print(
        "Step 4: Creating visualization coordinates"
    )

    print(
        "-" * 60
    )

    visualization_embeddings = (
        reduce_for_visualization(
            embeddings
        )
    )

    enriched_df[
        "umap_x"
    ] = visualization_embeddings[
        :, 0
    ]

    enriched_df[
        "umap_y"
    ] = visualization_embeddings[
        :, 1
    ]

    # ==============================================================
    # STEP 5 — CLUSTER DESCRIPTORS
    # ==============================================================

    print()

    print(
        "Step 5: Generating cluster descriptors"
    )

    print(
        "-" * 60
    )

    keywords = get_cluster_keywords(
        texts,
        clusterer.labels_,
    )

    for cluster_id, terms in (
        keywords.items()
    ):

        print()

        print(
            f"Cluster {cluster_id}:"
        )

        print(
            "  "
            + ", ".join(
                terms
            )
        )

    # ==============================================================
    # CLUSTER SUMMARY
    # ==============================================================

    print()

    print(
        "Cluster summary"
    )

    print(
        "-" * 60
    )

    print(
        enriched_df[
            "cluster_id"
        ]
        .value_counts()
        .sort_index()
    )

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    enriched_df.to_csv(
        CLUSTERS_FILE,
        index=False,
    )

    save_umap_plot(
        enriched_df
    )

    save_cluster_size_plot(
        enriched_df
    )

    # ==============================================================
    # COMPLETE
    # ==============================================================

    print()

    print(
        "Files created:"
    )

    print(
        f"  {CLUSTERS_FILE}"
    )

    print(
        f"  {UMAP_PLOT}"
    )

    print(
        f"  {CLUSTER_SIZE_PLOT}"
    )

    print()

    print(
        "Enriched clustering complete."
    )


if __name__ == "__main__":
    main()