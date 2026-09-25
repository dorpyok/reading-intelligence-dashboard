from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metadata_normalization import clean_text
from src.analytics.reading_dna import (
    ClusterConfig,
    add_book_attributes,
    aggregate_reader_attributes,
    assign_clusters,
    build_attribute_combinations,
    build_cluster_descriptions,
    build_cluster_texts,
    build_reader_cluster_profile,
    deduplicate_books,
    parse_list_value,
    select_final_k,
    score_reader_attribute_strength,
    tune_cluster_count,
)


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_RANDOM_STATE = 42

READER_PATHS = {
    "you": PROJECT_ROOT / "data" / "raw" / "you_books_enriched.csv",
    "sarah": PROJECT_ROOT / "data" / "raw" / "sarah_books_enriched.csv",
    "shannon": PROJECT_ROOT / "data" / "raw" / "shannon_books_enriched.csv",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "reading_dna"


def parse_subjects(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    return [text]


def load_reader(reader: str) -> pd.DataFrame:
    path = READER_PATHS[reader]
    if not path.exists():
        raise FileNotFoundError(f"Reader file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "source_book_id",
        "title",
        "author",
        "description",
        "subjects",
        "user_rating",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{reader}: missing required columns: {sorted(missing)}")

    if "reading_status" not in df.columns:
        # Model C can run against enriched V1 data, but canonical status
        # should be preferred when it is available.
        df["reading_status"] = ""

    return df


def prepare_reader(df: pd.DataFrame, reader: str) -> pd.DataFrame:
    result = df.copy()
    result["reader"] = reader
    result["subjects"] = result["subjects"].apply(parse_subjects)
    result["description"] = result["description"].apply(clean_text)
    result["title"] = result["title"].apply(clean_text)
    result["author"] = result["author"].apply(clean_text)
    return add_book_attributes(result)


def build_pooled_corpus(readers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pooled = pd.concat(readers.values(), ignore_index=True)
    pooled = deduplicate_books(pooled)
    return pooled.reset_index(drop=True)


def generate_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run(reader_selection: list[str], model_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    readers = {
        reader: prepare_reader(load_reader(reader), reader)
        for reader in reader_selection
    }

    pooled = build_pooled_corpus(readers)

    print("=" * 70)
    print("MODEL C — MULTI-INTEREST + MULTI-LABEL READING DNA")
    print("=" * 70)
    for reader, df in readers.items():
        print(f"{reader.title():8} {len(df):,} books")
    print(f"Pooled unique semantic books: {len(pooled):,}")

    print()
    print("Building semantic book representations...")
    texts = build_cluster_texts(pooled)

    print(f"Generating embeddings with {model_name}...")
    embeddings = generate_embeddings(texts, model_name)
    print(f"Embedding shape: {embeddings.shape}")

    print()
    print("Tuning cluster count...")
    tuning = tune_cluster_count(
        embeddings,
        ClusterConfig(
            min_k=6,
            max_k=18,
            n_init=20,
            random_state=DEFAULT_RANDOM_STATE,
        ),
    )
    tuning.to_csv(
        OUTPUT_DIR / "cluster_tuning.csv",
        index=False,
    )

    selected_k = select_final_k(
        tuning,
        min_stability=0.70,
        min_cluster_pct=0.01,
    )

    selected_row = tuning.loc[tuning["k"] == selected_k].iloc[0]

    print()
    print("Cluster tuning:")
    print(tuning.to_string(index=False))
    print()
    print(
        f"Selected final K: {selected_k} "
        f"(silhouette={selected_row['silhouette_mean']:.3f}, "
        f"stability={selected_row['stability_ari_mean']:.3f}, "
        f"min_cluster={int(selected_row['min_cluster_size'])})"
    )

    labels, centroids, model = assign_clusters(
        embeddings,
        selected_k,
        random_state=DEFAULT_RANDOM_STATE,
        n_init=20,
    )

    pooled["cluster_id"] = labels

    # ---------------------------------------------------------------
    # Pooled book-level model output
    # ---------------------------------------------------------------
    book_output = pooled.copy()
    book_output["attributes"] = book_output["attributes"].apply(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    book_output.to_csv(
        OUTPUT_DIR / "book_reading_dna.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Cluster descriptions
    # ---------------------------------------------------------------
    cluster_descriptions = build_cluster_descriptions(
        pooled,
        labels,
        top_terms=10,
    )
    cluster_descriptions.to_csv(
        OUTPUT_DIR / "cluster_descriptions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Reader-level outputs
    # ---------------------------------------------------------------
    reader_attributes = []
    reader_combinations = []
    reader_clusters = []

    for reader, df in readers.items():
        # Reattach the pooled cluster assignment by source-level ID.
        cluster_lookup = pooled.set_index("source_book_id")["cluster_id"]
        df = df.copy()
        df["cluster_id"] = df["source_book_id"].map(cluster_lookup)

        attrs = aggregate_reader_attributes(df, reader)
        attrs = score_reader_attribute_strength(attrs)
        reader_attributes.append(attrs)

        combinations = build_attribute_combinations(df, reader)
        reader_combinations.append(combinations)

        cluster_profile = build_reader_cluster_profile(
            df,
            df["cluster_id"].to_numpy(),
            reader,
        )
        reader_clusters.append(cluster_profile)

    reader_attributes_df = pd.concat(
        reader_attributes,
        ignore_index=True,
    )
    reader_attributes_df.to_csv(
        OUTPUT_DIR / "reader_attributes.csv",
        index=False,
    )

    combinations_df = pd.concat(
        reader_combinations,
        ignore_index=True,
    )
    combinations_df.to_csv(
        OUTPUT_DIR / "reader_attribute_combinations.csv",
        index=False,
    )

    reader_clusters_df = pd.concat(
        reader_clusters,
        ignore_index=True,
    )
    reader_clusters_df.to_csv(
        OUTPUT_DIR / "reader_clusters.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary_rows = []
    for reader in reader_selection:
        df = readers[reader]
        attrs = reader_attributes_df[
            reader_attributes_df["reader"] == reader
        ]

        summary_rows.append(
            {
                "reader": reader,
                "books": len(df),
                "books_with_attributes": int(
                    (df["attribute_count"] > 0).sum()
                ),
                "unique_attributes": int(attrs["attribute"].nunique()),
                "clusters": selected_k,
                "clustered_books": len(df),
                "positive_books": int(
                    (pd.to_numeric(df["user_rating"], errors="coerce") >= 4).sum()
                ),
                "intent_books": int(
                    df["reading_status"].astype(str).str.lower().eq("to_read").sum()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "model_c_summary.csv",
        index=False,
    )

    save_json(
        OUTPUT_DIR / "model_c_config.json",
        {
            "model": "Model C — Multi-Interest + Multi-Label Reading DNA",
            "embedding_model": model_name,
            "cluster_algorithm": "KMeans",
            "cluster_count": selected_k,
            "cluster_tuning_min_k": 6,
            "cluster_tuning_max_k": 18,
            "cluster_selection": {
                "minimum_stability_ari": 0.70,
                "minimum_cluster_fraction": 0.01,
                "primary_metric": "silhouette_mean",
                "tie_break": "lower_k",
            },
            "attribute_source": "Open Library subjects",
            "evidence_streams": [
                "preference",
                "exposure",
                "intent",
            ],
            "weighting": None,
            "random_state": DEFAULT_RANDOM_STATE,
        },
    )

    print()
    print("=" * 70)
    print("MODEL C SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print(f"Saved Model C outputs to: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Model C Reading DNA."
    )
    parser.add_argument(
        "--reader",
        default="all",
        choices=["you", "sarah", "shannon", "all"],
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    args = parser.parse_args()

    readers = (
        ["you", "sarah", "shannon"]
        if args.reader == "all"
        else [args.reader]
    )

    run(readers, args.model_name)


if __name__ == "__main__":
    main()
