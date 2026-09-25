
import ast
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.metadata_normalization import build_book_text

MODEL_NAME = "all-MiniLM-L6-v2"
POSITIVE_RATING_MIN = 4

READER_PATHS = {
    "you": PROJECT_ROOT / "data" / "raw" / "you_books_enriched.csv",
    "sarah": PROJECT_ROOT / "data" / "raw" / "sarah_books_enriched.csv",
    "shannon": PROJECT_ROOT / "data" / "raw" / "shannon_books_enriched.csv",
}


def parse_subjects(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [value]


def load_reader(reader):
    path = READER_PATHS[reader]
    df = pd.read_csv(path)
    ratings = pd.to_numeric(df["user_rating"], errors="coerce").replace(0, np.nan)
    df["rating_numeric"] = ratings
    df["is_positive"] = ratings >= POSITIVE_RATING_MIN
    return df


def build_texts(df):
    texts = []
    for _, row in df.iterrows():
        texts.append(
            build_book_text(
                title=row.get("title"),
                author=row.get("author"),
                subjects=parse_subjects(row.get("subjects")),
                description=row.get("description"),
            )
        )
    return texts


def calculate_metrics(embeddings, positive_indices):
    positive_embeddings = embeddings[positive_indices]
    centroid = positive_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    within = cosine_similarity(positive_embeddings)
    upper = within[np.triu_indices_from(within, k=1)]

    distances = cosine_similarity(positive_embeddings, centroid.reshape(1, -1)).ravel()

    return {
        "positive_count": len(positive_indices),
        "mean_pairwise_similarity": float(np.mean(upper)),
        "median_pairwise_similarity": float(np.median(upper)),
        "std_pairwise_similarity": float(np.std(upper)),
        "mean_centroid_similarity": float(np.mean(distances)),
        "median_centroid_similarity": float(np.median(distances)),
        "std_centroid_similarity": float(np.std(distances)),
        "min_centroid_similarity": float(np.min(distances)),
        "max_centroid_similarity": float(np.max(distances)),
    }


def make_plots(results_df, reader_data):
    output_dir = PROJECT_ROOT / "data" / "processed" / "preference_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Positive preference cohesion
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(
        x - width / 2,
        results_df["mean_pairwise_similarity"],
        width,
        label="Mean pairwise similarity",
    )
    ax.bar(
        x + width / 2,
        results_df["mean_centroid_similarity"],
        width,
        label="Mean similarity to centroid",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["reader"].str.title())
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Positive preference structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "positive_preference_cohesion.png", dpi=160)
    plt.close(fig)

    # 2. Distribution of similarity to each reader's positive centroid
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for reader, values in reader_data.items():
        ax.hist(values, bins=18, alpha=0.55, label=reader.title())
    ax.set_xlabel("Cosine similarity to positive preference centroid")
    ax.set_ylabel("Number of positive books")
    ax.set_title("How tightly positive books cluster around the reader centroid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "positive_centroid_similarity_distribution.png", dpi=160)
    plt.close(fig)

    # 3. Pairwise similarity distributions
    fig, ax = plt.subplots(figsize=(9, 5.5))
    box_data = []
    labels = []
    for reader in reader_data:
        df = reader_data[reader]
        box_data.append(df)
        labels.append(reader.title())
    ax.boxplot(box_data, tick_labels=labels)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Distribution of positive-book pairwise similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "positive_pairwise_similarity_boxplot.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Sentence-Transformer model.",
    )
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)

    summary_rows = []
    centroid_similarity_values = {}
    pairwise_values = {}

    for reader in ["you", "sarah", "shannon"]:
        print()
        print("=" * 70)
        print(f"Preference structure — {reader}")
        print("=" * 70)

        df = load_reader(reader)
        positive_indices = np.flatnonzero(df["is_positive"].to_numpy())

        print(f"Books: {len(df):,}")
        print(f"Positive books: {len(positive_indices):,}")

        texts = build_texts(df)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings)

        metrics = calculate_metrics(embeddings, positive_indices)
        metrics["reader"] = reader
        summary_rows.append(metrics)

        positive_embeddings = embeddings[positive_indices]
        centroid = positive_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroid_sims = cosine_similarity(
            positive_embeddings, centroid.reshape(1, -1)
        ).ravel()
        centroid_similarity_values[reader] = centroid_sims

        pairwise = cosine_similarity(positive_embeddings)
        pairwise_values[reader] = pairwise[
            np.triu_indices_from(pairwise, k=1)
        ]

        print(f"Mean pairwise similarity: {metrics['mean_pairwise_similarity']:.3f}")
        print(f"Median pairwise similarity: {metrics['median_pairwise_similarity']:.3f}")
        print(f"Mean centroid similarity: {metrics['mean_centroid_similarity']:.3f}")
        print(f"Median centroid similarity: {metrics['median_centroid_similarity']:.3f}")
        print(f"Centroid similarity SD: {metrics['std_centroid_similarity']:.3f}")

    summary = pd.DataFrame(summary_rows)[
        [
            "reader",
            "positive_count",
            "mean_pairwise_similarity",
            "median_pairwise_similarity",
            "std_pairwise_similarity",
            "mean_centroid_similarity",
            "median_centroid_similarity",
            "std_centroid_similarity",
            "min_centroid_similarity",
            "max_centroid_similarity",
        ]
    ]

    output_dir = PROJECT_ROOT / "data" / "processed" / "preference_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "preference_structure_summary.csv", index=False)

    # Save distributions in long form for reproducibility.
    centroid_rows = []
    pairwise_rows = []
    for reader in ["you", "sarah", "shannon"]:
        for value in centroid_similarity_values[reader]:
            centroid_rows.append({"reader": reader, "centroid_similarity": value})
        for value in pairwise_values[reader]:
            pairwise_rows.append({"reader": reader, "pairwise_similarity": value})

    pd.DataFrame(centroid_rows).to_csv(
        output_dir / "positive_centroid_similarity.csv", index=False
    )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "positive_pairwise_similarity.csv", index=False
    )

    make_plots(
        summary,
        {
            reader: centroid_similarity_values[reader]
            for reader in centroid_similarity_values
        },
    )

    # Correct the third plot input: pairwise distributions.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    box_data = [pairwise_values[r] for r in ["you", "sarah", "shannon"]]
    ax.boxplot(box_data, tick_labels=["You", "Sarah", "Shannon"])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Distribution of positive-book pairwise similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "positive_pairwise_similarity_boxplot.png", dpi=160)
    plt.close(fig)

    print()
    print("=" * 70)
    print("PREFERENCE STRUCTURE SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("Saved analysis files to:")
    print(output_dir)


if __name__ == "__main__":
    main()
