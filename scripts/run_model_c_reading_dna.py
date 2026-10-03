from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.analytics.metadata_normalization import (
    clean_text,
)

from src.analytics.reading_dna import (
    add_book_attributes,
    aggregate_reader_attributes,
    build_attribute_combinations,
    build_neighborhood_descriptions,
    build_reader_neighborhood_profile,
    score_reader_attribute_strength,
)

from src.analytics.semantic_neighborhoods import (
    summarize_neighborhood_solution,
)


CANONICAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
)

SEMANTIC_BOOKS_PATH = (
    CANONICAL_DIR
    / "semantic_books.csv"
)

EMBEDDINGS_PATH = (
    CANONICAL_DIR
    / "semantic_embeddings.npy"
)

EMBEDDING_METADATA_PATH = (
    CANONICAL_DIR
    / "semantic_embeddings_metadata.csv"
)

NEIGHBOR_ASSIGNMENTS_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhood_assignments.csv"
)

NEIGHBOR_TUNING_PATH = (
    CANONICAL_DIR
    / "semantic_neighborhood_tuning.csv"
)

READER_BOOK_EVIDENCE_PATH = (
    CANONICAL_DIR
    / "reader_book_evidence.csv"
)

WORK_MAPPING_PATH = (
    CANONICAL_DIR
    / "canonical_work_mapping.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reading_dna"
)


READER_NAMES = [
    "you",
    "sarah",
    "shannon",
]

READER_IDS = {
    "you": "goodreads_you",
    "sarah": "goodreads_sarah",
    "shannon": "goodreads_shannon",
}


def parse_list_value(
    value: object,
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, float) and np.isnan(
        value
    ):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(parsed, list):
            return [
                str(item)
                for item in parsed
                if str(item).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [text]


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name}: missing required "
            f"columns: {sorted(missing)}"
        )


def load_semantic_corpus() -> pd.DataFrame:
    print(
        "Loading canonical semantic corpus..."
    )

    semantic_books = pd.read_csv(
        SEMANTIC_BOOKS_PATH
    )

    require_columns(
        semantic_books,
        {
            "canonical_book_id",
            "title",
            "author",
            "semantic_text",
            "cleaned_subjects",
        },
        "semantic_books",
    )

    semantic_books["title"] = (
        semantic_books["title"]
        .map(clean_text)
    )

    semantic_books["author"] = (
        semantic_books["author"]
        .map(clean_text)
    )

    semantic_books["description"] = (
        semantic_books.get(
            "description",
            pd.Series(
                "",
                index=semantic_books.index,
            ),
        )
        .map(clean_text)
    )

    semantic_books["subjects"] = (
        semantic_books[
            "cleaned_subjects"
        ]
        .apply(parse_list_value)
    )

    semantic_books = (
        add_book_attributes(
            semantic_books
        )
    )

    if (
        semantic_books[
            "canonical_book_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Semantic corpus contains duplicate "
            "canonical_book_id values."
        )

    print(
        f"Semantic books: "
        f"{len(semantic_books):,}"
    )

    return semantic_books


def load_neighborhood_assignments() -> pd.DataFrame:
    print(
        "\nLoading semantic neighborhood "
        "assignments..."
    )

    assignments = pd.read_csv(
        NEIGHBOR_ASSIGNMENTS_PATH
    )

    require_columns(
        assignments,
        {
            "embedding_row",
            "canonical_book_id",
            "title",
            "author",
            "neighborhood_id",
            "is_noise",
        },
        "semantic_neighborhood_assignments",
    )

    if (
        assignments[
            "canonical_book_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Semantic neighborhood assignments "
            "contain duplicate canonical_book_id "
            "values."
        )

    print(
        f"Neighborhood assignments: "
        f"{len(assignments):,}"
    )

    print(
        f"Neighborhoods: "
        f"{assignments.loc[~assignments['is_noise'], 'neighborhood_id'].nunique()}"
    )

    print(
        f"Noise books: "
        f"{int(assignments['is_noise'].sum()):,}"
    )

    return assignments


def load_reader_evidence() -> pd.DataFrame:
    print(
        "\nLoading aggregated reader/book "
        "evidence..."
    )

    evidence = pd.read_csv(
        READER_BOOK_EVIDENCE_PATH
    )

    work_mapping = pd.read_csv(
        WORK_MAPPING_PATH
    )

    require_columns(
        evidence,
        {
            "reader_id",
            "canonical_book_id",
            "canonical_work_id",
            "reading_status",
            "is_read",
            "is_positive",
            "is_negative",
            "is_exposure",
            "is_tbr",
            "is_currently_reading",
            "is_dnf",
            "evidence_strength",
            "preference_signal",
        },
        "reader_book_evidence",
    )

    require_columns(
        work_mapping,
        {
            "canonical_book_id",
            "canonical_work_id",
        },
        "canonical_work_mapping",
    )

    duplicates = evidence.duplicated(
        [
            "reader_id",
            "canonical_book_id",
        ]
    )

    if duplicates.any():
        raise ValueError(
            "reader_book_evidence contains "
            "duplicate reader/book relationships."
        )

    reader_id_to_name = {
        value: key
        for key, value in READER_IDS.items()
    }

    evidence["reader"] = (
        evidence["reader_id"]
        .map(reader_id_to_name)
    )

    if evidence["reader"].isna().any():
        unknown = sorted(
            evidence.loc[
                evidence["reader"].isna(),
                "reader_id",
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown reader IDs: {unknown}"
        )

    mapping_check = (
        work_mapping[
            [
                "canonical_book_id",
                "canonical_work_id",
            ]
        ]
        .drop_duplicates()
    )

    if len(mapping_check) != len(
        work_mapping
    ):
        raise ValueError(
            "canonical_work_mapping contains "
            "duplicate canonical_book_id values."
        )

    merged = evidence.merge(
        mapping_check,
        on="canonical_book_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_mapping",
        ),
    )

    if (
        merged[
            "canonical_work_id_mapping"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Reader evidence contains books "
            "without canonical work mappings."
        )

    if not (
        merged[
            "canonical_work_id"
        ]
        .astype(str)
        ==
        merged[
            "canonical_work_id_mapping"
        ]
        .astype(str)
    ).all():
        raise ValueError(
            "Reader evidence and canonical work "
            "mapping disagree."
        )

    merged = merged.drop(
        columns=[
            "canonical_work_id_mapping"
        ]
    )

    print(
        f"Reader/book relationships: "
        f"{len(merged):,}"
    )

    print(
        f"Readers: "
        f"{merged['reader'].nunique()}"
    )

    return merged


def attach_semantic_layers(
    reader_evidence: pd.DataFrame,
    semantic_books: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    semantic_columns = [
        "canonical_book_id",
        "title",
        "author",
        "attributes",
        "attribute_count",
    ]

    semantic = semantic_books[
        semantic_columns
    ].copy()

    result = reader_evidence.merge(
        semantic,
        on="canonical_book_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_semantic",
        ),
    )

    if result[
        "attributes"
    ].isna().any():
        missing = int(
            result[
                "attributes"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Reader evidence is missing "
            f"semantic attributes: {missing}"
        )

    neighborhood_columns = [
        "canonical_book_id",
        "neighborhood_id",
        "neighborhood_probability",
        "is_noise",
    ]

    neighborhood = assignments[
        neighborhood_columns
    ].copy()

    result = result.merge(
        neighborhood,
        on="canonical_book_id",
        how="left",
        validate="many_to_one",
    )

    if result[
        "neighborhood_id"
    ].isna().any():
        missing = int(
            result[
                "neighborhood_id"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Reader evidence is missing "
            "semantic neighborhood assignments: "
            f"{missing}"
        )

    return result


def validate_reader_selection(
    reader_records: pd.DataFrame,
    reader_selection: list[str],
) -> None:

    available = set(
        reader_records[
            "reader"
        ].unique()
    )

    missing = (
        set(reader_selection)
        - available
    )

    if missing:
        raise ValueError(
            f"Requested readers are missing: "
            f"{sorted(missing)}"
        )


def save_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(
    reader_selection: list[str],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Load semantic book layer
    # ---------------------------------------------------------------

    semantic_books = (
        load_semantic_corpus()
    )

    # ---------------------------------------------------------------
    # Load density-based semantic neighborhoods
    # ---------------------------------------------------------------

    assignments = (
        load_neighborhood_assignments()
    )

    # ---------------------------------------------------------------
    # Load aggregated reader/book evidence
    # ---------------------------------------------------------------

    reader_evidence = (
        load_reader_evidence()
    )

    validate_reader_selection(
        reader_evidence,
        reader_selection,
    )

    # ---------------------------------------------------------------
    # Attach semantic layers
    # ---------------------------------------------------------------

    reader_records = (
        attach_semantic_layers(
            reader_evidence,
            semantic_books,
            assignments,
        )
    )

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MODEL C — READING DNA"
    )
    print("=" * 70)
    print(
        "Density-based semantic neighborhoods "
        "+ multi-label attributes "
        "+ aggregated reader evidence"
    )
    print()

    # ---------------------------------------------------------------
    # Save book-level Reading DNA layer
    # ---------------------------------------------------------------

    book_output = (
        semantic_books[
            [
                "canonical_book_id",
                "title",
                "author",
                "attributes",
                "attribute_count",
            ]
        ]
        .copy()
        .merge(
            assignments[
                [
                    "canonical_book_id",
                    "neighborhood_id",
                    "neighborhood_probability",
                    "is_noise",
                ]
            ],
            on="canonical_book_id",
            how="left",
            validate="one_to_one",
        )
    )

    book_output["attributes"] = (
        book_output[
            "attributes"
        ].apply(
            lambda values: json.dumps(
                values,
                ensure_ascii=False,
            )
        )
    )

    book_output.to_csv(
        OUTPUT_DIR
        / "book_reading_dna.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Neighborhood descriptions
    # ---------------------------------------------------------------

    neighborhood_descriptions = (
        build_neighborhood_descriptions(
            semantic_books.merge(
                assignments[
                    [
                        "canonical_book_id",
                        "neighborhood_id",
                    ]
                ],
                on="canonical_book_id",
                how="left",
                validate="one_to_one",
            )
        )
    )

    neighborhood_descriptions.to_csv(
        OUTPUT_DIR
        / "neighborhood_descriptions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Reader-level outputs
    # ---------------------------------------------------------------

    reader_attributes = []
    reader_combinations = []
    reader_neighborhoods = []

    for reader in reader_selection:

        df = (
            reader_records.loc[
                reader_records[
                    "reader"
                ]
                == reader
            ]
            .copy()
        )

        if df.empty:
            raise ValueError(
                f"No records found for reader: "
                f"{reader}"
            )

        # -----------------------------------------------------------
        # Attribute evidence
        # -----------------------------------------------------------

        attributes = (
            aggregate_reader_attributes(
                df,
                reader,
            )
        )

        attributes = (
            score_reader_attribute_strength(
                attributes
            )
        )

        reader_attributes.append(
            attributes
        )

        # -----------------------------------------------------------
        # Attribute co-occurrence
        # -----------------------------------------------------------

        combinations = (
            build_attribute_combinations(
                df,
                reader,
            )
        )

        reader_combinations.append(
            combinations
        )

        # -----------------------------------------------------------
        # Neighborhood evidence
        # -----------------------------------------------------------

        neighborhood_profile = (
            build_reader_neighborhood_profile(
                df,
                reader,
            )
        )

        reader_neighborhoods.append(
            neighborhood_profile
        )

    # ---------------------------------------------------------------
    # Save reader attributes
    # ---------------------------------------------------------------

    reader_attributes_df = pd.concat(
        reader_attributes,
        ignore_index=True,
    )

    reader_attributes_df.to_csv(
        OUTPUT_DIR
        / "reader_attributes.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save attribute combinations
    # ---------------------------------------------------------------

    combinations_df = pd.concat(
        reader_combinations,
        ignore_index=True,
    )

    combinations_df.to_csv(
        OUTPUT_DIR
        / "reader_attribute_combinations.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save reader neighborhood evidence
    # ---------------------------------------------------------------

    reader_neighborhoods_df = pd.concat(
        reader_neighborhoods,
        ignore_index=True,
    )

    reader_neighborhoods_df.to_csv(
        OUTPUT_DIR
        / "reader_neighborhoods.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Reader summary
    # ---------------------------------------------------------------

    summary_rows = []

    for reader in reader_selection:

        df = reader_records.loc[
            reader_records[
                "reader"
            ]
            == reader
        ]

        attrs = reader_attributes_df.loc[
            reader_attributes_df[
                "reader"
            ]
            == reader
        ]

        neighborhoods = (
            reader_neighborhoods_df.loc[
                reader_neighborhoods_df[
                    "reader"
                ]
                == reader
            ]
        )

        summary_rows.append(
            {
                "reader": reader,
                "reader_id": READER_IDS[
                    reader
                ],
                "reader_book_relationships": (
                    len(df)
                ),
                "canonical_works": int(
                    df[
                        "canonical_work_id"
                    ].nunique()
                ),
                "positive_books": int(
                    df[
                        "is_positive"
                    ].sum()
                ),
                "negative_books": int(
                    df[
                        "is_negative"
                    ].sum()
                ),
                "exposure_books": int(
                    df[
                        "is_exposure"
                    ].sum()
                ),
                "tbr_books": int(
                    df[
                        "is_tbr"
                    ].sum()
                ),
                "currently_reading": int(
                    df[
                        "is_currently_reading"
                    ].sum()
                ),
                "dnf_books": int(
                    df[
                        "is_dnf"
                    ].sum()
                ),
                "unknown_evidence": int(
                    (
                        df[
                            "evidence_strength"
                        ]
                        == "unknown"
                    ).sum()
                ),
                "unique_attributes": int(
                    attrs[
                        "attribute"
                    ].nunique()
                ),
                "neighborhoods_exposed": int(
                    neighborhoods[
                        "neighborhood_id"
                    ].nunique()
                ),
                "noise_books": int(
                    neighborhoods[
                        "is_noise"
                    ].sum()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "model_c_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Neighborhood solution diagnostics
    # ---------------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    neighborhood_labels = (
        assignments
        .sort_values(
            "embedding_row"
        )[
            "neighborhood_id"
        ]
        .to_numpy()
    )

    neighborhood_solution = (
        summarize_neighborhood_solution(
            embeddings,
            neighborhood_labels,
        )
    )

    # ---------------------------------------------------------------
    # Model C configuration
    # ---------------------------------------------------------------

    save_json(
        OUTPUT_DIR
        / "model_c_config.json",
        {
            "model": (
                "Model C — "
                "Multi-Interest + "
                "Multi-Label Reading DNA"
            ),
            "neighborhood_method": (
                "HDBSCAN density-based "
                "semantic neighborhoods"
            ),
            "neighborhood_assignments": str(
                NEIGHBOR_ASSIGNMENTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "neighborhood_tuning": str(
                NEIGHBOR_TUNING_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "semantic_corpus": str(
                SEMANTIC_BOOKS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "embedding_artifact": str(
                EMBEDDINGS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "embedding_metadata": str(
                EMBEDDING_METADATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "embedding_dimensions": int(
                embeddings.shape[1]
            ),
            "reader_evidence_source": (
                "aggregated reader/book evidence"
            ),
            "reader_evidence_artifact": str(
                READER_BOOK_EVIDENCE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "raw_reader_evidence_source": (
                "canonical reader_evidence.csv"
            ),
            "work_identity_source": (
                "canonical_work_mapping"
            ),
            "attribute_source": (
                "curated Open Library subjects "
                "from semantic_books"
            ),
            "evidence_streams": [
                "preference",
                "exposure",
                "intent",
            ],
            "neighborhood_solution": (
                neighborhood_solution
            ),
        },
    )

    # ---------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------

    print()
    print(
        "SEMANTIC NEIGHBORHOOD SUMMARY"
    )
    print("=" * 70)

    for key, value in (
        neighborhood_solution.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "MODEL C READER SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved Reading DNA outputs to:"
    )

    print(
        OUTPUT_DIR
    )


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run Model C Reading DNA "
            "from semantic neighborhoods."
        )
    )

    parser.add_argument(
        "--reader",
        default="all",
        choices=[
            "you",
            "sarah",
            "shannon",
            "all",
        ],
    )

    args = parser.parse_args()

    readers = (
        READER_NAMES
        if args.reader == "all"
        else [args.reader]
    )

    run(readers)


if __name__ == "__main__":
    main()