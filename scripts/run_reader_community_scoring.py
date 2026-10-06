"""Run the Reading DNA reader x community scoring experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Project import setup
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.reader_community_scoring import (
    build_reading_dna_rankings,
    score_reader_communities,
)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reading_dna"
    / "reader_community_evidence.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reading_dna"
)

SCORED_OUTPUT = (
    OUTPUT_DIR
    / "reader_community_reading_dna_experiment.csv"
)

RANKINGS_OUTPUT = (
    OUTPUT_DIR
    / "reader_community_reading_dna_rankings.csv"
)


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

def print_rankings(rankings: pd.DataFrame) -> None:
    """Print the Reading DNA rankings for inspection."""

    print()
    print("=" * 72)
    print("READING DNA RANKINGS")
    print("=" * 72)

    if rankings.empty:
        print("No qualifying rankings were generated.")
        return

    for reader_id in sorted(rankings["reader_id"].unique()):

        print()
        print(f"READER: {reader_id}")

        for signal in [
            "preference",
            "avoidance",
            "exploration",
        ]:

            subset = rankings[
                (rankings["reader_id"] == reader_id)
                & (rankings["signal"] == signal)
            ].copy()

            print()
            print(signal.upper())

            if subset.empty:
                print("  No qualifying communities.")
                continue

            display_columns = [
                "rank",
                "community_id",
                "score",
                "signal_evidence_strength",
                "evidence_strength",
                "preference_evidence_count",
                "exposure_only_book_count",
                "positive_book_count",
                "negative_book_count",
            ]

            print(
                subset[display_columns].to_string(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Reader/community evidence not found: {INPUT_PATH}"
        )

    evidence = pd.read_csv(INPUT_PATH)

    print("=" * 72)
    print("READING DNA SIGNAL EXPERIMENT")
    print("=" * 72)

    print(f"Input rows: {len(evidence)}")
    print(f"Readers: {evidence['reader_id'].nunique()}")
    print(f"Communities: {evidence['community_id'].nunique()}")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    scored = score_reader_communities(
        evidence,
        prior_strength=5.0,
        prior_preference=0.25,
    )

    # ------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------

    rankings = build_reading_dna_rankings(
        scored,
        top_n=10,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        SCORED_OUTPUT,
        index=False,
    )

    rankings.to_csv(
        RANKINGS_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    invalid_avoidance = scored[
        (scored["negative_book_count"] == 0)
        & (scored["avoidance_strength"] > 0)
    ]

    if not invalid_avoidance.empty:
        raise AssertionError(
            "Avoidance validation failed: communities with zero negative "
            "evidence received positive avoidance strength."
        )

    invalid_avoidance_rankings = rankings[
        (rankings["signal"] == "avoidance")
        & (rankings["negative_book_count"] == 0)
    ]

    if not invalid_avoidance_rankings.empty:
        raise AssertionError(
            "Avoidance ranking validation failed: a zero-negative "
            "community appeared in avoidance rankings."
        )

    # ------------------------------------------------------------------
    # Print rankings
    # ------------------------------------------------------------------

    print_rankings(rankings)

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print("Avoidance validation: PASS")

    print(
        "Zero-negative communities with positive avoidance score: "
        f"{len(invalid_avoidance)}"
    )

    print(
        "Zero-negative communities ranked for avoidance: "
        f"{len(invalid_avoidance_rankings)}"
    )

    print()
    print(f"Saved scored data: {SCORED_OUTPUT}")
    print(f"Saved rankings: {RANKINGS_OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()