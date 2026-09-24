from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.analytics.preference_profile import build_preference_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "goodreads_books.csv"
)


def get_input_file() -> Path:
    """
    Use a CSV path supplied on the command line.

    If no path is supplied, use the default Goodreads dataset.
    """

    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])

        if not input_file.is_absolute():
            input_file = PROJECT_ROOT / input_file

        return input_file

    return DEFAULT_INPUT_FILE


def print_rating_distribution(df: pd.DataFrame) -> None:
    """Print the distribution of Goodreads ratings."""

    print("Ratings:")

    if "user_rating" in df.columns:
        ratings = df["user_rating"]

        print(
            ratings
            .value_counts(dropna=False)
            .sort_index()
        )

    elif "rating" in df.columns:
        ratings = df["rating"]

        print(
            ratings
            .value_counts(dropna=False)
            .sort_index()
        )

    else:
        print("No rating column found.")

    print()


def print_dnf_books(df: pd.DataFrame) -> None:
    """Print books marked as did-not-finish."""

    print("Shelves containing DNF:")

    if "shelves" not in df.columns:
        print("No shelves column found.")
        print()
        return

    shelves = df["shelves"].fillna("")

    dnf = df[
        shelves.str.contains(
            "did-not-finish",
            case=False,
            regex=False,
        )
    ]

    print(f"DNF books: {len(dnf)}")

    if len(dnf) > 0:
        columns = [
            column
            for column in [
                "title",
                "user_rating",
                "rating",
                "shelves",
            ]
            if column in dnf.columns
        ]

        print(
            dnf[columns]
            .to_string(index=False)
        )

    print()


def print_preference_profile(
    df: pd.DataFrame,
) -> None:
    """Print the measured preference evidence profile."""

    profile = build_preference_profile(df)

    print("Preference Evidence Profile")
    print("-" * 40)

    print(f"Total books:          {profile['total_books']}")
    print(f"Rated books:          {profile['rated_books']}")
    print(f"Unrated books:        {profile['unrated_books']}")
    print(f"Positive books:       {profile['positive_books']}")
    print(f"Negative books:       {profile['negative_books']}")
    print(
        f"Negative ratings:     "
        f"{profile['negative_rating_books']}"
    )
    print(f"DNF books:            {profile['dnf_books']}")
    print(f"Neutral books:        {profile['neutral_books']}")

    print()

    print("Percent of total:")
    print(
        f"Rated:                "
        f"{profile['rated_pct']:.1%}"
    )
    print(
        f"Unrated:              "
        f"{profile['unrated_pct']:.1%}"
    )
    print(
        f"Positive:             "
        f"{profile['positive_pct']:.1%}"
    )
    print(
        f"Negative:             "
        f"{profile['negative_pct']:.1%}"
    )

    print()


def main() -> None:
    input_file = get_input_file()

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find input file: {input_file}"
        )

    df = pd.read_csv(input_file)

    print("Preference Data Inspection")
    print("=" * 40)
    print(f"Input: {input_file}")
    print(f"Books: {len(df)}")
    print()

    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()

    print_rating_distribution(df)

    print_dnf_books(df)

    print_preference_profile(df)


if __name__ == "__main__":
    main()