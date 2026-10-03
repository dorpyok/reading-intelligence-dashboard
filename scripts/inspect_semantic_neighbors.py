from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEIGHBORHOODS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "semantic_neighborhoods.csv"
)


BOOKS_TO_INSPECT = [
    "House of Earth and Blood",
    "Kill for Me, Kill for You",
    "Red Rising",
    "Salt: A World History",
    "Discovery in Great Sand Dunes National Park",
]


def find_book_matches(
    neighborhoods: pd.DataFrame,
    title_query: str,
) -> pd.DataFrame:
    mask = neighborhoods["query_title"].str.contains(
        title_query,
        case=False,
        na=False,
        regex=False,
    )

    return neighborhoods.loc[mask].copy()


def main() -> None:
    print("Loading semantic neighborhood graph...")

    neighborhoods = pd.read_csv(NEIGHBORHOODS_PATH)

    print(f"Rows: {len(neighborhoods):,}")

    print()
    print("=" * 80)
    print("SEMANTIC NEAREST-NEIGHBOR INSPECTION")
    print("=" * 80)

    for title_query in BOOKS_TO_INSPECT:

        matches = find_book_matches(
            neighborhoods,
            title_query,
        )

        if matches.empty:
            print()
            print(f"NOT FOUND: {title_query}")
            continue

        query_title = matches.iloc[0]["query_title"]
        query_author = matches.iloc[0]["query_author"]

        print()
        print("=" * 80)
        print(f"QUERY: {query_title}")
        print(f"AUTHOR: {query_author}")
        print("=" * 80)

        display = matches[
            [
                "rank",
                "neighbor_title",
                "neighbor_author",
                "similarity",
            ]
        ].copy()

        print(
            display
            .sort_values("rank")
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()