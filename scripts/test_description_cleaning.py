from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analytics.metadata_normalization import clean_text


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "you_books_enriched.csv"
)

SAMPLE_SIZE = 20


def main():
    dataframe = pd.read_csv(INPUT_FILE)

    sample = dataframe.head(SAMPLE_SIZE)

    print("=" * 80)
    print("DESCRIPTION CLEANING VALIDATION")
    print("=" * 80)
    print(f"Source: {INPUT_FILE}")
    print(f"Books sampled: {len(sample)}")
    print()

    for index, row in sample.iterrows():
        title = row.get("title", "")
        raw_description = row.get(
            "description",
            "",
        )

        cleaned_description = clean_text(
            raw_description
        )

        print("=" * 80)
        print(f"BOOK {index + 1}: {title}")
        print("=" * 80)

        print()
        print("RAW DESCRIPTION:")
        print(raw_description)

        print()
        print("CLEANED DESCRIPTION:")
        print(cleaned_description)

        print()


if __name__ == "__main__":
    main()