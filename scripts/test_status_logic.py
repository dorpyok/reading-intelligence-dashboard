import pandas as pd
from pathlib import Path


def split_shelves(value):
    if pd.isna(value) or not str(value).strip():
        return []

    return [
        shelf.strip()
        for shelf in str(value).split(",")
        if shelf.strip()
    ]


def derive_status(row):
    shelves = split_shelves(row.get("shelves"))

    # Strongest negative/active states first
    if "did-not-finish" in shelves:
        return "did_not_finish"

    if "currently-reading" in shelves:
        return "currently_reading"

    # Explicit to-read signal
    if "to-read" in shelves:
        return "to_read"

    # Evidence that the user completed/interacted with the book
    date_read = row.get("date_read")

    if pd.notna(date_read) and str(date_read).strip():
        return "read"

    rating = row.get("user_rating")

    try:
        if pd.notna(rating) and float(rating) > 0:
            return "read"
    except (ValueError, TypeError):
        pass

    return "unknown"


def analyze_file(file_path, user_name):
    df = pd.read_csv(file_path)

    df["derived_status"] = df.apply(derive_status, axis=1)

    print("\n" + "=" * 60)
    print(user_name)
    print("=" * 60)

    print(f"Total books: {len(df)}")
    print("\nStatus counts:")
    print(df["derived_status"].value_counts().sort_index())

    unknown = df[df["derived_status"] == "unknown"]

    print(f"\nUNKNOWN: {len(unknown)}")

    if len(unknown) > 0:
        print("\nUnknown books:")
        print(
            unknown[
                [
                    "title",
                    "author",
                    "date_read",
                    "date_added",
                    "user_rating",
                    "shelves",
                ]
            ].to_string(index=False)
        )

    return df


if __name__ == "__main__":

    files = {
        "Jessica": Path("data/raw/goodreads_books.csv"),
        "Sarah": Path("data/raw/sarah_books.csv"),
        "Shannon": Path("data/raw/shannon_books.csv"),
    }

    results = {}

    for user_name, file_path in files.items():

        if not file_path.exists():
            print(f"\nSKIPPING {user_name}")
            print(f"File not found: {file_path}")
            continue

        results[user_name] = analyze_file(file_path, user_name)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for user_name, df in results.items():
        counts = df["derived_status"].value_counts()

        print(
            f"{user_name}: "
            f"{len(df)} total | "
            f"{counts.get('read', 0)} read | "
            f"{counts.get('currently_reading', 0)} currently reading | "
            f"{counts.get('to_read', 0)} to read | "
            f"{counts.get('did_not_finish', 0)} DNF | "
            f"{counts.get('unknown', 0)} unknown"
        )