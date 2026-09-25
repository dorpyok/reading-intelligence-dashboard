from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"

READER_CONFIG = {
    "you": {
        "env_var": "GOODREADS_USER_ID",
        "output": RAW_DIR / "goodreads_books.csv",
        "enriched_output": RAW_DIR / "you_books_enriched.csv",
    },
    "sarah": {
        "env_var": "GOODREADS_USER_ID_SARAH",
        "output": RAW_DIR / "sarah_books.csv",
        "enriched_output": RAW_DIR / "sarah_books_enriched.csv",
    },
    "shannon": {
        "env_var": "GOODREADS_USER_ID_SHANNON",
        "output": RAW_DIR / "shannon_books.csv",
        "enriched_output": RAW_DIR / "shannon_books_enriched.csv",
    },
}


def load_script_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def ingest_reader(
    reader: str,
    user_id: str,
    goodreads_module,
) -> Path:
    config = READER_CONFIG[reader]
    output_file = config["output"]

    print()
    print("=" * 70)
    print(f"GOODREADS INGESTION — {reader.upper()}")
    print("=" * 70)

    books = goodreads_module.fetch_all_books(
        user_id=user_id,
    )

    dataframe = pd.DataFrame(books)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(dataframe):,} books to {output_file}"
    )

    return output_file


def validate_outputs(
    readers: list[str],
) -> None:
    print()
    print("=" * 70)
    print("PIPELINE VALIDATION")
    print("=" * 70)

    for reader in readers:
        config = READER_CONFIG[reader]

        raw_file = config["output"]
        enriched_file = config["enriched_output"]

        raw_count = (
            len(pd.read_csv(raw_file))
            if raw_file.exists()
            else 0
        )

        enriched_count = (
            len(pd.read_csv(enriched_file))
            if enriched_file.exists()
            else 0
        )

        print(
            f"{reader.title():<10} "
            f"raw={raw_count:,} "
            f"enriched={enriched_count:,}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Goodreads ingestion and Open Library enrichment "
            "for multiple readers."
        )
    )

    parser.add_argument(
        "--readers",
        nargs="+",
        choices=READER_CONFIG.keys(),
        default=["you", "sarah", "shannon"],
        help="Readers to process.",
    )

    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help=(
            "Use existing raw Goodreads CSVs and only run enrichment."
        ),
    )

    parser.add_argument(
        "--force-enrichment",
        action="store_true",
        help=(
            "Re-enrich every book instead of resuming completed matches."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print enrichment progress every N books.",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="Delay between uncached Open Library requests.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_arguments()

    readers = list(dict.fromkeys(args.readers))

    scripts_dir = PROJECT_ROOT / "scripts"

    goodreads_path = PROJECT_ROOT / "src" / "ingest" / "goodreads.py"
    enrich_path = scripts_dir / "enrich_reader.py"

    if not goodreads_path.exists():
        raise FileNotFoundError(
            f"Goodreads script not found: {goodreads_path}"
        )

    if not enrich_path.exists():
        raise FileNotFoundError(
            f"Enrichment script not found: {enrich_path}"
        )

    goodreads = load_script_module(
        "reading_dashboard_goodreads",
        goodreads_path,
    )

    enrich_reader_module = load_script_module(
        "reading_dashboard_enrich_reader",
        enrich_path,
    )

    print()
    print("=" * 70)
    print("READING INTELLIGENCE — FULL READER PIPELINE")
    print("=" * 70)
    print(
        "Readers: "
        + ", ".join(reader.title() for reader in readers)
    )

    for reader in readers:
        config = READER_CONFIG[reader]

        if args.skip_ingest:
            if not config["output"].exists():
                raise FileNotFoundError(
                    f"Raw file not found for {reader}: "
                    f"{config['output']}"
                )

            print()
            print(
                f"Skipping Goodreads ingestion for {reader}; "
                f"using {config['output']}"
            )

            continue

        user_id = os.getenv(
            config["env_var"]
        )

        if not user_id:
            raise ValueError(
                f"{config['env_var']} is not set. "
                "Add the Goodreads user ID to .env or use "
                "--skip-ingest for an existing raw CSV."
            )

        ingest_reader(
            reader=reader,
            user_id=user_id,
            goodreads_module=goodreads,
        )

    for reader in readers:
        enrich_reader_module.enrich_reader(
            reader=reader,
            force=args.force_enrichment,
            progress_every=max(
                1,
                args.progress_every,
            ),
            request_delay=max(
                0.0,
                args.request_delay,
            ),
        )

    validate_outputs(readers)

    print()
    print("=" * 70)
    print("FULL PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
