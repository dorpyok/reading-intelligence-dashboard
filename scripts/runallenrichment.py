from __future__ import annotations

import subprocess
import sys


READERS = [
    "you",
    "sarah",
    "shannon",
]


def main() -> None:
    for reader in READERS:
        print()
        print("=" * 70)
        print(f"Starting enrichment for: {reader}")
        print("=" * 70)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.enrich_reader",
                "--reader",
                reader,
            ],
            check=False,
        )

        if result.returncode != 0:
            print()
            print(
                f"WARNING: {reader} enrichment exited "
                f"with code {result.returncode}"
            )
            print("Continuing to the next reader.")

    print()
    print("=" * 70)
    print("ALL READER ENRICHMENT RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()