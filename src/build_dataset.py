"""Deprecated dataset builder stub.

This script is intentionally disabled to prevent accidental execution of
legacy dataset-building logic.
"""


def main() -> None:
    raise SystemExit(
        "src.build_dataset is deprecated and disabled. "
        "Use: python -m src.prepare_data"
    )


if __name__ == "__main__":
    main()
