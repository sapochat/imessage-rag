"""Compatibility wrapper for running the CLI as `python3 cli.py`."""

from src.cli import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
