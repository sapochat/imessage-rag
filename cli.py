"""Compatibility wrapper for running the CLI as `python3 cli.py`."""

from imessage_rag.cli import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
