"""Frozen-application entry point for the ``opencad`` command line.

Separate from the GUI entry point because this one keeps its console: the whole
point of the CLI is that it prints results and returns a meaningful exit code.
"""

from __future__ import annotations

import sys


def main():
    from src.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
