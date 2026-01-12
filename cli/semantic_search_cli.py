#!/usr/bin/env python3

import argparse

from lib.semantic_commands import (
    verify_command
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    subparsers.add_parser(
        "verify",
        help="Verify semantic search model",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
