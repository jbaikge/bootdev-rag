#!/usr/bin/env python3

import argparse

from lib.semantic_commands import (
    embed_command,
    verify_command,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate text embeddings",
    )
    embed_parser.add_argument(
        "text",
        type=str,
        help="Text to get embeddings for",
    )

    subparsers.add_parser(
        "verify",
        help="Verify semantic search model",
    )

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_command(args.text)
        case "verify":
            verify_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
