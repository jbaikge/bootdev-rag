#!/usr/bin/env python3

import argparse

from lib.semantic_commands import (
    chunk_command,
    embed_command,
    embedquery_command,
    search_command,
    verify_command,
    verify_embeddings_command,
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

    embedquery_parser = subparsers.add_parser(
        "embedquery",
        help="Generate text embeddings",
    )
    embedquery_parser.add_argument(
        "query",
        type=str,
        help="Text to get embeddings for",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search for text",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    subparsers.add_parser(
        "verify",
        help="Verify semantic search model",
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Build and verify embeddings",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk text",
    )
    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text",
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum number of words per chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Maximum number of words to overlap in each chunk",
    )

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "embed_text":
            embed_command(args.text)
        case "embedquery":
            embedquery_command(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "verify":
            verify_command()
        case "verify_embeddings":
            verify_embeddings_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
