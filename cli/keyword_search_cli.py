#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command
from lib.index_builder import build_command


def title_search(query: str) -> list:
    with open("data/movies.json") as f:
        data = json.load(f)
    found = []
    for movie in data["movies"]:
        if query in movie["title"]:
            found.append(movie)

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using BM25",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    subparsers.add_parser(
        "build",
        help="Build search index",
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
