import argparse

from lib.hybrid_commands import (
    normalize_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores to 0.0 - 1.0",
    )
    normalize_parser.add_argument(
        "scores",
        nargs="+",
        type=float,
        help="Scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Weighted search",
    )
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value 0.0 - 1.0",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
