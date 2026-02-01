import argparse
import os.path

from lib.evaluation_commands import (
    default_command,
)


def main():
    parser = argparse.ArgumentParser(
        description="Search Evaluation CLI",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "golden_dataset.json",
    ))

    default_command(path, args.limit)


if __name__ == "__main__":
    main()
