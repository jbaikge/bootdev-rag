#!/usr/bin/env python3

import argparse
import json

from lib.constants import BM25_K1
from lib.keyword_search import search_command
from lib.idf_search import idf_command
from lib.index_builder import build_command
from lib.term_search import term_command
from lib.tfidf_search import tfidf_command
from lib.bm25idf_search import bm25idf_command
from lib.bm25tf_search import bm25_tf_command


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

    tf_parser = subparsers.add_parser(
        "tf",
        help="Term frequency",
    )
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )
    tf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    idf_parser = subparsers.add_parser(
        "idf",
        help="Inverse Document Frequency",
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Term Frequency with Inverse Document Frequency",
    )
    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    bm25idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term",
    )
    bm25idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 IDF score for",
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term",
    )
    bm25_tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )
    bm25_tf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 TF score for",
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs='?',
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )

    args = parser.parse_args()

    match args.command:
        case "bm25idf":
            bm25idf = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(args.doc_id, args.term, args.k1)
            print(
                "BM25 TF score of '%s' in document '%d': %0.2f" %
                (args.term, args.doc_id, bm25tf)
            )
        case "build":
            build_command()
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']}")
        case "tf":
            print(term_command(args.doc_id, args.term))
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                "TF-IDF score of '%s' in document '%d': %0.2f" %
                (args.term, args.doc_id, tf_idf)
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
