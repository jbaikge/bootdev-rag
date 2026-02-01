import json

from .search_utils import load_movies
from .hybrid_search import HybridSearch


def default_command(dataset_path: str, limit: int) -> None:
    print(f"k={limit}")

    with open(dataset_path) as f:
        data = json.load(f)

    movies = load_movies()
    search = HybridSearch(movies)

    for test in data["test_cases"]:
        query = test["query"]
        relevant = test["relevant_docs"]

        found = []
        for result in search.rrf_search(query, 60, limit):
            found.append(result.doc["title"])

        relevant_set = set(relevant)
        found_set = set(found)
        intersection = relevant_set.intersection(found_set)
        precision = len(intersection) / len(found)

        print("")
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(found)}")
        print(f"  - Relevant: {', '.join(relevant)}")
