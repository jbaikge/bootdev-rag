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
        intersect = relevant_set.intersection(found_set)
        precision = len(intersect) / len(found)
        recall = len(intersect) / len(relevant)
        f1 = 2 * (precision * recall) / (precision + recall)

        print("")
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Intersection {len(intersect)}: {', '.join(intersect)}")
        print(f"  - Retrieved {len(found)}: {', '.join(found)}")
        print(f"  - Relevant {len(relevant)}: {', '.join(relevant)}")
