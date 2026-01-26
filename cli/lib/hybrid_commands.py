from .hybrid_search import HybridSearch
from .hybrid_utils import normalize
from .search_utils import load_movies


def normalize_command(scores: list[float]) -> None:
    for score in normalize(scores):
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    movies = load_movies()
    search = HybridSearch(movies)
    for i, result in enumerate(search.weighted_search(query, alpha, limit), 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   Hybrid Score: {result.hybrid_score:.4f}")
        print(
            f"   BM25 Score: {result.bm25_score:.4f},",
            f"Semantic: {result.semantic_score:.4f}"
        )
        print(f"   {result.doc['description'][:50]}")
