from .hybrid_search import HybridSearch
from .hybrid_utils import normalize
from .search_utils import load_movies


def normalize_command(scores: list[float]) -> None:
    for score in normalize(scores):
        print(f"* {score:.4f}")


def rrf_search_command(query: str, k: int, limit: int) -> None:
    movies = load_movies()
    search = HybridSearch(movies)
    for i, result in enumerate(search.rrf_search(query, k, limit), 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   RRF Score: {result.rrf_score:.3f}")
        print(
            f"   BM25 Rank: {result.bm25_rank},",
            f"Semantic Rank: {result.semantic_rank}"
        )
        print(f"   {result.doc['description'][:50]}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    movies = load_movies()
    search = HybridSearch(movies)
    for i, result in enumerate(search.weighted_search(query, alpha, limit), 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   Hybrid Score: {result.weighted_score:.4f}")
        print(
            f"   BM25 Score: {result.bm25_score:.4f},",
            f"Semantic: {result.semantic_score:.4f}"
        )
        print(f"   {result.doc['description'][:50]}")
