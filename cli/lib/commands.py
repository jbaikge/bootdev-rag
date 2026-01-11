from .constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex, BM25SearchResult
from .query_utils import clean
from .search_utils import load_movies


def bm25idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)


def bm25_search_command(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[BM25SearchResult]:
    index = InvertedIndex()
    index.load()
    return index.bm25_search(query, limit)
    pass


def bm25_tf_command(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)


def build_command() -> None:
    movies = load_movies()
    index = InvertedIndex()
    index.build(movies)
    index.save()


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()

    ids = []
    for term in clean(query):
        ids.extend(index.get_documents(term))
        if len(ids) >= DEFAULT_SEARCH_LIMIT:
            break

    movies = []
    for id in ids[:DEFAULT_SEARCH_LIMIT]:
        movies.append(index.docmap[id])

    return movies


def term_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term) * index.get_tf(doc_id, term)
