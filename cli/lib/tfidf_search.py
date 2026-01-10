from .inverted_index import InvertedIndex
from .search_utils import PROJECT_ROOT


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load(PROJECT_ROOT)
    return index.get_idf(term) * index.get_tf(doc_id, term)
