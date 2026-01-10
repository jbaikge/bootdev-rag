from .constants import BM25_K1
from .inverted_index import InvertedIndex
from .search_utils import PROJECT_ROOT


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    index = InvertedIndex()
    index.load(PROJECT_ROOT)
    return index.get_bm25_tf(doc_id, term, k1)
