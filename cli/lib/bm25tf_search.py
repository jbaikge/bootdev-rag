from .constants import BM25_B, BM25_K1
from .inverted_index import InvertedIndex


def bm25_tf_command(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)
