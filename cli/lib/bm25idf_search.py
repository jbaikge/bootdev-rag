from .inverted_index import InvertedIndex


def bm25idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)
