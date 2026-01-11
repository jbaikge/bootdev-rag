from .inverted_index import InvertedIndex


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term) * index.get_tf(doc_id, term)
