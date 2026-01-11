from .inverted_index import InvertedIndex


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term)
