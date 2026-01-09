from .inverted_index import InvertedIndex
from .search_utils import PROJECT_ROOT


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load(PROJECT_ROOT)
    return index.get_idf(term)
