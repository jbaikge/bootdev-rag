from .inverted_index import InvertedIndex
from .search_utils import PROJECT_ROOT


def term_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load(PROJECT_ROOT)
    return index.get_tf(doc_id, term)
