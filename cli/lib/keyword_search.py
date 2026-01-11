from .inverted_index import InvertedIndex
from .query_utils import clean
from .search_utils import DEFAULT_SEARCH_LIMIT


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
