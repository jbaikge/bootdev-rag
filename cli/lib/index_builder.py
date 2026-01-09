from .inverted_index import InvertedIndex
from .search_utils import PROJECT_ROOT, load_movies


def build_command() -> None:
    movies = load_movies()
    index = InvertedIndex()
    index.build(movies)
    index.save(PROJECT_ROOT)

    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
