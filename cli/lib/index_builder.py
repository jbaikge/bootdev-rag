from .inverted_index import InvertedIndex
from .search_utils import load_movies


def build_command() -> None:
    movies = load_movies()
    index = InvertedIndex()
    index.build(movies)
    index.save()
