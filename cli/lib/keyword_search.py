import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


translation_table = str.maketrans("", "", string.punctuation)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    query = clean(query)
    movies = load_movies()
    results = []
    for movie in movies:
        if query in clean(movie["title"]):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def lower(s: str) -> str:
    return s.lower()


def depunct(s: str) -> str:
    return s.translate(translation_table)


def clean(s: str) -> str:
    return depunct(lower(s))
