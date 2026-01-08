import string

from .search_utils import load_stopwords


translation_table = str.maketrans("", "", string.punctuation)
stop_words = load_stopwords()


def lower(s: str) -> str:
    return s.lower()


def depunct(s: str) -> str:
    return s.translate(translation_table)


def tokenize(s: str) -> list[str]:
    return s.split(" ")


def unstop(tokens: list[str]) -> list[str]:
    return list(set(tokens).difference(stop_words))


def clean(s: str) -> list[str]:
    return unstop(tokenize(depunct(lower(s))))


def match(query: list[str], against: list[str]) -> bool:
    for a in against:
        for q in query:
            if q in a:
                return True
    return False
