import re
import string

from nltk.stem import PorterStemmer

from .search_utils import load_stopwords


punctuation_table = str.maketrans("", "", string.punctuation + "\u2019")
whitespace_table = str.maketrans(
    string.whitespace,
    " " * len(string.whitespace),
)
stop_words = load_stopwords()
stemmer = PorterStemmer()


def lower(s: str) -> str:
    return s.lower()


def whitespace(s: str) -> str:
    return re.sub(r"  +", " ", s.translate(whitespace_table))


def depunct(s: str) -> str:
    return s.translate(punctuation_table)


def tokenize(s: str) -> list[str]:
    return s.split(" ")


def unstop(tokens: list[str]) -> list[str]:
    cleaned = []
    for token in tokens:
        if token in stop_words:
            continue
        cleaned.append(token)
    return cleaned


def stem(tokens: list[str]) -> list[str]:
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    return tokens


def clean(s: str) -> list[str]:
    return stem(unstop(tokenize(whitespace(depunct(lower(s))))))


def match(query: list[str], against: list[str]) -> bool:
    for a in against:
        for q in query:
            if q in a:
                return True
    return False
