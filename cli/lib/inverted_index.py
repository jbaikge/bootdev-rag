import collections
import math
import os
import os.path
import pickle

from .constants import BM25_K1
from .query_utils import clean


class InvertedIndex:
    def __init__(self):

        # Dictionary mapping tokens to sets of document IDs
        self.index = {}

        # Dictionary mapping document IDs to their full document objects
        self.docmap = {}

        # Dictionary mapping document IDs to Counter objects
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes text, then adds each token to the index with the document ID
        """

        tokens = clean(text)
        if doc_id == 1:
            print(tokens)
            print(collections.Counter(tokens))
        self.term_frequencies[doc_id] = collections.Counter(tokens)

        for token in set(tokens):
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """
        Retrieves the set of document IDs for a given token and returns them as
        a list of sorted IDs.

        Assume the input term is a single token
        """

        lower_term = term.lower()
        if lower_term not in self.index:
            return []

        return self.index[lower_term]

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate the BM25 IDF score from the supplied term
        """

        tokens = clean(term)

        if len(tokens) == 0:
            raise RuntimeError(f"empty term after cleaning: {term}")

        if len(tokens) > 1:
            raise RuntimeError(f"too many tokens in term: {term}")

        docs = 0
        if tokens[0] in self.index:
            docs = len(self.index[tokens[0]])

        total = len(self.docmap)

        return math.log((total - docs + 0.5) / (docs + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        """
        Applies a saturation formula to the existing Term Frequency (TF)
        """

        tf = self.get_tf(doc_id, term)
        print(doc_id, term, k1, tf)
        return (tf * (k1 + 1)) / (tf + k1)

    def get_idf(self, term: str) -> float:
        """
        Inverse document frequency measures how many documents in the dataset
        contain a term, then converts it to a number where more rare terms
        rank higher.
        """

        tokens = clean(term)

        if len(tokens) == 0:
            raise RuntimeError(f"empty term after cleaning: {term}")

        if len(tokens) > 1:
            raise RuntimeError(f"too many tokens in term: {term}")

        matches = 0
        if tokens[0] in self.index:
            matches = len(self.index[tokens[0]])

        return math.log((len(self.docmap) + 1) / (matches + 1))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the times the token appears in the document with the given ID.

        If the term doesn't exist in that document, return 0.

        Term is tokenized, but also assumed to only be one token
        """

        tokens = clean(term)

        if len(tokens) == 0:
            raise RuntimeError(f"empty term after cleaning: {term}")

        if len(tokens) > 1:
            raise RuntimeError(f"too many tokens in term: {term}")

        if doc_id not in self.term_frequencies:
            raise RuntimeError(f"unknown document ID: {doc_id}")

        return self.term_frequencies[doc_id][tokens[0]]

    def build(self, movies: list[dict]) -> None:
        """
        Iterates over all the movies and adds them to both the index and the
        doc map.

        When adding the movie data to the index, concatenates the title and
        description to use as the input text.
        """

        for m in movies:
            text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], text)
            self.docmap[m["id"]] = m

    def save(self, root_dir: str) -> None:
        """
        Saves the index and the docmap to a cache directory inside of root_dir
        in pickle format.
        """

        cache_dir = os.path.join(root_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        index_file = os.path.join(cache_dir, "index.pkl")
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)

        docmap_file = os.path.join(cache_dir, "docmap.pkl")
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)

        term_freq_file = os.path.join(cache_dir, "term_frequencies.pkl")
        with open(term_freq_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self, root_dir: str) -> None:
        cache_dir = os.path.join(root_dir, "cache")
        if not os.path.exists(cache_dir):
            raise RuntimeError(f"cache dir does not exist: {cache_dir}")

        index_file = os.path.join(cache_dir, "index.pkl")
        if not os.path.exists(index_file):
            raise RuntimeError(f"index file does not exist: {index_file}")

        with open(index_file, "rb") as f:
            self.index = pickle.load(f)

        docmap_file = os.path.join(cache_dir, "docmap.pkl")
        if not os.path.exists(docmap_file):
            raise RuntimeError(f"docmap file does not exist: {docmap_file}")

        with open(docmap_file, "rb") as f:
            self.docmap = pickle.load(f)

        term_freq_file = os.path.join(cache_dir, "term_frequencies.pkl")
        if not os.path.exists(term_freq_file):
            raise RuntimeError(
                f"term frequencies file does not exist: {term_freq_file}"
            )

        with open(term_freq_file, "rb") as f:
            self.term_frequencies = pickle.load(f)
