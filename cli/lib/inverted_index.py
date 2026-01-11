import collections
import math
import os
import os.path
import pickle

from .constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from .query_utils import clean
from .search_utils import CACHE_PATH


class BM25SearchResult:
    def __init__(self, movie: dict, score: float):
        self.movie = movie
        self.score = score


class InvertedIndex:
    def __init__(self):

        # Dictionary mapping tokens to sets of document IDs
        self.index = {}

        # Dictionary mapping document IDs to their full document objects
        self.docmap = {}

        # Dictionary mapping document IDs to their document lengths
        self.doc_lengths = {}

        # Dictionary mapping document IDs to Counter objects
        self.term_frequencies = {}

        self.index_file = "index.pkl"
        self.docmap_file = "docmap.pkl"
        self.term_freq_file = "term_frequencies.pkl"
        self.doc_lengths_file = "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes text, then adds each token to the index with the document ID
        """

        tokens = clean(text)
        self.term_frequencies[doc_id] = collections.Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

        for token in set(tokens):
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def __get_avg_doc_length(self) -> float:
        """
        Calculate the average document length of all documents
        """

        num_docs = len(self.doc_lengths)
        if num_docs == 0:
            return 0.0

        total = 0
        for id in self.doc_lengths:
            total += self.doc_lengths[id]

        return total / num_docs

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[BM25SearchResult]:
        tokens = clean(query)
        scores = {}

        for token in tokens:
            if token not in self.index:
                continue

            for id in self.index[token]:
                if id not in scores:
                    scores[id] = 0
                scores[id] += self.bm25(id, token)

        ordered = dict(sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ))
        results = []

        for id in ordered:
            if len(results) >= limit:
                break

            results.append(BM25SearchResult(self.docmap[id], ordered[id]))

        return results

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

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        """
        Applies a saturation formula to the existing Term Frequency (TF)
        """

        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

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

    def save(self, dir: str = CACHE_PATH) -> None:
        """
        Saves the index and the docmap to a cache directory inside of root_dir
        in pickle format.
        """

        os.makedirs(dir, exist_ok=True)

        with open(os.path.join(dir, self.index_file), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(dir, self.docmap_file), "wb") as f:
            pickle.dump(self.docmap, f)

        with open(os.path.join(dir, self.term_freq_file), "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(os.path.join(dir, self.doc_lengths_file), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self, dir: str = CACHE_PATH) -> None:
        if not os.path.exists(dir):
            raise RuntimeError(f"cache dir does not exist: {dir}")

        with open(os.path.join(dir, self.index_file), "rb") as f:
            self.index = pickle.load(f)

        with open(os.path.join(dir, self.docmap_file), "rb") as f:
            self.docmap = pickle.load(f)

        with open(os.path.join(dir, self.term_freq_file), "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(os.path.join(dir, self.doc_lengths_file), "rb") as f:
            self.doc_lengths = pickle.load(f)
