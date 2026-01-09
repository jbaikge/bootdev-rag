import os
import os.path
import pickle

from .query_utils import clean


class InvertedIndex:
    def __init__(self):

        # Dictionary mapping tokens to sets of document IDs
        self.index = {}

        # Dictionary mapping document IDs to their full document objects
        self.docmap = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes text, then adds each token to the index with the document ID
        """

        for token in set(clean(text)):
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
