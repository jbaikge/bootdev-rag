import numpy as np
import os.path

from sentence_transformers import SentenceTransformer

from .query_utils import cosine_similarity
from .search_utils import CACHE_PATH


class SemanticResult:
    def __init__(self, score: float, title: str, desc: str):
        self.score = score
        self.title = title
        self.description = desc
        self.normal_score = None


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = f"{CACHE_PATH}/movie_embeddings.npy"

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("empty text")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> list:
        self.documents = documents

        doc_strs = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strs.append(f'{doc["title"]}: {doc["description"]}')

        self.embeddings = self.model.encode(doc_strs, show_progress_bar=True)

        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> list:
        self.documents = documents

        doc_strs = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strs.append(f'{doc["title"]}: {doc["description"]}')

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[SemanticResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        q_embedding = self.generate_embedding(query)
        scores = []
        for i, embedding in enumerate(self.embeddings):
            scores.append((
                cosine_similarity(q_embedding, embedding),
                self.documents[i],
            ))

        scores.sort(key=lambda s: s[0], reverse=True)

        results = []
        for score in scores[:limit]:
            results.append(SemanticResult(
                score[0],
                score[1]["title"],
                score[1]["description"],
            ))

        return results
