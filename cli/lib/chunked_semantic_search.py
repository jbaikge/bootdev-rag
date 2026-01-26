import json
import numpy as np
import os.path
import re

from .constants import SCORE_PRECISION
from .query_utils import cosine_similarity
from .search_utils import CACHE_PATH
from .semantic_search import SemanticSearch


class ChunkedResult:
    def __init__(self, id: int, title: str, desc: str, score: float, metadata: dict):
        self.doc_id = id
        self.title = title
        self.description = desc
        self.score = score
        self.metadata = metadata


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(
            CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(
            CACHE_PATH, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        self.chunk_metadata = []
        for doc_idx, doc in enumerate(documents):
            if doc["description"] == "":
                continue
            chunks = self.semantic_chunk(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                self.chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
        )

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({
                "chunks": self.chunk_metadata,
                "total_chunks": len(all_chunks),
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)

            with open(self.chunk_metadata_path) as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def semantic_chunk(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        text = text.strip()
        if text == "":
            return []

        re_sentence = r"(?<=[.!?])\s+"
        text_splits = re.split(re_sentence, text)

        # Remove leading and trailing whitespace from sentences
        sentences = []
        for s in text_splits:
            stripped = s.strip()
            if stripped == "":
                continue
            sentences.append(s)

        chunks = []
        while len(sentences) > chunk_size:
            chunks.append(sentences[:chunk_size])
            sentences = sentences[chunk_size - overlap:]

        if len(sentences) > 0:
            chunks.append(sentences)

        return chunks

    def search_chunks(self, query: str, limit: int = 10) -> list[ChunkedResult]:
        if self.chunk_embeddings is None:
            raise ValueError(
                "not initialized, did you call load_or_create_chunk_embeddings"
            )

        query_embedding = self.generate_embedding(query)
        scores = []
        for i, embedding in enumerate(self.chunk_embeddings):
            scores.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": cosine_similarity(query_embedding, embedding),
            })

        movie_scores = {}
        for score in scores:
            movie_idx = score["movie_idx"]
            if movie_idx not in movie_scores:
                movie_scores[movie_idx] = score
                continue

            if movie_scores[movie_idx]["score"] < score["score"]:
                movie_scores[movie_idx] = score

        sorted_scores = sorted(
            movie_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []
        for score in sorted_scores[:limit]:
            doc = self.documents[score["movie_idx"]]
            results.append(ChunkedResult(
                doc["id"],
                doc["title"],
                doc["description"][:100],
                score["score"],
                score or {},
            ))

        return results
