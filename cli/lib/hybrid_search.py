import os

from .inverted_index import BM25SearchResult, InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .hybrid_utils import (
    hybrid_score,
    normalize,
)


class HybridResult:
    def __init__(self, id: int, bm25: float, semantic: float, hybrid: float, doc: dict):
        self.doc_id = id
        self.bm25_score = bm25
        self.semantic_score = semantic
        self.hybrid_score = hybrid
        self.doc = doc


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_file):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[BM25SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = []
        for result in bm25_results:
            bm25_scores.append(result.score)

        for i, score in enumerate(normalize(bm25_scores)):
            bm25_results[i].normal_score = score

        print(f"Embeddings? {self.semantic_search.embeddings is None}")
        semantic_results = self.semantic_search.search_chunks(
            query,
            limit * 500,
        )
        semantic_scores = []
        for result in semantic_results:
            semantic_scores.append(result.score)

        for i, score in enumerate(normalize(semantic_scores)):
            semantic_results[i].normal_score = score

        score_map = {}
        for result in bm25_results:
            id = result.movie["id"]
            score_map[id] = HybridResult(
                id,
                result.normal_score,
                0.0,
                0.0,
                self.semantic_search.document_map[id],
            )

        for result in semantic_results:
            id = result.doc_id
            if id in score_map:
                score_map[id].semantic_score = result.normal_score
            else:
                score_map[id] = HybridResult(
                    id,
                    0.0,
                    result.normal_score,
                    0.0,
                    self.semantic_search.document_map[id],
                )

            hs = score_map[id]
            hs.hybrid_score = hybrid_score(
                hs.bm25_score,
                hs.semantic_score,
                alpha,
            )

        return sorted(
            score_map.values(),
            key=lambda v: v.hybrid_score,
            reverse=True,
        )[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
