import os

from .inverted_index import BM25SearchResult, InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .hybrid_utils import (
    hybrid_score,
    normalize,
    rrf_score,
)


class RRFResult:
    def __init__(self, id: int, bm25: int, semantic: int, doc: dict):
        self.doc_id = id
        self.bm25_rank = bm25
        self.semantic_rank = semantic
        self.rrf_score = 0.0
        self.doc = doc

    def calculate_score(self, k: int) -> None:
        if self.bm25_rank > 0:
            self.rrf_score += rrf_score(self.bm25_rank, k)

        if self.semantic_rank > 0:
            self.rrf_score += rrf_score(self.semantic_rank, k)


class WeightedResult:
    def __init__(self, id: int, bm25: float, semantic: float, weighted: float, doc: dict):
        self.doc_id = id
        self.bm25_score = bm25
        self.semantic_score = semantic
        self.weighted_score = weighted
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

    def rrf_search(self, query: str, k: int, limit: int) -> list[RRFResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query,
            limit * 500,
        )

        rank_map = {}
        for i, result in enumerate(bm25_results, 1):
            id = result.movie["id"]
            rank_map[id] = RRFResult(
                id,
                i,
                0,
                self.semantic_search.document_map[id],
            )

        for i, result in enumerate(semantic_results, 1):
            id = result.doc_id
            if id in rank_map:
                rank_map[id].semantic_rank = i
            else:
                rank_map[id] = RRFResult(
                    id,
                    i,
                    0,
                    self.semantic_search.document_map[id],
                )

            rank_map[id].calculate_score(k)

        return sorted(
            rank_map.values(),
            key=lambda v: v.rrf_score,
            reverse=True,
        )[:limit]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[WeightedResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = []
        for result in bm25_results:
            bm25_scores.append(result.score)

        for i, score in enumerate(normalize(bm25_scores)):
            bm25_results[i].normal_score = score

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
            score_map[id] = WeightedResult(
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
                score_map[id] = WeightedResult(
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
