def hybrid_score(
        bm25_score: float,
        semantic_score: float,
        alpha: float = 0.5,
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1] * len(scores)

    normalized = []
    for score in scores:
        normalized.append((score - min_score) / (max_score - min_score))

    return normalized
