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
