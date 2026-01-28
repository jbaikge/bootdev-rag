import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .hybrid_utils import normalize
from .search_utils import load_movies


model = "gemini-2.5-flash"


def normalize_command(scores: list[float]) -> None:
    for score in normalize(scores):
        print(f"* {score:.4f}")


def rrf_search_command(query: str, k: int, limit: int, enhance: str) -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    contents = None
    if enhance == "rewrite":
        contents = f"""
        Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:"""

    if enhance == "spell":
        contents = f"""
        Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.

        Corrected:"""

    if contents is not None:
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
        print(
            f"Enhanced query ({enhance}):",
            f"'{query}' -> '{response.text}'\n",
        )
        query = response.text

    movies = load_movies()
    search = HybridSearch(movies)
    for i, result in enumerate(search.rrf_search(query, k, limit), 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   RRF Score: {result.rrf_score:.3f}")
        print(
            f"   BM25 Rank: {result.bm25_rank},",
            f"Semantic Rank: {result.semantic_rank}"
        )
        print(f"   {result.doc['description'][:50]}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    movies = load_movies()
    search = HybridSearch(movies)
    for i, result in enumerate(search.weighted_search(query, alpha, limit), 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   Hybrid Score: {result.weighted_score:.4f}")
        print(
            f"   BM25 Score: {result.bm25_score:.4f},",
            f"Semantic: {result.semantic_score:.4f}"
        )
        print(f"   {result.doc['description'][:50]}")
