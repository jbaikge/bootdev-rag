import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .hybrid_search import HybridSearch
from .hybrid_utils import normalize
from .search_utils import load_movies


model = "gemini-2.5-flash"


def normalize_command(scores: list[float]) -> None:
    for score in normalize(scores):
        print(f"* {score:.4f}")


def rrf_search_command(
    query: str,
    k: int,
    limit: int,
    enhance: str,
    rerank_method: str,
) -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    contents = None
    if enhance == "expand":
        contents = f"""
        Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """

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

    if rerank_method in ["batch", "individual"]:
        original_limit = limit
        limit = limit * 5

    movies = load_movies()
    search = HybridSearch(movies)
    results = search.rrf_search(query, k, limit)

    if rerank_method == "individual":
        for result in results:
            rerank_contents = f"""
            Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {result.doc.get("title", "")}
            Description: {result.doc.get("description", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""

            response = client.models.generate_content(
                model=model,
                contents=rerank_contents,
            )

            if response.text is None:
                # Quietly ignore
                # print("Result:")
                # print(json.dumps(result.doc))
                # print("Response:")
                # print(response)
                continue

            result.rerank_score = float(response.text)
            time.sleep(3.5)

        results = sorted(
            results,
            key=lambda v: v.rerank_score,
            reverse=True,
        )[:original_limit]

    if rerank_method == "batch":
        doc_list = []
        for result in results:
            doc_list.append(result.doc)

        doc_list_str = json.dumps(doc_list)
        rerank_contents = f"""
        Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else, do not wrap in markdown. For example:

        [75, 12, 34, 2, 1]
        """

        response = client.models.generate_content(
            model=model,
            contents=rerank_contents,
        )

        if response.text is None:
            print(f"Failed to get result: {response}")
            return

        ids = json.loads(response.text)
        for result in results:
            try:
                result.rerank_score = ids.index(result.doc_id) + 1
            except ValueError:
                result.rerank_score = 0

        results = sorted(
            results,
            key=lambda v: v.rerank_score,
            reverse=False,
        )[:original_limit]

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.doc['title']}")
        print(f"   Rerank Score: {result.rerank_score:.3f}/10")
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
