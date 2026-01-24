import re

from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies
from .semantic_search import SemanticSearch


def chunk_command(text: str, chunk_size: int, overlap: int):
    words = text.split(" ")
    chunks = []
    while len(words) > chunk_size:
        chunks.append(words[:chunk_size])
        words = words[chunk_size - overlap:]

    if len(words) > 0:
        chunks.append(words)

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {' '.join(chunk)}")


def verify_command():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_chunks_command():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def embed_command(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embedquery_command(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Query: {text}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search_command(query: str, limit: int):
    movies = load_movies()
    search = SemanticSearch()
    search.load_or_create_embeddings(movies)
    for i, result in enumerate(search.search(query, limit), 1):
        print(f"{i}. {result['title']}")
        print(f"   {result["description"]}")


def search_chunked_command(query: str, limit: int):
    movies = load_movies()
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(movies)
    for i, result in enumerate(search.search_chunks(query, limit), 1):
        print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"    {result["document"]}")


def semantic_chunk_command(text: str, chunk_size: int, overlap: int):
    search = ChunkedSemanticSearch()
    chunks = search.semantic_chunk(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {' '.join(chunk)}")


def verify_embeddings_command():
    movies = load_movies()
    search = SemanticSearch()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        "Embeddings shape: %d vectors in %d dimensions" %
        (embeddings.shape[0], embeddings.shape[1])
    )
