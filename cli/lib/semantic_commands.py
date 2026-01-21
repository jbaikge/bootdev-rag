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


def verify_embeddings_command():
    movies = load_movies()
    search = SemanticSearch()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        "Embeddings shape: %d vectors in %d dimensions" %
        (embeddings.shape[0], embeddings.shape[1])
    )
