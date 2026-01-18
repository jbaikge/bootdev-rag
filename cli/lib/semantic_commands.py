from .search_utils import load_movies
from .semantic_search import SemanticSearch


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


def verify_embeddings_command():
    movies = load_movies()
    search = SemanticSearch()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        "Embeddings shape: %d vectors in %d dimensions" %
        (embeddings.shape[0], embeddings.shape[1])
    )
