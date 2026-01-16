from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("empty text")

        return self.model.encode([text])[0]
