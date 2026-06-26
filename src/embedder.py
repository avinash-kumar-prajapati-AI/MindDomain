from typing import List
from sentence_transformers import SentenceTransformer
from src.config import config

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.embedding.model, device="cpu")
    return _model

def embed(text: str, is_query: bool = False) -> List[float]:
    """
    Generates a 384-dimensional dense vector representation of text.
    For BGE models, adds search queries and document chunk prefixes to align retrieval accuracy.
    """
    model = get_model()
    # Format query or document matching BGE instructions
    if is_query:
        formatted_text = f"Represent this sentence for searching relevant passages: {text}"
    else:
        formatted_text = f"Represent this passage for retrieval: {text}"
        
    embedding = model.encode(formatted_text, convert_to_numpy=True)
    return embedding.tolist()
