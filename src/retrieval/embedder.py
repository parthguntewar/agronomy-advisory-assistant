"""
Embedding model wrapper using sentence-transformers.
Provides vector generation for query texts and agronomy knowledge chunks.
"""
import logging
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer
from src.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class AgronomyEmbedder:
    """Wrapper around SentenceTransformer with caching and normalization."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self.embedding_dim = self._model.get_embedding_dimension()
        else:
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {self.model_name} (Dimension: {self.embedding_dim})")

    def embed_text(self, text: str) -> List[float]:
        """Generates normalized embedding for a single text string."""
        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates normalized embeddings for a list of texts."""
        if not texts:
            return []
        embs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return embs.tolist()


_GLOBAL_EMBEDDER = None

def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> AgronomyEmbedder:
    """Singleton getter for the embedder instance."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None or _GLOBAL_EMBEDDER.model_name != model_name:
        _GLOBAL_EMBEDDER = AgronomyEmbedder(model_name)
    return _GLOBAL_EMBEDDER
