"""
High-level semantic retriever for the Agronomy Advisory Assistant.
Orchestrates embedding generation, vector search, and result enrichment.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.config import PROCESSED_KB, TOP_K_RETRIEVAL
from src.ingestion.schema import KnowledgeBaseChunk
from src.retrieval.embedder import get_embedder, AgronomyEmbedder
from src.retrieval.vector_store import ChromaAgronomyStore, RetrievalResult

logger = logging.getLogger(__name__)


class AgronomyRetriever:
    """End-to-end semantic retriever for agricultural advisory evidence."""

    def __init__(
        self,
        embedder: Optional[AgronomyEmbedder] = None,
        store: Optional[ChromaAgronomyStore] = None
    ):
        self.embedder = embedder or get_embedder()
        self.store = store or ChromaAgronomyStore()

    def index_knowledge_base(self, jsonl_path: Path = PROCESSED_KB, reset: bool = False) -> int:
        """
        Loads processed knowledge chunks from JSONL, embeds them, and persists to ChromaDB.
        
        Args:
            jsonl_path: Path to processed knowledge_base.jsonl.
            reset: Whether to clear existing index before adding.
            
        Returns:
            Number of indexed chunks.
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {path}")

        if reset:
            self.store.reset()

        chunks: List[KnowledgeBaseChunk] = []
        texts: List[str] = []

        logger.info(f"Reading chunks from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                chunk = KnowledgeBaseChunk.from_dict(data)
                chunks.append(chunk)
                texts.append(chunk.retrieval_text)

        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.embedder.model_name}...")
        embeddings = self.embedder.embed_batch(texts)

        logger.info("Persisting chunks and vectors into ChromaDB...")
        count = self.store.add_chunks(chunks, embeddings)
        return count

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        crop_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Executes semantic retrieval for a farmer query.
        
        Args:
            query: Natural language farmer query.
            top_k: Number of chunks to retrieve.
            crop_filter: Optional crop name filter.
            
        Returns:
            List of RetrievalResult objects.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_emb = self.embedder.embed_text(cleaned_query)
        results = self.store.search(query_emb, top_k=top_k, crop_filter=crop_filter)
        return results

    def format_results_for_display(self, query: str, results: List[RetrievalResult]) -> str:
        """Formats retrieval results into human-readable CLI text with complete metadata."""
        lines = [
            "=" * 70,
            f"FARMER QUERY: {query}",
            f"MATCHES FOUND: {len(results)}",
            "=" * 70
        ]

        if not results:
            lines.append("No matching agronomic evidence found in knowledge base.")
            lines.append("-" * 70)
            return "\n".join(lines)

        for rank, res in enumerate(results, 1):
            lines.append(f"\n[RANK {rank}] Similarity Score: {res.similarity_score:.4f}")
            lines.append(f"  Chunk ID:        {res.chunk_id}")
            lines.append(f"  Crop:            {res.crop_clean} (Raw: {res.crop})")
            lines.append(f"  Location:        {res.district}, {res.state}")
            lines.append(f"  Advisory Domain: {res.advisory_domain} | Topic: {res.query_type}")
            lines.append(f"  Record Year:     {res.original_year or 'Unspecified historical'}")
            lines.append(f"  QC Flags:        {', '.join(res.qc_flags) if res.qc_flags else 'None'}")
            lines.append(f"  QC Severity:     {res.qc_severity} | RAG Mode: {res.rag_use_mode}")
            lines.append(f"  Farmer Query:    {res.farmer_query}")
            lines.append(f"  Expert Advisory: {res.expert_advisory}")
            lines.append(f"  Attribution:     {res.source_attribution}")
            lines.append(f"  Safety Notice:   {res.safety_notice}")
            lines.append("-" * 70)

        return "\n".join(lines)
