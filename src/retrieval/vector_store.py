"""
Vector store implementation using ChromaDB for persistent local indexing.
Stores agronomy knowledge chunks with metadata and executes cosine similarity searches.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import chromadb
from chromadb.config import Settings

from src.config import VECTOR_STORE_DIR, SOURCE_ATTRIBUTION, AGRONOMIC_SAFETY_NOTICE
from src.ingestion.schema import KnowledgeBaseChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "agronomic_advisory_kb"

@dataclass
class RetrievalResult:
    """Represents an individual retrieved advisory match with scores and metadata."""
    chunk_id: str
    similarity_score: float
    crop: str
    crop_clean: str
    state: str
    district: str
    advisory_domain: str
    query_type: str
    original_year: Optional[int]
    farmer_query: str
    expert_advisory: str
    qc_severity: str
    qc_confidence: str
    qc_flags: List[str]
    rag_use_mode: str
    source_attribution: str
    safety_notice: str
    retrieval_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChromaAgronomyStore:
    """Persistent ChromaDB client for indexing and retrieving agronomic knowledge chunks."""

    def __init__(self, persist_dir: Path = VECTOR_STORE_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB PersistentClient at: {self.persist_dir}")
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection configured for cosine distance
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to collection '{COLLECTION_NAME}' (Existing documents: {self.count()})")

    def count(self) -> int:
        """Returns the number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Deletes all records from the collection."""
        logger.warning(f"Resetting collection '{COLLECTION_NAME}'")
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[KnowledgeBaseChunk], embeddings: List[List[float]]) -> int:
        """
        Adds knowledge base chunks and their precomputed embeddings to the index.
        
        Args:
            chunks: List of validated KnowledgeBaseChunk objects.
            embeddings: List of embedding vectors corresponding 1-to-1 with chunks.
            
        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.retrieval_text)
            
            # ChromaDB metadata values must be primitive str, int, float, or bool
            meta = {
                "source_example_id": chunk.source_example_id,
                "crop": chunk.crop,
                "crop_clean": chunk.crop_clean,
                "state": chunk.state,
                "district": chunk.district,
                "block": chunk.block,
                "advisory_domain": chunk.advisory_domain,
                "query_type": chunk.query_type,
                "rag_use_mode": chunk.rag_use_mode,
                "qc_severity": chunk.qc_severity,
                "qc_confidence": chunk.qc_confidence,
                "qc_flags": "|".join(chunk.qc_flags),
                "original_year": chunk.original_year if chunk.original_year else 0,
                "farmer_query": chunk.farmer_query[:500],
                "expert_advisory": chunk.expert_advisory[:1000],
            }
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Indexed {len(ids)} chunks into ChromaDB (Total in index: {self.count()})")
        return len(ids)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        crop_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Performs vector similarity search against the indexed chunks.
        
        Args:
            query_embedding: The vector embedding of the user's question.
            top_k: Number of nearest chunks to retrieve.
            crop_filter: Optional exact crop name filter.
            
        Returns:
            List of RetrievalResult objects ordered by descending cosine similarity.
        """
        where_clause = None
        if crop_filter:
            where_clause = {"crop_clean": crop_filter}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieval_matches: List[RetrievalResult] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return retrieval_matches

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for chunk_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # Convert cosine distance to cosine similarity: sim = 1.0 - distance
            # For normalized embeddings with hnsw:space=cosine, distance is in [0, 2]
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            
            qc_flags_str = meta.get("qc_flags", "")
            qc_flags = [f.strip() for f in qc_flags_str.split("|") if f.strip()]

            year = meta.get("original_year")
            if year == 0:
                year = None

            match = RetrievalResult(
                chunk_id=chunk_id,
                similarity_score=round(similarity_score, 4),
                crop=meta.get("crop", "Unknown"),
                crop_clean=meta.get("crop_clean", "Unknown"),
                state=meta.get("state", "Unknown"),
                district=meta.get("district", "Unknown"),
                advisory_domain=meta.get("advisory_domain", "general"),
                query_type=meta.get("query_type", "general"),
                original_year=year,
                farmer_query=meta.get("farmer_query", ""),
                expert_advisory=meta.get("expert_advisory", ""),
                qc_severity=meta.get("qc_severity", "none"),
                qc_confidence=meta.get("qc_confidence", "medium"),
                qc_flags=qc_flags,
                rag_use_mode=meta.get("rag_use_mode", "use_as_evidence"),
                source_attribution=SOURCE_ATTRIBUTION,
                safety_notice=AGRONOMIC_SAFETY_NOTICE,
                retrieval_text=doc,
            )
            retrieval_matches.append(match)

        return retrieval_matches
