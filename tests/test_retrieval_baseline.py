"""
Baseline retrieval test suite covering 5 diagnostic scenarios:
1. Normal in-domain crop-management question.
2. Completely non-agricultural out-of-domain question.
3. Unindexed agricultural crop with no evidence in KB.
4. Outdated / time-sensitive domain question (weather / government scheme).
5. Ambiguous crop-disease question without crop name or specific symptoms.
"""
import sys
import unittest
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_KB
from src.retrieval.retriever import AgronomyRetriever


class TestRetrievalBaseline(unittest.TestCase):
    """Test suite evaluating baseline semantic retrieval performance and boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = AgronomyRetriever()
        # Verify or build index
        if cls.retriever.store.count() == 0:
            print("\n[SETUP] Index is empty. Building index from processed knowledge base...")
            cls.retriever.index_knowledge_base(PROCESSED_KB, reset=True)
        print(f"\n[SETUP] Active index contains {cls.retriever.store.count()} chunks.")

    def run_query(self, query: str) -> Dict[str, Any]:
        """Helper to run retrieval and return top match details."""
        results = self.retriever.retrieve(query, top_k=3)
        if not results:
            return {
                "query": query,
                "has_match": False,
                "top_score": 0.0,
                "top_crop": None,
                "top_chunk_id": None,
                "top_advisory": None,
                "results": []
            }
        top = results[0]
        return {
            "query": query,
            "has_match": True,
            "top_score": top.similarity_score,
            "top_crop": top.crop_clean,
            "top_chunk_id": top.chunk_id,
            "top_advisory": top.expert_advisory,
            "qc_flags": top.qc_flags,
            "results": results
        }

    def test_01_normal_crop_management(self):
        """Scenario 1: Normal in-domain agronomic query with crop and problem."""
        query = "How to control insects in Satawar crop?"
        res = self.run_query(query)
        print(f"\n--- Scenario 1: Normal Crop Management ---")
        print(f"Query: {query}")
        print(f"Top Score: {res['top_score']:.4f} | Crop: {res['top_crop']} | ID: {res['top_chunk_id']}")
        print(f"Advisory snippet: {res['top_advisory'][:120]}...")
        
        self.assertTrue(res["has_match"])
        self.assertGreaterEqual(res["top_score"], 0.45, "In-domain match should have strong semantic similarity")
        self.assertIn("Satawar", res["top_crop"], "Retrieved crop must match query crop")

    def test_02_unrelated_to_agriculture(self):
        """Scenario 2: Completely non-agricultural technical query."""
        query = "How do I configure a React frontend router in Node.js?"
        res = self.run_query(query)
        print(f"\n--- Scenario 2: Unrelated Non-Agricultural Query ---")
        print(f"Query: {query}")
        print(f"Top Score: {res['top_score']:.4f} | Irrelevant Match: {res['top_crop']}")
        
        # Semantic similarity for completely unrelated domain should be very low
        self.assertLess(res["top_score"], 0.25, "Unrelated query must not achieve high similarity score")

    def test_03_no_relevant_evidence_in_kb(self):
        """Scenario 3: Agricultural question about an unindexed crop."""
        query = "What is the recommended grafting method for Brazilian Dragon Fruit in winter?"
        res = self.run_query(query)
        print(f"\n--- Scenario 3: Agricultural Query with No KB Evidence ---")
        print(f"Query: {query}")
        print(f"Top Score: {res['top_score']:.4f} | Forced Match: {res['top_crop']}")
        
        # The crop 'Dragon Fruit' does not exist in our 26-chunk KB
        # Vector similarity returns the closest vector, but with low relevance
        self.assertNotEqual(res["top_crop"], "Dragon Fruit")
        self.assertLess(res["top_score"], 0.45, "Unindexed crop query should not exceed confident threshold")

    def test_04_outdated_or_time_sensitive(self):
        """Scenario 4: Outdated / time-sensitive domain question (weather / scheme)."""
        query = "Will it rain tomorrow in Mehsana district Gujarat and when is the 10th PM Kisan installment date?"
        res = self.run_query(query)
        print(f"\n--- Scenario 4: Outdated / Time-Sensitive Query ---")
        print(f"Query: {query}")
        print(f"Top Score: {res['top_score']:.4f} | Match: {res['top_crop']}")
        print(f"Top advisory snippet: {res['top_advisory'][:120]}...")
        
        # Because weather and government schemes were filtered out during ingestion,
        # the retriever cannot retrieve stale weather/scheme advice, but forced nearest neighbor occurs.
        # This confirms that retrieval alone cannot refuse; an explicit guardrail is required!
        print("Finding: Vector search returns nearest neighbor despite query being out-of-scope for static KB.")

    def test_05_ambiguous_crop_disease(self):
        """Scenario 5: Ambiguous crop-disease question without crop name or symptoms."""
        query = "My leaves have yellow spots and are curling. What spray should I use?"
        res = self.run_query(query)
        print(f"\n--- Scenario 5: Ambiguous Query Without Crop Name ---")
        print(f"Query: {query}")
        print(f"Top Score: {res['top_score']:.4f} | Arbitrary Top Match: {res['top_crop']}")
        print(f"Prescribed chemical snippet: {res['top_advisory'][:120]}...")
        
        # Proves that retrieval alone is hazardous: it arbitrarily retrieves a chemical for Okra/Tomato
        # Demonstrates why Phase 3 must have a Symptom/Crop Clarification Guardrail.
        self.assertTrue(res["has_match"])


if __name__ == "__main__":
    unittest.main()
