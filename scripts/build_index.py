"""
Script to build or rebuild the local vector store index from processed knowledge chunks.
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_KB
from src.retrieval.retriever import AgronomyRetriever

def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index from processed agronomy chunks")
    parser.add_argument("--input", type=str, default=str(PROCESSED_KB), help="Path to processed knowledge_base.jsonl")
    parser.add_argument("--reset", action="store_true", help="Clear existing collection before indexing")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Processed knowledge base not found at: {input_path}")
        print("Please run ingestion first: python -m src.ingestion.ingest")
        sys.exit(1)

    print(f"Building vector index from: {input_path}")
    retriever = AgronomyRetriever()
    count = retriever.index_knowledge_base(input_path, reset=args.reset)
    print(f"Successfully indexed {count} agronomic chunks into ChromaDB.")

if __name__ == "__main__":
    main()
