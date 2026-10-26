"""
CLI utility to test semantic retrieval for a farmer question.
Displays similarity score, advisory text, metadata, quality flags, and safety notice.
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import AgronomyRetriever

def main():
    parser = argparse.ArgumentParser(description="Query the Agronomy Advisory RAG vector store")
    parser.add_argument("query", type=str, nargs="?", help="Farmer question in English")
    parser.add_argument("--crop", type=str, default=None, help="Optional crop filter (e.g. 'Paddy (Dhan)')")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Enter a farmer question (or press Ctrl+C to exit):")
        try:
            query = input("> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    if not query:
        print("Error: Empty query provided.")
        sys.exit(1)

    retriever = AgronomyRetriever()
    results = retriever.retrieve(query, top_k=args.top_k, crop_filter=args.crop)
    output = retriever.format_results_for_display(query, results)
    print(output)

if __name__ == "__main__":
    main()
