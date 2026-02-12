# main.py
# /CLI for qdrant search experiment/
# adriana r.f. (@adrmisty)
# feb-2026

import argparse
from qdrant_client import QdrantClient
from qdrant_search.search import Ingestor
from qdrant_search.ingest import Engine
from qdrant_search.config import settings

def main():
    parser = argparse.ArgumentParser(description="Document search experiment")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- INGEST ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest logical layer JSON into Qdrant")
    p_ingest.add_argument("--file", type=str, default=settings.DATA_PATH, help="Path to input JSON file")

    # --- SEARCH ---
    p_search = subparsers.add_parser("search", help="Search for KOs")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--n", type=int, default=5, help="Number of results to return")
    p_search.add_argument("--k", type=int, default=3, help="Number of topics to display")

    args = parser.parse_args()
    
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    if args.command == "ingest":
        
        print("Make sure to run beforehand: docker run -d -p 6335:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant")
        
        ingestor = Ingestor(
                qdrant_client=client, 
                collection_name=settings.COLLECTION_NAME
        )
        ingestor.ingest(file_path=args.file)

    elif args.command == "search":
        engine = Engine(client)
        engine.search(query=args.query, limit=args.n, k=args.k)


if __name__ == "__main__":
    main()