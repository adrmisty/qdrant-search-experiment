# main.py
# /CLI for search experiments with Qdrant and OpenSearch/
# adriana r.f. (@adrmisty)
# feb/may-2026

import argparse
from qdrant_client import QdrantClient
from opensearchpy import OpenSearch

from qdrant_open_search.config import settings
from qdrant_open_search.ingest import KoQdrantIngestor, KoOpenSearchIngestor
from qdrant_open_search.search import KoQdrantSearch, KoOpenSearch

def main():
    parser = argparse.ArgumentParser(description="Document search experiment")
    parser.add_argument("--engine", choices=["qdrant", "opensearch"], default="qdrant", help="Vector DB engine to use")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- INGEST ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest logical layer JSON")
    p_ingest.add_argument("--file", type=str, default=settings.DATA_PATH, help="Path to input JSON file")

    # --- SEARCH ---
    p_search = subparsers.add_parser("search", help="Search for KOs")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--n", type=int, default=5, help="Number of results to return")
    p_search.add_argument("--k", type=int, default=3, help="Number of topics to display")

    args = parser.parse_args()
    
    if args.engine == "qdrant":
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        if args.command == "ingest":
            print("> (!) Make sure Qdrant is running on Docker...")
            ingestor = KoQdrantIngestor(qdrant_client=client, collection_name=settings.COLLECTION_NAME)
            ingestor.ingest(file_path=args.file)
        elif args.command == "search":
            engine = KoQdrantSearch(client)
            engine.search(query=args.query, limit=args.n, k=args.k)

    elif args.engine == "opensearch":
        client = OpenSearch(
            hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
            use_ssl=False
        )
        if args.command == "ingest":
            print("> (!) Make sure OpenSearch is running on Docker...")
            ingestor = KoOpenSearchIngestor(os_client=client, index_name=settings.INDEX_NAME)
            ingestor.ingest(file_path=args.file)
        elif args.command == "search":
            engine = KoOpenSearch(client)
            engine.search(query=args.query, limit=args.n, k=args.k)

if __name__ == "__main__":
    main()