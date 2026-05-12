# search.py
# /search engine based on QDRANT and OPENSEARCH vector databases/
# adriana r.f. (@adrmisty)
# feb/may-2026

import time
from abc import ABC, abstractmethod
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import settings

class BaseSearch(ABC):
    def __init__(self, client):
        self.client = client
        self.model = settings.EMBED_MODEL

    @abstractmethod
    def _execute_query(self, query_vector: list, limit: int) -> list:
        """Execute DB-specific search and return a standardized list of dicts: {score, title, topics}"""
        pass

    def search(self, query: str, limit: int = 5, k: int = 3):
        print(f"\n *** EXPERIMENT > Query: '{query}' *** ")
        
        start_time = time.time()
        query_vector = self.model.get_query_embedding(query)

        try:
            hits = self._execute_query(query_vector, limit)
        except Exception as e:
            print(f"\t(!) Search failed. Error: {e}")
            return

        elapsed = time.time() - start_time

        print(f">>> Found {len(hits)} relevant results in {elapsed:.4f} seconds:\n")
        print("-" * 60)
        for hit in hits:
            topics = ", ".join(hit.get("topics", [])[:k])
            print(f"- Score: {hit['score']:.4f}")
            print(f"- Title: {hit.get('title', '---')}")
            print(f"- Topics: {topics}")
            print("-" * 60)


# *** extension: OpenSearch + Qdrant (over a common interface)

class KoQdrantSearch(BaseSearch):
    def _execute_query(self, query_vector: list, limit: int) -> list:
        response = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )
        raw_hits = response if isinstance(response, list) else getattr(response, 'points', response)
        
        return [{
            "score": hit.score,
            "title": hit.payload.get("title", "---"),
            "topics": hit.payload.get("topics", [])
        } for hit in raw_hits]


class KoOpenSearch(BaseSearch):
    def _execute_query(self, query_vector: list, limit: int) -> list:
        os_query = {
            "size": limit,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": limit}}}
        }
        response = self.client.search(index=settings.INDEX_NAME, body=os_query)
        
        return [{
            "score": hit["_score"],
            "title": hit["_source"].get("metadata", hit["_source"]).get("title", "---"),
            "topics": hit["_source"].get("metadata", hit["_source"]).get("topics", [])
        } for hit in response["hits"]["hits"]]