# search.py
# /search engine based on QDRANT vector database/
# adriana r.f. (@adrmisty)
# feb-2026

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import settings
from qdrant_search.config import settings
import time

class KoSearch:
    def __init__(self, client):
        self.client = client
        self.model = HuggingFaceEmbedding(model_name=settings.MODEL_NAME, cache_folder=settings.CACHE_DIR, trust_remote_code=True, device="cuda")

    def search(self, query: str, limit: int = 5, k: int = 3):
        """Semantic vector search for a given query, in the vectorial DB.."""
        
        print(f"\n *** EXPERIMENT > Query: '{query}' *** ")
        
        start_time = time.time()
        query_vector = self.model.get_query_embedding(query)

        try: # .search() deprecated now, attributeerror
            response = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                limit=limit
            )
            
            if isinstance(response, list):
                hits = response
            else:
                hits = getattr(response, 'points', response)
            
        except Exception as e:
            print(f"\t(!) Search failed - no index found. Error: {e}")
            return

        end_time = time.time()
        elapsed = end_time - start_time

        print(f">>> Found {len(hits)} relevant results in {elapsed:.4f} seconds:\n")
        print("-" * 60)
        for hit in hits:
            score = hit.score
            payload = hit.payload
            title = payload.get("title", "---")
            topics = ", ".join(payload.get("topics", [])[:k])
            
            print(f"- Score: {score:.4f}")
            print(f"- Title: {title}")
            print(f"- Topics: {topics}")
            print("-" * 60)