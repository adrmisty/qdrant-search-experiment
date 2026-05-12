# ingestor.py
# /ingests logical layer documents from JSON into Qdrant / OpenSearch/
# adriana r.f. (@adrmisty)
# feb/may-2026

# ingest.py
# /ingests logical layer documents from JSON into Qdrant / OpenSearch/
# adriana r.f. (@adrmisty)
# feb/may-2026

import json
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from qdrant_client.http import models
from .config import settings

class BaseIngestor(ABC):
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.embed_model = settings.EMBED_MODEL
        self.storage_context = None  # init in impls.

    @abstractmethod
    def _prepare_db(self):
        """Handle database specific recreation logic (index/collection deletion & creation)"""
        pass

    def parse(self, item: dict) -> Document:
        title = item.get("title", "")
        payload = {
            "title": title,
            "original_id": item.get("_id", {}).get("$oid", ""),
            "url": item.get("@id", ""),
            "topics": item.get("topics", []),
            "category": item.get("category", "Document"),
            "subcategory": item.get("subcategory", []),
            "language": ", ".join(item.get("languages", [])),
            "project_id": item.get("project_id", "")
        }
        return Document(
            text=f"Title: {title}",
            metadata=payload,
            excluded_embed_metadata_keys=["url", "original_id", "project_id", "title"]
        )

    def ingest(self, file_path: str):
        self._prepare_db()
        
        print(f"*** EXPERIMENT >>> Document ingestion into {self.__class__.__name__.replace('Ingestor', '')} ***")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"\t(!) > JSON input file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents_to_ingest = []
        for item in tqdm(raw_data):
            try:
                documents_to_ingest.append(self.parse(item))
            except Exception as e:
                print(f"\t(!) > Skipping item due to error: {e}")
                continue

        VectorStoreIndex.from_documents(
            documents_to_ingest, 
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        print(">>> Logical layer KO ingestion done!")


class KoQdrantIngestor(BaseIngestor):
    def __init__(self, qdrant_client, collection_name: str):
        super().__init__(collection_name)
        self.client = qdrant_client
        vector_store_db = QdrantVectorStore(client=self.client, collection_name=self.target_name)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store_db)

    def _prepare_db(self):
        if self.client.collection_exists(collection_name=self.target_name):
            print(f'[(set-up) Deleting previous collection > {self.target_name}...]')
            self.client.delete_collection(collection_name=self.target_name)
        
        print(f'[(set-up) Creating new collection > {self.target_name}...]')
        self.client.create_collection(
            collection_name=self.target_name,
            vectors_config=models.VectorParams(size=settings.dimension, distance=models.Distance.COSINE)
        )


class KoOpenSearchIngestor(BaseIngestor):
    def __init__(self, os_client, index_name: str):
        super().__init__(index_name)
        self.client = os_client
        
        os_vector_client = OpensearchVectorClient(
            endpoint=f"http://{settings.OPENSEARCH_HOST}:{settings.OPENSEARCH_PORT}",
            index=self.target_name,
            dim=settings.dimension,
            use_ssl=False,         
            verify_certs=False     
        )
        vector_store_db = OpensearchVectorStore(client=os_vector_client)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store_db)

    def _prepare_db(self):
        if self.client.indices.exists(index=self.target_name):
            print(f'[(set-up) Deleting previous index > {self.target_name}...]')
            self.client.indices.delete(index=self.target_name)
        print(f'[(set-up) Readying index > {self.target_name}...]')