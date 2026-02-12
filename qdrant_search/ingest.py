# ingestor.py
# /ingests logical layer documents from JSON/
# adriana r.f. (@adrmisty)
# feb-2026

import json
from typing import Dict, Any
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client.http import models
from tqdm import tqdm
from .config import settings

class KoIngestor:
    """Ingestion of cdocuments as Llama Index documents."""
    
    def __init__(self, qdrant_client, collection_name):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.vector_store_db = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store_db)
        self.embed_model = HuggingFaceEmbedding(model_name=settings.MODEL_NAME, cache_folder=settings.CACHE_DIR, trust_remote_code=True, device="cuda")
        
    def parse(self, item: Dict[str, Any]) -> Document:
        """Parses a JSON item (title + relevant fields, and metadata) into a LlamaIndex document to be ingested and embedded onto a vector DB."""
        title = item.get("title", "")
        #description = item.get("description", "") or ""
        #keywords = ", ".join(item.get("keywords", []) or [])
        
        text = f"Title: {title}"
        #text_content = f"Title: {title}\nDescription: {description}\nKeywords: {keywords}"

        # TODO: adapt to whatever JSON structure
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
            text=text,
            metadata=payload,
            # ignore these id's
            excluded_embed_metadata_keys=["url", "original_id", "project_id", "title"]
        )


    def ingest(self, file_path: str):
        """Loads the JSON file, parses all items, and indexes them into Qdrant."""
        self._recreate_collection(self.collection_name)
        
        print("*** EXPERIMENT >>> Document ingestion into QDRANT ***")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"\t(!) > JSON input file not found: {file_path}")

        print(f"> Loading data from {file_path}...")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents_to_ingest = []
        print(f"> Parsing {len(raw_data)} items...")
        
        for i in tqdm(range(0, len(raw_data))):
            item = raw_data[i]
            
            try:
                doc = self.parse(item)
                documents_to_ingest.append(doc)
            except Exception as e:
                print(f"\t(!) > Skipping item due to error: {e}")
                continue

        print(f"> Generating embeddings and indexing into '{self.collection_name}'...")
        VectorStoreIndex.from_documents(
            documents_to_ingest, 
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        print(">>> Logical layer KO ingestion done!")

    # -------------------------------------------------------------------------------------------

    def _recreate_collection(self, collection_name):
        """Deletes and re-creates the Qdrant collection with correct config."""
        
        self._delete_collection_if_exists(collection_name)
        
        print(f'[(set-up) Creating new collection > {collection_name}...]')
        
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.dimension, 
                distance=models.Distance.COSINE
            )
        )

    def _delete_collection_if_exists(self, collection_name):
        if self.qdrant_client.collection_exists(collection_name=collection_name):
            print(f'[(set-up) Deleting previous collection > {collection_name}...]')
            self.qdrant_client.delete_collection(collection_name=collection_name)