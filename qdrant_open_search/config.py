import os
from pydantic_settings import BaseSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class Settings(BaseSettings):
    # qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6337
    COLLECTION_NAME: str = "eufb_search"
    CACHE_DIR: str = "/tmp/huggingface"
    
    # opensearch settings
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    INDEX_NAME: str = "eufb_search"
    
    # multilingual embedding model (1st is the one used in ko-quality)
    #MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MODEL_NAME : str = "BAAI/bge-m3"
    dimension : int = 1024
    EMBED_MODEL : HuggingFaceEmbedding = HuggingFaceEmbedding(MODEL_NAME, cache_folder=CACHE_DIR, trust_remote_code=True)
    
    # input data
    DATA_PATH: str = os.path.join(os.path.dirname(__file__), "data", "logical_layer.ko_metadata.json")

settings = Settings()