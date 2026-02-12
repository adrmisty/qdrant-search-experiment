# Document search experiment

An experimental pipeline designed to test the ingestion and semantic search capabilities for document search. This project utilizes **Qdrant** as a vector database and **LlamaIndex** / **SentenceTransformers** to generate multilingual embeddings for title-based retrieval. Tested on a 7k document dataset, returning responses in an average of 0.2-0.3 seconds.

## Project Overview

The goal of this experiment is to validate whether documents can be effectively searched and retrieved using semantic vector embeddings of their titles and metadata.

### Core Components

1.  **Ingestor**: A batch processing module that reads documents metadata from a raw JSON, creates `LlamaIndex` documents, and uploads them to Qdrant.
2.  **Vector database**: Uses **Qdrant** to store and index high-dimensional vectors.
3.  **Search engine**: A lightweight semantic searcher that encodes user queries and retrieves the most relevant KOs based on cosine similarity.
4.  **Embeddings**: Uses the `BAAI/bge-m3` model to support multilingual titles.

---

## Getting Started

### Prerequisites

* **Python 3.10+**
* **Docker** (to run the Qdrant server)
* **NVIDIA GPU** (for faster embedding generation)

### Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:adrmisty/qdrant-search-experiment.git
    cd qdrant-search
    ```

2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The application is controlled via a CLI entry point: `qdrant_search.main`.

### 1. Start QDrant server

Before running any Python scripts, you must have the Qdrant database running in a Docker container.

```bash
# on an unused port, e.g. 6335
docker run -d -p 6335:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

### 2. Ingest KOs.
   ```bash
   python3 -m qdrant_search.main ingest
   ```

### 3. Search KOs.
    e.g. in a NLP-related semantic field.
   ```bash
   python3 -m qdrant_search.main search "dialogue systems"
   python3 -m qdrant_search.main search "neural MT" --n 10 --k 5   
   ```

## Project structure

```
qdrant-search-experiment/
├── requirements.txt            
├── README.md                   
└── qdrant_search/          # src
    ├── __init__.py
    ├── main.py                 
    ├── config.py               
    ├── ingestor.py             # parsing JSON and indexing into Qdrant
    ├── search.py               # query encoding and searching
    └── data/
        └── logical_layer.json  # input data
```


### Configuration

Configuration is managed via the `pydantic-settings` library. You can override defaults (to be defined in `config.py`) using environment variables or a .env file.


## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python.*
