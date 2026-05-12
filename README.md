# Document Search Experiment

An experimental pipeline designed to test ingestion and semantic search capabilities for document retrieval. This project utilizes **Qdrant** and **OpenSearch** as vector databases and **LlamaIndex** / **HuggingFace** to generate multilingual embeddings for title-based retrieval. Tested on a 7k document dataset, returning responses in an average of 0.2-0.3 seconds.

## Project Overview

Validates whether documents can be effectively searched and retrieved using semantic vector embeddings of their titles and metadata across different vector database engines.

### Core Components

1.  **Ingestor**: A batch processing module that reads document metadata from a raw JSON, creates `LlamaIndex` documents, and uploads them to the chosen vector DB.
2.  **Vector Databases**: Supports **Qdrant** and **OpenSearch** to store and index high-dimensional vectors.
3.  **Search Engine**: A lightweight semantic searcher that encodes user queries and retrieves the most relevant KOs based on cosine similarity/k-NN.
4.  **Embeddings**: Uses the `BAAI/bge-m3` model to support multilingual titles.

---

## Getting Started

### Prerequisites

* **Python 3.10+**
* **Docker** (to run the DB servers)
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

The application is controlled via a CLI entry point: `qdrant_open_search.main`. 
You can specify the target database engine using the `--engine` flag (`qdrant` or `opensearch`). Default is `qdrant`.

### 1. Start Database Servers

Start the respective database running in a Docker container before executing Python scripts.

**Qdrant:**
```bash
# on an unused port, e.g. 6335
docker run -d -p 6335:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

**OpenSearch:**

```bash
docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest
```

### 2. Ingest KOs.

```bash
python3 -m qdrant_open_search.main --engine qdrant ingest
python3 -m qdrant_open_search.main --engine opensearch ingest
```

### 3. Search KOs.

```bash
python3 -m qdrant_open_search.main --engine qdrant search "mediterranean landscape"
python3 -m qdrant_open_search.main --engine opensearch search "carbon flow" --n 10 --k 5   

```

## Project structure

```
qdrant-search-experiment/
├── requirements.txt            
├── README.md                   
└── qdrant_search/          # src
    ├── __init__.py
    ├── main.py                 # CLI entry point
    ├── config.py               # Pydantic settings
    ├── ingest.py               # JSON parsing & indexing (Qdrant & OpenSearch)
    ├── search.py               # query encoding & retrieval (Qdrant & OpenSearch)
    └── data/
        └── logical_layer.json  # input data

```

### Configuration

Configuration is managed via the `pydantic-settings` library. Override defaults in `config.py` using environment variables or a `.env` file.

## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python.*

```

```