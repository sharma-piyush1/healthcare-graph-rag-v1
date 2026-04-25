# ⚕️ Clinical Graph-RAG System

This repository contains the Minimum Viable Product (MVP) for a deterministic, factually-grounded medical assistant. The architecture is designed to eliminate Large Language Model (LLM) hallucinations in clinical settings by enforcing strict, multi-hop graph retrieval before generating responses.

## Architecture Overview

Standard Retrieval-Augmented Generation (RAG) struggles with complex, interconnected clinical rules. This system resolves that liability using a **Hybrid Graph-Vector Architecture**:
1. **Semantic Routing:** User queries are vectorized locally using CPU-optimized Hugging Face models and routed through a Qdrant vector database to identify the exact medical entity.
2. **Deterministic Retrieval:** The identified entity maps to a Neo4j Knowledge Graph. A directed acyclic graph (DAG) orchestrated by LangGraph executes multi-hop Cypher queries to extract explicit clinical guidelines and contraindications.
3. **Grounded Generation:** The extracted constraints are passed to the Groq API (Llama-3.3-70b-versatile) for final synthesis.

## Engineering Constraints
This system was engineered to run locally under strict hardware limitations:
* **Compute:** Intel i5 CPU (No dedicated GPU)
* **Memory:** 16GB RAM limit (Neo4j and Qdrant container memory explicitly capped)
* **Budget:** $0 (Free-tier Groq API and local open-source embeddings)

## Core Tech Stack
* **Knowledge Graph:** Neo4j (Dockerized)
* **Vector Store:** Qdrant (Dockerized)
* **Orchestration:** LangGraph
* **Inference:** Groq API (Llama-3.3-70b-versatile)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local CPU)
* **Evaluation:** Ragas
* **Frontend:** Streamlit + `streamlit-agraph`

## Installation

1. Clone the repository and initialize the virtual environment:
```bash
conda create -n health_rag python=3.10 -y
conda activate health_rag
pip install -r requirements.txt
```

2. Configure environment variables in .env:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=securepassword123
GROQ_API_KEY=your_api_key_here
```

3. Deploy lightweight database containers:
```bash
# Neo4j (Memory Capped)
docker run --name neo4j-healthrag -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/abc123 -e NEO4J_server_memory_pagecache_size=1G -e NEO4J_server_memory_heap_initial__size=1G -e NEO4J_server_memory_heap_max__size=1G neo4j:5

# Qdrant (Memory Capped)
docker run --name qdrant-healthrag -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__MEMORY_LIMIT=1G -d qdrant/qdrant
```

## Execution Sequence

To initialize the system from scratch, run the following scripts in sequence:

1. Initialize Graph State: Enforces Neo4j schema constraints and injects the baseline mock clinical data.
```bash
python schema_init.py
```

2. Initialize Vector Index: Extracts entities from Neo4j, generates dense vectors locally, and maps them into Qdrant.
```bash
python vector_init.py
```

3. Launch User Interface: Starts the interactive Streamlit application with visual graph traceability.
```bash
streamlit run app.py
```

## Evaluation Metrics

The architecture includes an automated ragas evaluation pipeline (evaluate_rag.py) to mathematically verify system safety.

v1 MVP Benchmark (Golden Dataset):

Faithfulness: 1.00 / 1.00 (Zero hallucinations detected; answers fully inferred from retrieved context).

Context Recall: 1.00 / 1.00 (Perfect retrieval of required clinical guidelines).