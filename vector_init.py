import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Updated import path to match our 0.1.x environment
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# 1. Infrastructure Initialization
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "securepassword123"))
)

qdrant_client = QdrantClient("localhost", port=6333)

# Using a lightweight, fast CPU-optimized embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_SIZE = 384 

def initialize_vector_store():
    collection_name = "medical_entities"
    
    # Recreate collection for idempotency
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logging.info(f"Qdrant collection '{collection_name}' initialized.")

    # Extract entities from Neo4j
    points = []
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n) WHERE 'Disease' IN labels(n) OR 'Drug' IN labels(n) OR 'Symptom' IN labels(n) RETURN n.id AS id, n.name AS name, labels(n)[0] AS type")
        
        for idx, record in enumerate(result):
            text_to_embed = f"{record['type']}: {record['name']}"
            vector = embeddings.embed_query(text_to_embed)
            
            points.append(PointStruct(
                id=idx,
                vector=vector,
                payload={"graph_id": record['id'], "name": record['name'], "type": record['type']}
            ))
            
    # Upsert to Qdrant
    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        logging.info(f"Successfully embedded and stored {len(points)} entities in Qdrant.")

if __name__ == "__main__":
    try:
        initialize_vector_store()
    finally:
        neo4j_driver.close()