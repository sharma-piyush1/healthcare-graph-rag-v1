import os
import logging
from typing import TypedDict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# 1. State Definition
class GraphRAGState(TypedDict):
    query: str
    entities: dict
    retrieved_context: str
    graph_data: dict  # NEW: Stores structured data for the UI
    generation: str

# 2. Infrastructure Setup
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "securepassword123"))
)

qdrant_client = QdrantClient("localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
COLLECTION_NAME = "medical_entities"

# 3. Node Functions
def extract_entities(state: GraphRAGState):
    """Extracts raw medical concepts from the user query."""
    logging.info("NODE: Extracting Entities")
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract potential diseases, drugs, or symptoms from the query. Return strict JSON: {{\"concepts\": [\"concept1\", \"concept2\"]}}. If none, return empty list."),
        ("human", "{query}")
    ])
    chain = prompt | llm | parser
    return {"entities": chain.invoke({"query": state["query"]})}

def hybrid_retrieve(state: GraphRAGState):
    """Maps semantic concepts to exact Graph Nodes, then executes deterministic Cypher queries."""
    logging.info("NODE: Hybrid Retrieval (Vector -> Graph)")
    concepts = state["entities"].get("concepts", [])
    context = []
    
    # NEW: Data structure for UI rendering
    graph_data = {"nodes": set(), "edges": []} 
    
    with neo4j_driver.session() as session:
        for concept in concepts:
            vector = embeddings.embed_query(concept)
            search_result = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=1
            )
            
            if not search_result:
                continue
                
            matched_node = search_result[0].payload
            official_name = matched_node['name']
            node_type = matched_node['type']
            
            logging.info(f"Mapped user concept '{concept}' -> Graph Node '{official_name}' ({node_type})")
            graph_data["nodes"].add((official_name, node_type))
            
            if node_type == "Disease":
                result = session.run(
                    "MATCH (d:Disease {name: $name})-[r]-(connected) "
                    "RETURN d.name AS entity, type(r) AS relationship, connected.name AS target, labels(connected)[0] AS target_type, d.clinical_guideline AS guideline",
                    name=official_name
                )
                for record in result:
                    context.append(f"{record['entity']} {record['relationship']} {record['target']} (Guideline: {record['guideline']})")
                    graph_data["nodes"].add((record['target'], record['target_type']))
                    graph_data["edges"].append((record['entity'], record['relationship'], record['target']))
                    
            elif node_type == "Drug":
                result = session.run(
                    "MATCH (dr:Drug {name: $name})-[r]-(connected) "
                    "RETURN dr.name AS entity, type(r) AS relationship, connected.name AS target, labels(connected)[0] AS target_type",
                    name=official_name
                )
                for record in result:
                    context.append(f"{record['entity']} {record['relationship']} {record['target']}")
                    graph_data["nodes"].add((record['target'], record['target_type']))
                    graph_data["edges"].append((record['entity'], record['relationship'], record['target']))

    context_str = "\n".join(set(context)) if context else "No relevant medical guidelines found in the knowledge graph."
    return {"retrieved_context": context_str, "graph_data": graph_data}

def generate_response(state: GraphRAGState):
    """Generates the response anchored strictly to the graph context."""
    logging.info("NODE: Generating Grounded Response")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical AI assistant. Answer the user query using ONLY the provided knowledge graph context. Do not hallucinate."),
        ("human", "Query: {query}\n\nGraph Context:\n{context}")
    ])
    chain = prompt | llm
    return {"generation": chain.invoke({"query": state["query"], "context": state["retrieved_context"]}).content}

# 4. Graph Construction
workflow = StateGraph(GraphRAGState)
workflow.add_node("extract", extract_entities)
workflow.add_node("retrieve", hybrid_retrieve)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 5. Execution Test
if __name__ == "__main__":
    # Deliberately using non-exact medical terminology to test the semantic routing
    test_query = "What medication is recommended if a patient has high blood sugar?"
    print(f"\n--- Processing Query: '{test_query}' ---\n")
    
    result = app.invoke({"query": test_query})
    
    print("\n--- Final Output ---")
    print(result["generation"])
    
    neo4j_driver.close()