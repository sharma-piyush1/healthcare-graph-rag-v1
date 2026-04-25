import os
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv

# Configure strict logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "securepassword123"))

class HealthcareKnowledgeGraph:
    def __init__(self, uri, auth):
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4j database.")
        except ServiceUnavailable as e:
            logging.error(f"Failed to connect to Neo4j. Ensure Docker container is running. Error: {e}")
            raise

    def close(self):
        self.driver.close()

    def initialize_schema(self):
        """Creates unique constraints to prevent duplication and index lookups."""
        queries = [
            "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE",
            "CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
        logging.info("Graph constraints initialized.")

    def populate_mock_data(self):
        """Injects deterministic multi-hop medical data for the MVP."""
        cypher_query = """
        // 1. Create Diseases
        MERGE (d1:Disease {id: 'D001', name: 'Type 2 Diabetes', clinical_guideline: 'Initiate Metformin as first-line therapy.'})
        MERGE (d2:Disease {id: 'D002', name: 'Hypertension', clinical_guideline: 'Target BP < 130/80 mmHg. Consider ACE inhibitors.'})
        MERGE (d3:Disease {id: 'D003', name: 'Chronic Kidney Disease', clinical_guideline: 'Monitor eGFR. Avoid nephrotoxic agents.'})

        // 2. Create Drugs
        MERGE (dr1:Drug {id: 'RX001', name: 'Metformin', class: 'Biguanide', max_dose_mg: 2000})
        MERGE (dr2:Drug {id: 'RX002', name: 'Lisinopril', class: 'ACE Inhibitor', max_dose_mg: 40})
        MERGE (dr3:Drug {id: 'RX003', name: 'Ibuprofen', class: 'NSAID', max_dose_mg: 3200})

        // 3. Create Symptoms
        MERGE (s1:Symptom {id: 'S001', name: 'Polyuria'})
        MERGE (s2:Symptom {id: 'S002', name: 'Elevated Blood Pressure'})
        MERGE (s3:Symptom {id: 'S003', name: 'Acute Renal Failure'})

        // 4. Create Relationships (Multi-hop pathways)
        MERGE (d1)-[:HAS_SYMPTOM]->(s1)
        MERGE (d2)-[:HAS_SYMPTOM]->(s2)
        
        MERGE (dr1)-[:TREATS {evidence_level: 'Class I'}]->(d1)
        MERGE (dr2)-[:TREATS {evidence_level: 'Class I'}]->(d2)
        
        // Contraindications and Adverse Events
        MERGE (dr3)-[:CONTRAINDICATED_FOR]->(d3)
        MERGE (dr3)-[:CAUSES_ADVERSE_EVENT]->(s3)
        """
        with self.driver.session() as session:
            session.run(cypher_query)
        logging.info("Mock medical knowledge graph populated successfully.")

if __name__ == "__main__":
    kg = HealthcareKnowledgeGraph(URI, AUTH)
    try:
        kg.initialize_schema()
        kg.populate_mock_data()
    finally:
        kg.close()