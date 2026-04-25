import os
import logging
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the application from your existing agent script
from graph_rag_agent import app, neo4j_driver

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

def generate_evaluation_data():
    """Generates responses for our golden QA dataset using the Graph-RAG agent."""
    logging.info("Generating responses for Golden Dataset...")
    
    questions = [
        "What is the recommended treatment for Type 2 Diabetes?",
        "Why is Ibuprofen contraindicated for Chronic Kidney Disease?"
    ]
    
    # Corrected: Flat list of strings instead of list of lists
    ground_truths = [
        "The recommended treatment is Metformin as first-line therapy.",
        "Ibuprofen is an NSAID that is contraindicated for Chronic Kidney Disease because it can cause Acute Renal Failure."
    ]

    answers = []
    contexts = []

    for q in questions:
        result = app.invoke({"query": q})
        answers.append(result["generation"])
        contexts.append([result["retrieved_context"]])
        
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

def run_evaluation():
    """Executes the Ragas evaluation triad."""
    dataset = generate_evaluation_data()
    
    logging.info("Initializing Evaluation Models...")
    # Upgraded to 70B model to ensure strict adherence to Ragas output parsers
    eval_llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logging.info("Running Ragas Evaluation (Faithfulness & Context Recall)...")
    
    # Suppress verbose Ragas logging
    logging.getLogger("ragas").setLevel(logging.ERROR)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n" + "="*40)
    print("RAGAS EVALUATION METRICS")
    print("="*40)
    print(f"Faithfulness Score:   {results.get('faithfulness', 0.0):.2f} / 1.00")
    print(f"Context Recall Score: {results.get('context_recall', 0.0):.2f} / 1.00")
    print("="*40)

if __name__ == "__main__":
    try:
        run_evaluation()
    finally:
        neo4j_driver.close()