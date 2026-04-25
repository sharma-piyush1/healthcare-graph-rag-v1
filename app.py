import streamlit as st
import logging
from streamlit_agraph import agraph, Node, Edge, Config

logging.getLogger().setLevel(logging.ERROR)
from graph_rag_agent import app as graph_agent

st.set_page_config(page_title="Clinical Graph-RAG Assistant", page_icon="⚕️", layout="wide")
st.title("⚕️ Clinical Graph-RAG Assistant")
st.markdown("Deterministic medical retrieval powered by Neo4j and Llama-3.3.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Node color mapping based on medical entity type
COLOR_MAP = {
    "Disease": "#FF4B4B",
    "Drug": "#00CC96",
    "Symptom": "#FFA15A"
}

def render_graph(graph_data):
    """Converts the state dictionary into an interactive Agraph UI component."""
    nodes = []
    edges = []
    
    # Process Nodes
    for name, node_type in graph_data.get("nodes", []):
        nodes.append( Node(id=name, 
                           label=name, 
                           size=25, 
                           color=COLOR_MAP.get(node_type, "#888888")) )
        
    # Process Edges
    for source, relationship, target in graph_data.get("edges", []):
        edges.append( Edge(source=source, 
                           label=relationship, 
                           target=target, 
                           type="CURVE_SMOOTH") )
        
    config = Config(width=700, 
                    height=400, 
                    directed=True,
                    physics=True, 
                    hierarchical=False)
    
    if nodes:
        agraph(nodes=nodes, edges=edges, config=config)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph_data" in message and message["graph_data"]["nodes"]:
            render_graph(message["graph_data"])

if prompt := st.chat_input("Enter clinical query (e.g., 'treatment for high blood sugar')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying Knowledge Graph..."):
            try:
                result = graph_agent.invoke({"query": prompt})
                
                response_text = result.get("generation", "Error generating response.")
                graph_data = result.get("graph_data", {"nodes": set(), "edges": []})
                
                st.markdown(response_text)
                
                if graph_data["nodes"]:
                    st.markdown("### Deterministic Knowledge Graph Trace")
                    render_graph(graph_data)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "graph_data": graph_data
                })
            
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")