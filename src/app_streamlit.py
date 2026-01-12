# 📁 app_streamlit.py - Streamlit Web Interface

import streamlit as st
from pathlib import Path
import os

from rag_chain import ComplianceRAG
from ner_infer import FinancialNER

# Change to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Page config
st.set_page_config(
    page_title="Financial Compliance Copilot",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .entity-person { background-color: #ffcccc; }
    .entity-org { background-color: #ccffcc; }
    .entity-account { background-color: #ccccff; }
    </style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    with st.spinner("🔧 Loading AI models..."):
        rag = ComplianceRAG()
        ner = FinancialNER()
    return rag, ner

# Header
st.markdown('<h1 class="main-header">💼 Financial Compliance Copilot</h1>', unsafe_allow_html=True)
st.markdown("**AI-powered assistant for compliance queries and entity extraction**")

# Load models
try:
    rag, ner = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📚 Ask Questions (RAG)", "🔍 Extract Entities (NER)"])

# Tab 1: RAG Q&A
with tab1:
    st.header("Ask Compliance Questions")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What documents are required for KYC?",
        key="rag_question"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", key="ask_btn")
    with col2:
        verbose = st.checkbox("Show detailed retrieval info", value=False)
    
    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get retriever info
                retriever = rag.index.as_retriever(similarity_top_k=3)
                docs = retriever.retrieve(question)
                
                # Get answer
                answer = rag.answer(question, verbose=verbose)
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.info(answer)
                
                # Show retrieved docs
                if docs:
                    with st.expander(f"📄 Retrieved {len(docs)} documents"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Document {i}** (score: {doc.score:.3f})")
                            st.text(doc.text[:300] + "...")
                            st.divider()
                            
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Tab 2: NER
with tab2:
    st.header("Extract Financial Entities")
    
    text = st.text_area(
        "Enter text:",
        placeholder="e.g., Rahul transferred money to HDFC account 1234567890",
        height=100,
        key="ner_text"
    )
    
    extract_button = st.button("Extract Entities", type="primary", key="extract_btn")
    
    if extract_button and text:
        with st.spinner("🔍 Extracting entities..."):
            try:
                entities = ner.extract_grouped(text)
                
                if entities:
                    st.markdown("### 📋 Extracted Entities")
                    
                    # Display entities with colors
                    entity_html = text
                    for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
                        entity_type = entity['type'].lower()
                        color_class = f"entity-{entity_type.split('_')[0]}"
                        replacement = f'<span class="entity-box {color_class}">{entity["text"]} <small>({entity["type"]})</small></span>'
                        entity_html = entity_html.replace(entity['text'], replacement)
                    
                    st.markdown(entity_html, unsafe_allow_html=True)
                    
                    # Table view
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Entity**")
                        for e in entities:
                            st.text(e['text'])
                    with col2:
                        st.markdown("**Type**")
                        for e in entities:
                            st.text(e['type'])
                else:
                    st.warning("No entities found")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This copilot uses:
    - **RAG**: LlamaIndex + FAISS + Llama 3
    - **NER**: Fine-tuned DistilBERT
    
    **Example Questions:**
    - What documents are needed for KYC?
    - What should be flagged for AML?
    
    **Example Text for NER:**
    - Rahul sent money to HDFC
    - Transfer to account 1234567890
    """)
    
    st.divider()
    
    if st.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()