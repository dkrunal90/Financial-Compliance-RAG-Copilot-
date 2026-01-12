# 📁 5️⃣ rag_chain.py

# 👉 RAG orchestration (retrieval + LLaMA)

from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama import ChatOllama
from pathlib import Path
import os

class ComplianceRAG:
    def __init__(self, model_name="llama3.2"):
        # Get project root and change to it
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        os.chdir(project_root)
        
        print(f"📁 Working directory: {project_root}")
        
        # CRITICAL: Set embedding model BEFORE loading index
        print("🔧 Initializing embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model
        Settings.chunk_size = 256
        Settings.chunk_overlap = 20
        
        # Load index
        try:
            print("📂 Loading index from disk...")
            index_path = Path("indexes/simple_index")
            
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index directory not found: {index_path.absolute()}\n"
                    "Please run: python src/ingest_index.py"
                )
            
            storage_context = StorageContext.from_defaults(
                persist_dir="indexes/simple_index"
            )
            self.index = load_index_from_storage(storage_context)
            
            # Test the index immediately
            print("🧪 Testing index...")
            test_retriever = self.index.as_retriever(similarity_top_k=1)
            test_docs = test_retriever.retrieve("test")
            print(f"✅ Index loaded - test retrieved {len(test_docs)} docs")
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            raise
        
        # Initialize Ollama
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0.1  # Lower temperature for factual answers
            )
            print(f"✅ LLM loaded (model: {model_name})")
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            print(f"Make sure Ollama is running and model is installed:")
            print(f"  ollama pull {model_name}")
            raise

    def answer(self, question, verbose=True, concise=False):
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            verbose: Whether to print detailed information
            concise: Whether to generate brief answers (better for BLEU evaluation)
            
        Returns:
            Answer string
        """
        if verbose:
         print(f"\n🔍 Query: '{question}'")
    
    # Create retriever
        retriever = self.index.as_retriever(
        similarity_top_k=3,
        verbose=verbose
    )
    
    # Retrieve documents
        docs = retriever.retrieve(question)
    
        if verbose:
            print(f"📚 Retrieved {len(docs)} documents")
    
        if len(docs) == 0:
        # Try a broader search
            print("⚠️  No results. Trying broader search...")
            retriever2 = self.index.as_retriever(similarity_top_k=5)
            docs = retriever2.retrieve("financial compliance")
        
            if len(docs) == 0:
                return "❌ No documents found in index. The index may be empty."
    
    # Show retrieved documents
        if verbose:
            for i, doc in enumerate(docs[:3]):
                print(f"\n   📄 Doc {i+1} (relevance: {doc.score:.3f}):")
                preview = doc.text[:200].replace('\n', ' ')
                print(f"   {preview}...")
    
    # Build context from retrieved docs
        context = "\n\n---\n\n".join([d.text for d in docs])
    
    # Create prompt based on mode
        if concise:
            prompt = f"""Answer the question using ONLY the context below. Be brief and concise - list only the key facts without explanations.

Context:
{context}

Question: {question}

Brief answer (maximum 2 sentences):"""
        else:
            prompt = f"""You are a financial compliance assistant. Answer the question using ONLY the information from the context below.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the context provided
- Be specific and cite relevant details
- If the context doesn't contain the answer, say so

Answer:"""
    
        if verbose:
            print("\n🤖 Generating answer with LLM...")
    
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"❌ Error generating response: {e}"

# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Testing Compliance RAG System")
    print("="*80)
    
    try:
        rag = ComplianceRAG()
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        exit(1)
    
    # Test questions
    questions = [
        "What documents are required for KYC?",
        "What should be flagged for AML monitoring?",
        "What address proof is acceptable?"
    ]
    
    for q in questions:
        print(f"\n{'='*80}")
        answer = rag.answer(q, verbose=True)
        print(f"\n💡 Answer:\n{answer}")
        print("="*80)