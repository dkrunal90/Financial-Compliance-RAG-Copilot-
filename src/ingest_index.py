# 📁 4️⃣ ingest_index.py

# 👉 Build vector index for RAG

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

def main():
    print("="*80)
    print("Building Vector Index for RAG")
    print("="*80)
    
    # Create indexes directory
    Path("indexes").mkdir(parents=True, exist_ok=True)
    
    # Load documents
    print("\n📄 Loading documents from data/docs/...")
    try:
        docs = SimpleDirectoryReader("data/docs").load_data()
        print(f"   ✅ Loaded {len(docs)} documents")
    except Exception as e:
        print(f"   ❌ Error loading documents: {e}")
        print("   Make sure you've run: python src/create_sample_data.py")
        return
    
    if len(docs) == 0:
        print("   ❌ No documents found in data/docs/")
        return
    
    # Show document details
    for i, doc in enumerate(docs, 1):
        print(f"\n   Document {i}:")
        print(f"   - Source: {doc.metadata.get('file_name', 'unknown')}")
        print(f"   - Length: {len(doc.text)} characters")
        print(f"   - Preview: {doc.text[:100]}...")
    
    # Set up embedding model
    print("\n🔧 Setting up embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20
    
    # Build vector index
    print("\n📊 Building vector index...")
    print("   (This may take a minute...)")
    
    index = VectorStoreIndex.from_documents(
        docs,
        show_progress=True
    )
    
    # Test retrieval before saving
    print("\n🧪 Testing retrieval...")
    retriever = index.as_retriever(similarity_top_k=2)
    test_queries = [
        "KYC documents",
        "AML monitoring",
        "compliance requirements"
    ]
    
    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"   Query: '{query}' → Retrieved {len(results)} documents")
        if results:
            print(f"      Top result score: {results[0].score:.4f}")
    
    # Save index to disk
    print("\n💾 Saving index to disk...")
    index.storage_context.persist(persist_dir="indexes/simple_index")
    
    print("\n✅ Vector index built and saved successfully!")
    print(f"📁 Location: indexes/simple_index/")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()