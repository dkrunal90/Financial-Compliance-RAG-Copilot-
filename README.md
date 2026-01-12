# 💼 Financial Compliance RAG Copilot

AI-powered assistant for financial compliance queries, combining **Retrieval-Augmented Generation (RAG)** with **Fine-tuned Named Entity Recognition (NER)**.

## 🎯 Features

- **RAG-based Q&A**: Query internal compliance documents using LlamaIndex + FAISS + Llama 3
- **Financial NER**: Extract entities (PERSON, ORG, ACCOUNT_NUMBER) using fine-tuned DistilBERT
- **BLEU Evaluation**: Measure answer quality against gold-standard Q&A pairs
- **Interactive CLI**: User-friendly command-line interface
- **REST API**: FastAPI backend for integration
- **Web UI**: Streamlit interface

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/dkrunal90/Financial-Compliance-RAG-Copilot-.git
cd Financial-Compliance-RAG-Copilot-

# Create virtual environment
conda create -p venv python=3.10
conda activate ./venv

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama and Pull Model
```bash
# Install from https://ollama.ai
ollama pull llama3.2
```

### 3. Generate Data and Train Models
```bash
# Generate sample data
python src/create_sample_data.py

# Train NER model (~5 minutes)
python src/ner_train.py

# Build vector index
python src/ingest_index.py
```

### 4. Run Application
```bash
# Interactive CLI
python src/chat_cli.py

# REST API
python src/api.py

# Streamlit UI
streamlit run src/app_streamlit.py
```

## 📊 Usage Examples

### CLI Chat
```
💬 You: ask What documents are required for KYC?

🤖 Answer:
For KYC, you need:
1. Government ID (PAN/SSN/Passport)
2. Address proof
3. Recent photograph
4. Bank account details
```

### NER Extraction
```
💬 You: ner Rahul transferred money to HDFC account 1234567890

📋 Extracted Entities:
   • Rahul          → PERSON
   • HDFC           → ORG
   • 1234567890     → ACCOUNT_NUMBER
```

## 🐳 Docker Deployment
```bash
docker-compose up -d
```

## 📁 Project Structure
```
├── src/
│   ├── create_sample_data.py  # Generate training data
│   ├── ner_train.py            # Train NER model
│   ├── ner_infer.py            # NER inference
│   ├── ingest_index.py         # Build vector index
│   ├── rag_chain.py            # RAG orchestration
│   ├── evaluate_bleu.py        # Evaluation
│   ├── chat_cli.py             # CLI interface
│   ├── api.py                  # REST API
│   └── app_streamlit.py        # Web UI
├── data/                       # Sample data (generated)
├── models/                     # Trained models (generated)
├── indexes/                    # Vector indexes (generated)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 🔧 Technologies

- **NER**: DistilBERT (HuggingFace)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **RAG**: LlamaIndex
- **LLM**: Llama 3.2 (Ollama)

## 📝 License

MIT License

## 🤝 Contributing

Pull requests welcome!
EOF

