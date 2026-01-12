💼 Financial Compliance RAG Copilot

An AI‑powered copilot for financial compliance teams that combines **Retrieval‑Augmented Generation (RAG)** with a **fine‑tuned NER model** to answer policy questions, highlight risky entities, and evaluate answer quality.[1]

***

## 🎯 What this copilot can do

- **Ask policy questions in plain English**  
  RAG‑based Q&A over your internal compliance corpus using **LlamaIndex + FAISS + Llama 3** via Ollama.[1]

- **Spot critical financial entities instantly**  
  Fine‑tuned **DistilBERT** NER to tag PERSON, ORG, ACCOUNT_NUMBER and other sensitive fields from raw text or chat input.[1]

- **Measure answer quality, not just vibes**  
  BLEU‑based evaluation against gold Q&A pairs so you can track how good the assistant really is over time.[1]

- **Use it however you like**  
  - Interactive **CLI** for power users  
  - **FastAPI REST API** for backend integration  
  - **Streamlit Web UI** for analysts and reviewers[1]

***

## 🚀 Quick start in 4 steps

### 1️⃣ Clone and environment

```bash
git clone https://github.com/dkrunal90/Financial-Compliance-RAG-Copilot-.git
cd Financial-Compliance-RAG-Copilot-

# Create virtual environment
conda create -p venv python=3.10
conda activate ./venv

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Install Ollama + LLM

```bash
# Install from https://ollama.ai
ollama pull llama3.2
```

### 3️⃣ Prepare data and models

```bash
# Generate synthetic compliance Q&A + NER data
python src/create_sample_data.py

# Fine-tune DistilBERT for financial NER (~5 minutes)
python src/ner_train.py

# Build FAISS vector index for RAG
python src/ingest_index.py
```

### 4️⃣ Run your copilot

```bash
# Interactive CLI
python src/chat_cli.py

# REST API
python src/api.py

# Web UI
streamlit run src/app_streamlit.py
```

***

## 📊 What using it looks like

### 💬 Chat over policies (CLI)

```text
You: ask What documents are required for KYC?

Assistant:
For KYC, you typically need:
1. Government ID (PAN/SSN/Passport)
2. Address proof
3. Recent photograph
4. Bank account details

[Answer grounded in retrieved compliance documents]
```

### 🧾 Financial NER extraction

```text
You: ner Rahul transferred money to HDFC account 1234567890

Extracted Entities:
 • Rahul        → PERSON
 • HDFC         → ORG
 • 1234567890   → ACCOUNT_NUMBER
```

***

## 🐳 One‑command Docker deploy

```bash
docker-compose up -d
```

This spins up the FastAPI backend and supporting services defined in `docker-compose.yml` so you can use the API and UI without manual setup.[1]

***

## 📂 Project layout

```text
├── src/
│   ├── create_sample_data.py   # Generate synthetic training data
│   ├── ner_train.py            # Fine-tune DistilBERT for NER
│   ├── ner_infer.py            # NER inference utilities
│   ├── ingest_index.py         # Build FAISS vector index for RAG
│   ├── rag_chain.py            # RAG orchestration with LlamaIndex
│   ├── evaluate_bleu.py        # BLEU-based answer evaluation
│   ├── chat_cli.py             # CLI entrypoint
│   ├── api.py                  # FastAPI REST service
│   └── app_streamlit.py        # Streamlit web UI
├── data/                       # Generated sample data
├── models/                     # Trained NER / saved checkpoints
├── indexes/                    # Vector indexes (FAISS)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```


***

## 🔧 Under the hood

- **NER**: DistilBERT (HuggingFace), fine‑tuned for financial entities.[1]
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for dense semantic search.[1]
- **Vector store**: FAISS for fast similarity search over compliance chunks.[1]
- **RAG orchestration**: LlamaIndex to wire loaders, index, retriever, and LLM together.[1]
- **LLM**: Llama 3.2 served locally via Ollama for low‑latency, private inference.[1]

***

## 📝 License & 🤝 Contributions

- Licensed under **MIT** – use it, tweak it, ship it.[1]
- Pull requests are very welcome: new entity types, better evaluation metrics, or production hardening (auth, logging, tracing) are all great places to contribute.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/30658921/9063d921-4282-4257-9b0a-5065eca98c3c/README-2.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/30658921/230e3302-67ef-4e82-aea8-a604a64ca6ce/Krunal-Desai_AI-ML-Engineer.pdf)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/30658921/9f9d9777-375d-45ae-a5e0-a9350745029c/image.jpg)
