** Archiecture

<img width="2848" height="1600" alt="image" src="https://github.com/user-attachments/assets/ea16ec64-fdf9-407a-9dfd-b15b70374ead" />

** Pipeline

<img width="1344" height="768" alt="image" src="https://github.com/user-attachments/assets/495c9d62-1439-4c23-879e-7910b780899b" />

***

# 💼 Financial Compliance RAG Copilot  
*Your AI teammate for “Where exactly does the policy say that?” moments.*[1]

Financial Compliance RAG Copilot turns dry policy PDFs into an interactive assistant that can **answer questions, spot risky entities, and grade its own answers** using RAG + NER + evaluation in one stack.[1]

***

## 🧩 What this copilot actually does

- 🔍 **Reads your policies for you**  
  Ask natural‑language questions and get grounded answers powered by **LlamaIndex + FAISS + Llama 3** via Ollama.[1]

- 🕵️ **Highlights who did what, where**  
  A fine‑tuned **DistilBERT** model tags PERSON, ORG, ACCOUNT_NUMBER and other sensitive entities directly from raw text.[1]

- 📏 **Checks its own work**  
  BLEU‑based evaluation against gold Q&A pairs so you can track quality instead of trusting vibes.[1]

- 🧑‍💻 **Meets you where you are**  
  - Terminal lover? Use the **CLI chat**.  
  - Building products? Plug into the **FastAPI REST API**.  
  - Analyst or reviewer? Open the **Streamlit Web UI** and just type.[1]

***

## ⚡ 5‑minute launch

### 1️⃣ Grab the repo and set up Python

```bash
git clone https://github.com/dkrunal90/Financial-Compliance-RAG-Copilot-.git
cd Financial-Compliance-RAG-Copilot-

conda create -p venv python=3.10
conda activate ./venv

pip install -r requirements.txt
```

### 2️⃣ Give it a brain (Ollama + Llama 3.2)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```

### 3️⃣ Teach it your world

```bash
# 1) Create synthetic compliance Q&A + NER data
python src/create_sample_data.py

# 2) Fine-tune DistilBERT for financial NER (~5 minutes)
python src/ner_train.py

# 3) Build FAISS vector index over your policy corpus
python src/ingest_index.py
```

### 4️⃣ Start talking to it

```bash
# CLI chat
python src/chat_cli.py

# REST API
python src/api.py

# Streamlit web app
streamlit run src/app_streamlit.py
```

Now you have a local “compliance Copilot” that understands your documents.[1]

***

## 🧪 What it feels like

### 💬 Chatting with policies (CLI)

```text
You: ask What documents are required for KYC?

Copilot:
For KYC, you typically need:
1. Government ID (PAN/SSN/Passport)
2. Address proof
3. Recent photograph
4. Bank account details

[Answer grounded in internal KYC policy sections]
```

### 🧾 Instant entity spotlight (NER)

```text
You: ner Rahul transferred money to HDFC account 1234567890

Entities:
 • Rahul        → PERSON
 • HDFC         → ORG
 • 1234567890   → ACCOUNT_NUMBER
```

Use it to quickly scan suspicious notes, chats, or transaction descriptions.[1]

***

## 🐳 Ship it with one command

Want it running like a service instead of a script?

```bash
docker-compose up -d
```

`docker-compose.yml` wires up the app, so you get the API + UI in containers with minimal fuss.[1]

***

## 🧱 Under the hood (for engineers)

```text
├── src/
│   ├── create_sample_data.py   # Synthetic Q&A + NER training data
│   ├── ner_train.py            # Fine-tune DistilBERT for financial NER
│   ├── ner_infer.py            # NER inference helpers
│   ├── ingest_index.py         # Build FAISS vector index
│   ├── rag_chain.py            # LlamaIndex RAG pipeline
│   ├── evaluate_bleu.py        # BLEU scoring for answers
│   ├── chat_cli.py             # CLI interface
│   ├── api.py                  # FastAPI REST backend
│   └── app_streamlit.py        # Streamlit dashboard
├── data/                       # Generated sample data
├── models/                     # Trained NER checkpoints
├── indexes/                    # FAISS indexes
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```


***

## 🛠️ Tech stack at a glance

- 🧠 **NER**: DistilBERT (HuggingFace), fine‑tuned for financial entities.[1]
- 🔡 **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for semantic search across policy chunks.[1]
- 📚 **Vector store**: FAISS for fast similarity search.  
- 🔗 **RAG orchestration**: LlamaIndex to connect loaders, index, retriever, and LLM.  
- 🤖 **LLM**: Llama 3.2 served locally via Ollama.  
- 🌐 **APIs & UI**: FastAPI + Streamlit.[1]

***

## 📜 License & 💡 How to contribute

- Licensed under **MIT** – feel free to fork, extend, and integrate.[1]
- Ideas for great PRs:  
  - New entity types (e.g., TAX_ID, CARD_NUMBER, SWIFT).  
  - Extra evaluation metrics (e.g., ROUGE, human feedback logging).  
  - Production features: auth, rate limiting, structured logging, OpenTelemetry.[1]

If you share your GitHub link later, this can be tuned even more as a portfolio‑style README specifically for hiring managers.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/30658921/9063d921-4282-4257-9b0a-5065eca98c3c/README-2.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/30658921/230e3302-67ef-4e82-aea8-a604a64ca6ce/Krunal-Desai_AI-ML-Engineer.pdf)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/30658921/9f9d9777-375d-45ae-a5e0-a9350745029c/image.jpg)
