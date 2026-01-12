# 📁 api.py - FastAPI REST API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from pathlib import Path
import os

from rag_chain import ComplianceRAG
from ner_infer import FinancialNER

# Change to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Compliance Copilot API",
    description="AI-powered compliance assistant with RAG and NER",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models at startup
rag = None
ner = None

@app.on_event("startup")
async def startup_event():
    global rag, ner
    print("🚀 Initializing models...")
    try:
        rag = ComplianceRAG()
        ner = FinancialNER()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    verbose: bool = False

class QuestionResponse(BaseModel):
    question: str
    answer: str
    retrieved_docs: int

class NERRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    type: str
    tokens: List[str]

class NERResponse(BaseModel):
    text: str
    entities: List[Entity]

# Health check endpoint
@app.get("/")
def root():
    return {
        "message": "Financial Compliance Copilot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "ner": "/ner",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_loaded": rag is not None,
        "ner_loaded": ner is not None
    }

# RAG endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a compliance question using RAG
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get retriever to count docs
        retriever = rag.index.as_retriever(similarity_top_k=3)
        docs = retriever.retrieve(request.question)
        
        # Get answer
        answer = rag.answer(request.question, verbose=request.verbose)
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            retrieved_docs=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NER endpoint
@app.post("/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest):
    """
    Extract financial entities from text
    """
    if ner is None:
        raise HTTPException(status_code=503, detail="NER system not initialized")
    
    try:
        entities = ner.extract_grouped(request.text)
        
        return NERResponse(
            text=request.text,
            entities=[Entity(**e) for e in entities]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to True for development
    )