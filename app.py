import gradio as gr
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

from src.rag_chain import ComplianceRAG
from src.ner_infer import FinancialNER

# Initialize
print("Loading models...")
rag = ComplianceRAG()
ner = FinancialNER()

def ask_question(question):
    try:
        return rag.answer(question, verbose=False)
    except Exception as e:
        return f"Error: {e}"

def extract_entities(text):
    try:
        entities = ner.extract_grouped(text)
        if not entities:
            return "No entities found"
        return "\n".join([f"• {e['text']} → {e['type']}" for e in entities])
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Financial Compliance Copilot") as demo:
    gr.Markdown("# 💼 Financial Compliance Copilot")
    
    with gr.Tab("📚 Ask Questions"):
        question_input = gr.Textbox(label="Question", placeholder="What documents are required for KYC?")
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=5)
        ask_btn.click(ask_question, question_input, answer_output)
    
    with gr.Tab("🔍 Extract Entities"):
        text_input = gr.Textbox(label="Text", placeholder="Rahul transferred money to HDFC account 1234567890")
        extract_btn = gr.Button("Extract", variant="primary")
        entities_output = gr.Textbox(label="Entities", lines=5)
        extract_btn.click(extract_entities, text_input, entities_output)

demo.launch()
