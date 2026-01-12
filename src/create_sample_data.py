## 📁 1️⃣ create_sample_data.py

## 👉 Creates documents + NER data + evaluation data

# Used to interact with the operating system (paths, folders)
import os

# Used to read/write structured data (NER samples, QA pairs)
import json

# Used to generate random synthetic examples
import random

# Path is safer than string paths for file operations
from pathlib import Path

# Fix randomness so results are reproducible
random.seed(42)

# 🔹 Mock financial documents (RAG knowledge base)
# Dictionary where:
# key   → filename
# value → document content
DOCS = {
    "kyc_policy.txt": """
KYC Policy (Know Your Customer)
Required documents for individual customers:
1) Government ID: PAN (India) or SSN (US) or Passport
2) Address proof: utility bill, bank statement, or rental agreement
3) Recent photograph
4) Bank account details
""".strip(),

    "aml_rules.txt": """
AML Monitoring Rules
- Flag unusually large transactions
- Flag rapid movement of funds
- Identify suspicious counterparties
""".strip(),
}

## 🔹 Create required folders
def ensure_dirs():
    # Creates data/docs directory if it doesn't exist
    # parents=True → create parent folders if missing
    # exist_ok=True → don’t crash if folder already exists
    Path("data/docs").mkdir(parents=True, exist_ok=True)

## 🔹 Write documents to disk
def write_docs():
    # Loop through each document
    for name, content in DOCS.items():
        # Write content to a text file
        Path(f"data/docs/{name}").write_text(content, encoding="utf-8")

## 🔹 Simple tokenizer
def tokenize_simple(text):
    # Splits sentence by spaces
    # Used to keep NER token alignment easy
    return text.strip().split()

## 🔹 BIO label generator
def make_bio_labels(tokens, spans):
    # Start with all tokens labeled as Outside (O)
    labels = ["O"] * len(tokens)

    # spans = [(start_index, end_index, label)]
    for s, e, lab in spans:
        labels[s] = f"B-{lab}"     # Beginning of entity
        for i in range(s + 1, e):
            labels[i] = f"I-{lab}" # Inside entity

    return labels

## 🔹 Generate synthetic NER data
def gen_ner_examples(n=300):
    examples = []

    # Sample values for synthetic data
    names = ["Rahul", "Neha", "Aarav"]
    banks = ["HDFC", "ICICI"]
    accounts = ["1234567890", "9988776655"]

    for _ in range(n):
        # Pick random values
        name = random.choice(names)
        bank = random.choice(banks)
        account = random.choice(accounts)
        
        # Create sentence
        sentence = f"{name} transferred money to {bank} account {account}"
        
        tokens = tokenize_simple(sentence)
        
        # Find actual positions of entities in tokens
        # tokens will be: ["Rahul", "transferred", "money", "to", "HDFC", "account", "1234567890"]
        name_idx = tokens.index(name)
        bank_idx = tokens.index(bank)
        account_idx = tokens.index(account)
        
        spans = [
            (name_idx, name_idx + 1, "PERSON"),
            (bank_idx, bank_idx + 1, "ORG"),
            (account_idx, account_idx + 1, "ACCOUNT_NUMBER")
        ]

        labels = make_bio_labels(tokens, spans)

        examples.append({"tokens": tokens, "labels": labels})

    return examples

## 🔹 Save NER dataset
def write_ner_train(data):
    with open("data/ner_train.jsonl", "w") as f:
        for row in data:
            # JSONL = one JSON object per line
            f.write(json.dumps(row) + "\n")

## 🔹 Create BLEU evaluation Q/A set
# Update create_sample_data.py

def write_qa_eval():
    qa = [
        {
            "question": "What documents are required for KYC?",
            "answer": "For KYC (Know Your Customer), the required documents for individual customers are: 1) Government ID such as PAN for India, SSN for US, or Passport, 2) Address proof including utility bill, bank statement, or rental agreement, 3) Recent photograph, and 4) Bank account details."
        },
        {
            "question": "What transactions should be flagged for AML monitoring?",
            "answer": "For AML (Anti-Money Laundering) monitoring, the following should be flagged: unusually large transactions, rapid movement of funds, and suspicious counterparties."
        }
    ]

    with open("data/qa_eval.json", "w") as f:
        json.dump(qa, f, indent=2)

## 🔹 Main execution
def main():
    ensure_dirs()
    write_docs()
    ner_data = gen_ner_examples()
    write_ner_train(ner_data)
    write_qa_eval()

if __name__ == "__main__":
    main()