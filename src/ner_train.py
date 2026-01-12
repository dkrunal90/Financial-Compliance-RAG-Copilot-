## 📁 2️⃣ ner_train.py

## 👉 Fine-tunes BERT for Financial NER

# Load and process datasets
from datasets import Dataset

# HuggingFace Transformer tools
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification  # ← Add this
)

import json

# 🔹 Model selection
# DistilBERT is small and fast
MODEL_NAME = "distilbert-base-cased"

# 🔹 Load training data
def load_data():
    rows = []
    with open("/Users/krunaldesai/Desktop/RAG Copilot/data/ner_train.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# 🔹 Tokenization + label alignment
def tokenize_align(examples, tokenizer, label2id):
    # Convert words to subword tokens
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,  # Don't pad here - let DataCollator handle it
        max_length=128
    )

    all_labels = []
    
    # Process each example in the batch
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignored by loss (special tokens)
            else:
                # Map the label for this word
                label_ids.append(label2id[examples["labels"][i][word_id]])
        
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

# 🔹 Training logic
def main():
    print("📚 Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} training examples")

    # Get unique labels
    labels = sorted({l for row in data for l in row["labels"]})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"🏷️  Found {len(labels)} unique labels: {labels}")

    # Create dataset
    dataset = Dataset.from_list(data)

    # Load tokenizer
    print(f"🔧 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize dataset
    print("⚙️  Tokenizing dataset...")
    dataset = dataset.map(
        lambda x: tokenize_align(x, tokenizer, label2id), 
        batched=True,
        remove_columns=dataset.column_names
    )

    # Load model
    print(f"🤖 Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    # Training arguments
    args = TrainingArguments(
        output_dir="models/ner_financial",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        push_to_hub=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator  # ← This is critical!
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving model...")
    model.save_pretrained("models/ner_financial/final")
    tokenizer.save_pretrained("models/ner_financial/final")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()