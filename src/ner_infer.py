# 📁 3️⃣ ner_infer.py

# 👉 Run NER at inference time

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class FinancialNER:
    def __init__(self, model_path="models/ner_financial/final"):
        print(f"🔧 Loading NER model from {model_path}...")
        # Load trained NER model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        print("✅ NER model loaded successfully")

    def extract(self, text):
        # Tokenize with proper handling
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        
        # Get word_ids to properly align predictions with original tokens
        word_ids = inputs.word_ids()

        entities = []
        previous_word_id = None
        
        for idx, word_id in enumerate(word_ids):
            # Skip special tokens (None) and repeated subword tokens
            if word_id is None or word_id == previous_word_id:
                continue
            
            pred_id = predictions[0][idx].item()
            label = self.model.config.id2label[pred_id]
            
            if label != "O":
                entities.append({
                    "token": tokens[word_id],
                    "label": label
                })
            
            previous_word_id = word_id

        return entities
    
    def extract_grouped(self, text):
        """Extract entities and group consecutive tokens of same type"""
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        word_ids = inputs.word_ids()

        entities = []
        current_entity = None
        previous_word_id = None
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                continue
            
            pred_id = predictions[0][idx].item()
            label = self.model.config.id2label[pred_id]
            
            if label != "O":
                # Remove B- or I- prefix to get entity type
                entity_type = label.split("-")[-1] if "-" in label else label
                is_begin = label.startswith("B-")
                
                # Start new entity if it's a B- tag or type changed
                if is_begin or current_entity is None or current_entity["type"] != entity_type:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": tokens[word_id],
                        "type": entity_type,
                        "tokens": [tokens[word_id]]
                    }
                else:
                    # Continue current entity
                    current_entity["text"] += " " + tokens[word_id]
                    current_entity["tokens"].append(tokens[word_id])
            else:
                # End current entity when we hit O
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            previous_word_id = word_id
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        return entities

# Example usage
if __name__ == "__main__":
    # Initialize NER
    ner = FinancialNER()
    
    # Test samples
    test_sentences = [
        "Rahul transferred money to HDFC account 1234567890",
        "Neha sent payment to ICICI account 9988776655",
        "Aarav deposited funds in HDFC account 1234567890"
    ]
    
    print("\n" + "="*80)
    print("Testing Financial NER")
    print("="*80)
    
    for sentence in test_sentences:
        print(f"\n📝 Sentence: {sentence}")
        print("\n   Individual entities:")
        entities = ner.extract(sentence)
        for entity in entities:
            print(f"      • {entity['token']:20s} → {entity['label']}")
        
        print("\n   Grouped entities:")
        grouped = ner.extract_grouped(sentence)
        for entity in grouped:
            print(f"      • {entity['text']:25s} → {entity['type']}")
    
    print("\n" + "="*80)