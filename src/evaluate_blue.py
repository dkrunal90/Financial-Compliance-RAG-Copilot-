# 📁 6️⃣ evaluate_bleu.py

# 👉 Evaluate RAG quality with BLEU score

import json
from pathlib import Path
import os
from sacrebleu.metrics import BLEU
from rag_chain import ComplianceRAG

def load_qa_eval():
    """Load gold Q/A pairs for evaluation"""
    qa_path = Path("data/qa_eval.json")
    
    if not qa_path.exists():
        print(f"❌ Evaluation file not found: {qa_path.absolute()}")
        print("Run: python src/create_sample_data.py")
        return None
    
    with open(qa_path) as f:
        return json.load(f)

def evaluate_rag():
    """Evaluate RAG system using BLEU score"""
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("="*80)
    print("RAG System Evaluation (BLEU Score)")
    print("="*80)
    
    # Load evaluation data
    print("\n📊 Loading evaluation data...")
    qa_pairs = load_qa_eval()
    
    if not qa_pairs:
        return None
    
    print(f"   ✅ Loaded {len(qa_pairs)} Q/A pairs")
    
    # Initialize RAG system
    print("\n🤖 Initializing RAG system...")
    try:
        rag = ComplianceRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return None
    
    # Initialize BLEU metric
    bleu = BLEU()
    
    predictions = []
    references = []
    
    print("\n" + "="*80)
    print("Running Evaluation (Concise Mode for BLEU)")
    print("="*80)
    
    # Evaluate each Q/A pair
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        gold_answer = qa["answer"]
        
        print(f"\n[{i}/{len(qa_pairs)}] Question: {question}")
        
        # Get RAG prediction in CONCISE mode
        try:
            predicted_answer = rag.answer(question, verbose=False, concise=True)
        except Exception as e:
            print(f"   ❌ Error generating answer: {e}")
            predicted_answer = ""
        
        print(f"   📌 Gold:      {gold_answer}")
        print(f"   🤖 Predicted: {predicted_answer}")
        
        predictions.append(predicted_answer)
        references.append([gold_answer])
    
    # Rest of the code remains the same...  # BLEU expects list of references
    
    # Calculate BLEU score
    print("\n" + "="*80)
    print("Computing BLEU Score...")
    print("="*80)
    
    score = bleu.corpus_score(predictions, references)
    
    print(f"\n📊 Results:")
    print(f"   BLEU Score: {score.score:.2f}")
    print(f"   Precision:")
    print(f"     - 1-gram: {score.precisions[0]:.2f}")
    print(f"     - 2-gram: {score.precisions[1]:.2f}")
    print(f"     - 3-gram: {score.precisions[2]:.2f}")
    print(f"     - 4-gram: {score.precisions[3]:.2f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if score.score >= 80:
        print("   🎉 Excellent - Very high quality answers")
    elif score.score >= 60:
        print("   ✅ Good - Answers are generally accurate")
    elif score.score >= 40:
        print("   ⚠️  Fair - Answers need improvement")
    else:
        print("   ❌ Poor - Significant improvements needed")
    
    print("\n" + "="*80)
    
    # Save detailed results
    results = {
        "bleu_score": score.score,
        "precisions": {
            "1-gram": score.precisions[0],
            "2-gram": score.precisions[1],
            "3-gram": score.precisions[2],
            "4-gram": score.precisions[3]
        },
        "num_questions": len(qa_pairs),
        "predictions": [
            {
                "question": qa["question"],
                "gold": qa["answer"],
                "predicted": pred
            }
            for qa, pred in zip(qa_pairs, predictions)
        ]
    }
    
    # Save results to file
    results_path = Path("evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_path}")
    
    return score.score

if __name__ == "__main__":
    score = evaluate_rag()
    
    if score is None:
        print("\n❌ Evaluation failed. Make sure:")
        print("  1. You've run: python src/create_sample_data.py")
        print("  2. You've run: python src/ingest_index.py")
        print("  3. Ollama is running with llama3.2")
        exit(1)
    else:
        print(f"\n✅ Evaluation complete! Final BLEU Score: {score:.2f}")