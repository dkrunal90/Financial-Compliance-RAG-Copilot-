# 📁 7️⃣ chat_cli.py

# 👉 Interactive CLI demo for Financial Compliance Copilot

from rag_chain import ComplianceRAG
from ner_infer import FinancialNER
import sys

def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   💼 Financial Compliance Copilot                          ║
║   Powered by RAG + Fine-tuned NER                          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_help():
    help_text = """
Available Commands:
  ask <question>     - Ask a compliance question (RAG)
  ner <text>         - Extract entities from text
  help               - Show this help message
  exit / quit        - Exit the application

Examples:
  ask What documents are needed for KYC?
  ner Rahul transferred money to HDFC account 1234567890
    """
    print(help_text)

def main():
    print_banner()
    
    print("🔧 Initializing systems...")
    try:
        rag = ComplianceRAG()
        ner = FinancialNER()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("\nMake sure you've run:")
        print("  1. python src/create_sample_data.py")
        print("  2. python src/ner_train.py")
        print("  3. python src/ingest_index.py")
        sys.exit(1)
    
    print("✅ All systems ready!\n")
    print_help()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            elif command == "help":
                print_help()
            
            elif command == "ask":
                if len(parts) < 2:
                    print("❌ Please provide a question. Example: ask What is KYC?")
                    continue
                
                question = parts[1]
                print(f"\n🔍 Searching knowledge base...")
                answer = rag.answer(question, verbose=False)
                print(f"\n🤖 Answer:\n{answer}")
            
            elif command == "ner":
                if len(parts) < 2:
                    print("❌ Please provide text. Example: ner Rahul sent money to HDFC")
                    continue
                
                text = parts[1]
                print(f"\n🔍 Extracting entities...")
                entities = ner.extract_grouped(text)
                
                if entities:
                    print("\n📋 Extracted Entities:")
                    for entity in entities:
                        print(f"   • {entity['text']:25s} → {entity['type']}")
                else:
                    print("   No entities found")
            
            else:
                # Assume it's a question if no command specified
                print(f"\n🔍 Searching knowledge base...")
                answer = rag.answer(user_input, verbose=False)
                print(f"\n🤖 Answer:\n{answer}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()