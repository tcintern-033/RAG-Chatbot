import sys
from src.rag import answer_question, compare_top_k_rag

def ask_question_loop():
    """
    Interactive chatbot loop allowing continuous Q&A.
    """
    print("\n============================================================")
    print("RAG CHATBOT")
    print("============================================================")
    print("Your documents are ready.")
    print("Type 'exit' or 'back' to return to the main menu.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to main menu...")
            break

        if not user_input:
            print("Please enter a question.\n")
            continue

        if user_input.lower() in ["exit", "back", "quit"]:
            print("\nReturning to main menu...")
            break

        answer, docs = answer_question(user_input)

        print("\n============================================================")
        print("AI RESPONSE")
        print("============================================================")
        print(answer)

        if docs:
            print("\n============================================================")
            print("RETRIEVED CONTEXT")
            print("============================================================")
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "sample.txt")
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                print(f"\n--- Chunk {i} ---")
                print(f"Source: {source}")
                print(f"Chunk ID: {chunk_id}")
                print("\nContent:")
                print(doc.page_content)
            print("\n============================================================\n")
        else:
            print("\n============================================================\n")

def main():
    while True:
        print("\n==================================================")
        print("RAG CHATBOT")
        print("==================================================")
        print("1. Ask Question")
        print("2. Compare Top-K")
        print("3. Exit")

        try:
            choice = input("\nChoose an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if choice == "1":
            ask_question_loop()
        elif choice == "2":
            question = input("\nEnter your question: ").strip()
            if not question:
                print("Please enter a question.")
            else:
                compare_top_k_rag(question)
        elif choice == "3" or choice.lower() == "exit":
            print("Exiting...")
            sys.exit(0)
        else:
            print("Error: Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
