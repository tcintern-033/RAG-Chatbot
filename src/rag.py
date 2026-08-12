import os
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import CHROMA_PATH, COLLECTION_NAME, DEFAULT_TOP_K
from src.embeddings import get_embeddings

def _get_vectorstore():
    """Helper function to load the persistent ChromaDB vector store."""
    if not os.path.exists(CHROMA_PATH):
        print("\nError: ChromaDB database not found. Please run Option 1 to index documents first.")
        return None

    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"\nError initializing embeddings: {e}")
        return None

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def get_retriever(k: int = DEFAULT_TOP_K):
    """
    Creates and returns a LangChain Retriever interface from ChromaDB.
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return None
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def get_llm():
    """
    Initializes and returns the Google Gemini Chat Model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY is missing or invalid in .env")

    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

def answer_question(question: str, k: int = DEFAULT_TOP_K) -> Tuple[str, List[Document]]:
    """
    Complete RAG function:
    1. Validates the user question.
    2. Retrieves Top-K relevant document chunks from ChromaDB.
    3. Builds the injected context string.
    4. Formats the grounded prompt template.
    5. Sends the prompt to Google Gemini LLM.
    6. Returns (AI answer, retrieved documents).
    """
    cleaned_question = question.strip() if question else ""
    if not cleaned_question:
        return "Please enter a question.", []

    retriever = get_retriever(k=k)
    if not retriever:
        return "Error: Database is not indexed. Please index documents first.", []

    try:
        documents = retriever.invoke(cleaned_question)
    except Exception as e:
        return f"Error retrieving context from vector store: {e}", []

    if not documents:
        return "No relevant documents were found for this question.", []

    # Build Context String (Context Injection)
    context = "\n\n".join(doc.page_content for doc in documents)

    # Factual Grounded Prompt Template
    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents to answer that question."

Do not invent facts.
Do not use outside knowledge.
Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:"""
    )

    try:
        llm = get_llm()
        formatted_prompt = prompt_template.format(context=context, question=cleaned_question)
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, documents
    except Exception as e:
        return f"Error generating answer with Gemini LLM: {e}", documents

def compare_top_k_rag(question: str):
    """
    Runs the RAG pipeline for K=2, K=4, and K=6 to compare retrieval and generation quality.
    """
    cleaned_question = question.strip() if question else ""
    if not cleaned_question:
        print("\nPlease enter a question.")
        return

    top_k_values = [2, 4, 6]
    print(f"\n============================================================")
    print(f"TOP-K COMPARISON FOR QUERY: '{cleaned_question}'")
    print(f"============================================================")

    for k in top_k_values:
        print(f"\n============================================================")
        print(f"                     K = {k}")
        print(f"============================================================")
        
        answer, docs = answer_question(cleaned_question, k=k)
        
        print("\n--- RETRIEVED CONTEXT ---")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "sample.txt")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            print(f"\n[Chunk {i}] (Source: {source} | Chunk ID: {chunk_id})")
            print(doc.page_content)
        
        print("\n--- AI RESPONSE ---")
        print(answer)
        print("============================================================")
