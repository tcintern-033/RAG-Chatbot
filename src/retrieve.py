import os
from langchain_chroma import Chroma
from src.config import CHROMA_PATH, COLLECTION_NAME, DEFAULT_TOP_K
from src.embeddings import get_embeddings

def _get_vectorstore():
    """Helper function to load the ChromaDB vector store."""
    if not os.path.exists(CHROMA_PATH):
        print("Error: ChromaDB database not found. Please index documents first.")
        return None

    try:
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"Error: {e}")
        return None

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore

def search_documents():
    """
    Retrieves the most relevant document chunks based on a user query
    using the retriever interface.
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": DEFAULT_TOP_K}
    )

    print("\n============================================================")
    print("RETRIEVAL PIPELINE")
    print("============================================================")
    
    query = input("\nEnter your question: ").strip()
    if not query:
        print("Error: Empty user queries are not allowed.")
        return

    print("\n============================================================")
    print("RETRIEVED CONTEXT")
    print("============================================================")

    chunks = retriever.invoke(query)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id', 'Unknown')}")
        print("\nContent:")
        print(chunk.page_content)

    print("\n============================================================")
    print(f"Retrieved {len(chunks)} chunks")
    print("============================================================\n")

def compare_top_k():
    """
    Allows the user to select Top-K and tests retrieving chunks.
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return

    print("\nChoose Top-K:")
    print("1. 2")
    print("2. 4")
    print("3. 6")
    choice = input("Enter choice (1/2/3): ").strip()
    
    k_map = {"1": 2, "2": 4, "3": 6}
    if choice not in k_map:
        print("Error: Invalid choice.")
        return
    
    k = k_map[choice]
    
    query = input("\nEnter your question: ").strip()
    if not query:
        print("Error: Empty user queries are not allowed.")
        return
        
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    chunks = retriever.invoke(query)
    
    print("\n============================================================")
    print(f"Top-K: {k}")
    print(f"Retrieved chunks: {len(chunks)}")
    print("============================================================")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)

def demonstrate_similarity_search(query: str, k: int = 4):
    """
    Demonstrates direct similarity search on the vectorstore.
    This bypasses the retriever interface and queries ChromaDB directly.
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return
    
    print("\n[Direct Similarity Search]")
    chunks = vectorstore.similarity_search(query, k=k)
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk.page_content[:100] + "...")
