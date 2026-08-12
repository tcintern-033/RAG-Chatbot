import os
import time
try:
    from langchain_community.document_loaders import TextLoader
except ImportError:
    TextLoader = None
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import CHROMA_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from src.embeddings import get_embeddings

def index_documents():
    """
    Loads a document, splits it into chunks, generates embeddings,
    and stores them in ChromaDB using Google Gemini Embeddings.
    Batches chunk requests to stay within Google's free-tier rate limits.
    """
    print("Loading document...")
    if not os.path.exists("data/sample.txt"):
        print("Error: data/sample.txt not found.")
        return

    if TextLoader is not None:
        loader = TextLoader("data/sample.txt", encoding="utf-8")
        documents = loader.load()
    else:
        with open("data/sample.txt", "r", encoding="utf-8") as f:
            content = f.read()
        documents = [Document(page_content=content, metadata={"source": "sample.txt"})]
    print("Document loaded successfully.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "source": "sample.txt",
            "chunk_id": i
        }

    print(f"Number of chunks created: {len(chunks)}")
    print("Generating embeddings...")

    try:
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Process in batches of 40 to avoid Google API 429 Rate Limits (100 req/min limit)
    batch_size = 40
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Indexing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=CHROMA_PATH,
                        collection_name=COLLECTION_NAME
                    )
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if attempt < max_retries - 1 and "429" in str(e):
                    print(f"[Rate Limit] Google API 100 req/min quota reached. Pausing 15 seconds before retry...")
                    time.sleep(15)
                else:
                    raise e
        
        # Pause if more batches remain to respect rate limits
        if i + batch_size < len(chunks):
            time.sleep(2)
    
    print("Documents successfully stored in ChromaDB.")

if __name__ == "__main__":
    index_documents()
