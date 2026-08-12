import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

def get_embeddings():
    """
    Returns the Google Gemini embedding model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY was not found or is a placeholder. Please create a .env file and add your API key.")
    
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")