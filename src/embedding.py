import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings_model(provider: str = "gemini"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Enter your Gemini API key in the sidebar.")
    
    logger.info("Initializing GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
    )



