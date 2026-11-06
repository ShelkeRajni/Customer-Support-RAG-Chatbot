import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_vector_store(documents: List[Document], embeddings) -> FAISS:
    """
    Creates a new FAISS vector store from the given documents and embeddings model.
    """
    try:
        logger.info(f"Building FAISS vector store from {len(documents)} document chunks...")
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("FAISS vector store successfully created.")
        return vector_store
    except Exception as e:
        logger.error(f"Error building FAISS vector store: {e}")
        raise

def save_vector_store(vector_store: FAISS, folder_path: str, index_name: str = "index") -> bool:
    """
    Saves the FAISS vector store local index to the disk.
    """
    try:
        logger.info(f"Saving FAISS index to {folder_path}...")
        os.makedirs(folder_path, exist_ok=True)
        vector_store.save_local(folder_path, index_name=index_name)
        logger.info("FAISS index successfully saved to disk.")
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS index to disk: {e}")
        return False

def load_vector_store(folder_path: str, embeddings, index_name: str = "index") -> FAISS:
    """
    Loads a FAISS vector store from disk.
    """
    try:
        logger.info(f"Loading FAISS index from {folder_path}...")
        if not os.path.exists(os.path.join(folder_path, f"{index_name}.faiss")):
            raise FileNotFoundError(f"FAISS index file {index_name}.faiss not found in {folder_path}.")
            
        vector_store = FAISS.load_local(
            folder_path, 
            embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index successfully loaded.")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise
