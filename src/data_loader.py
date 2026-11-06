import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(directory_path: str) -> List[Document]:
    """
    Loads all PDF and TXT files from the specified directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}. Creating it.")
        os.makedirs(directory_path, exist_ok=True)
        return documents
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            continue
        try:
            if file_name.lower().endswith('.pdf'):
                logger.info(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_name.lower().endswith('.txt'):
                logger.info(f"Loading TXT: {file_path}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            else:
                logger.warning(f"Skipping unsupported file: {file_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    logger.info(f"Loaded {len(documents)} document pages/sections.")
    return documents
