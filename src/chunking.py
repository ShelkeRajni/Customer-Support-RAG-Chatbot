import logging
from typing import List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document], 
    chunk_size: int = 800, 
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits list of Documents into smaller chunks using token-based RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): List of LangChain Document objects.
        chunk_size (int): Max size of each chunk in tokens.
        chunk_overlap (int): Overlap size in tokens.
        
    Returns:
        List[Document]: List of split Document objects.
    """
    try:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
        logger.info(f"Splitting {len(documents)} document pages/sections with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Using tiktoken-based character text splitter to split on tokens safely
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="gpt-4o-mini"
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Successfully generated {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []
