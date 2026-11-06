import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_retriever(
    vector_store: FAISS, 
    search_type: str = "similarity", 
    k: int = 4
):
    """
    Returns a LangChain retriever from the FAISS vector store.
    
    Args:
        vector_store (FAISS): Loaded FAISS vector store.
        search_type (str): Retrieval algorithm, e.g., 'similarity' or 'mmr'.
        k (int): Number of documents to retrieve.
        
    Returns:
        VectorStoreRetriever: Configured retriever object.
    """
    try:
        logger.info(f"Configuring retriever: search_type={search_type}, k={k}")
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return retriever
    except Exception as e:
        logger.error(f"Error configuring retriever: {e}")
        raise

def similarity_search(
    vector_store: FAISS, 
    query: str, 
    k: int = 4
) -> List[Document]:
    """
    Performs raw similarity search directly on the vector store.
    
    Args:
        vector_store (FAISS): Loaded FAISS vector store.
        query (str): The search query.
        k (int): Number of documents to retrieve.
        
    Returns:
        List[Document]: List of matching documents.
    """
    try:
        logger.info(f"Performing similarity search for query: '{query}'")
        docs = vector_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        logger.error(f"Error executing similarity search: {e}")
        return []
