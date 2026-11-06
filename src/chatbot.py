import os
import logging
from typing import Any
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chatbot_chain(retriever, llm_provider: str = "gemini") -> Any:
    """
    Constructs and returns a conversational RAG pipeline using Google Gemini.
    
    Args:
        retriever: LangChain retriever object connected to FAISS vector store.
        llm_provider (str): LLM provider. Currently supports "gemini".
        
    Returns:
        Chain: Compiled LangChain RAG pipeline.
    """
    try:
        # 1. Initialize Google Gemini LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Google API Key. Please enter it in the sidebar."
            )
        logger.info("Initializing Google Gemini LLM (gemini-3.5-flash)...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )

        # 2. History-aware retriever
        # Rewrites user question into a standalone query using conversation history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 3. Answer generation chain
        # Strictly bounds answers to retrieved context to prevent hallucination
        system_prompt = (
            "You are SupportGPT, an advanced customer support AI assistant.\n"
            "Answer the user question using ONLY the provided context below.\n"
            "If the answer is not found in the context, respond with:\n"
            "'I am sorry, I could not find that information in our documents. "
            "Is there anything else I can help you with?'\n"
            "Do not make up or assume any information not in the context.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # 4. Combine into final RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info("Gemini chatbot RAG chain successfully created.")
        return rag_chain

    except Exception as e:
        logger.error(f"Error creating chatbot chain: {e}")
        raise

