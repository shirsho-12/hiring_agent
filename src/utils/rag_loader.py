import os
from typing import List
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Recursively loads all supported files (PDF, Markdown) from a directory."""
    documents = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load_and_split())
                    logger.info(f"Successfully loaded and split {filename}")
                elif filename.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                    documents.extend(loader.load())
                    logger.info(f"Successfully loaded {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
    return documents

def create_vector_store_from_sources(sources_path: str, cache_path: str = "data/vector_store") -> FAISS:
    """Creates or loads a FAISS vector store from all supported documents."""
    index_path = os.path.join(cache_path, "faiss_index")
    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_path):
        logger.info(f"Loading cached vector store from {index_path}...")
        try:
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Cached vector store loaded successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load cached vector store: {e}. Rebuilding...")

    logger.info(f"Creating new vector store from all sources in {sources_path}...")
    documents = load_documents_from_directory(sources_path)
    if not documents:
        logger.warning("No documents found to create a vector store.")
        return FAISS.from_texts([""], embeddings)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info("Vector store created successfully.")

    try:
        logger.info(f"Saving vector store to {index_path}...")
        vector_store.save_local(index_path)
        logger.info("Vector store saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")

    return vector_store
