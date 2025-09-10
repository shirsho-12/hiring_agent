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

def create_vector_store_from_sources(directory_path: str) -> FAISS:
    """Creates a FAISS vector store from all supported documents in a directory."""
    logger.info(f"Creating vector store from all sources in {directory_path}...")
    documents = load_documents_from_directory(directory_path)
    if not documents:
        logger.warning("No documents found to create a vector store.")
        # Return an empty vector store if no documents are found
        return FAISS.from_texts([""], OpenAIEmbeddings())
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info("Vector store created successfully from all sources.")
    return vector_store
