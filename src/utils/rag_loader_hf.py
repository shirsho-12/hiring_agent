import os
from typing import List, Optional
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from .hf_embeddings import HFEmbeddingsWrapper
from langchain_community.vectorstores import FAISS

logger = get_logger(__name__)


class RAGLoader:
    """Handles loading documents and creating/loading FAISS vector stores with HuggingFace embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the RAG loader.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """
        self.embeddings_wrapper = HFEmbeddingsWrapper(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load and split documents from a directory."""
        documents = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        docs = self.text_splitter.split_documents(docs)
                        documents.extend(docs)
                        logger.info(
                            f"Successfully loaded and split {filename} into {len(docs)} chunks"
                        )
                    elif filename.endswith(".md"):
                        loader = UnstructuredMarkdownLoader(file_path)
                        docs = loader.load()
                        docs = self.text_splitter.split_documents(docs)
                        documents.extend(docs)
                        logger.info(
                            f"Successfully loaded and split {filename} into {len(docs)} chunks"
                        )
                    elif filename.endswith(".txt"):
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        docs = self.text_splitter.split_documents(docs)
                        documents.extend(docs)
                        logger.info(
                            f"Successfully loaded and split {filename} into {len(docs)} chunks"
                        )
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
        return documents

    def get_vector_store(
        self, sources_path: str, cache_path: str = "data/vector_store"
    ) -> FAISS:
        """
        Get or create a FAISS vector store with HuggingFace embeddings.

        Args:
            sources_path: Path to the directory containing source documents
            cache_path: Path to cache the vector store

        Returns:
            FAISS vector store
        """
        index_path = os.path.join(cache_path, "faiss_index")

        # Try to load existing vector store
        if os.path.exists(index_path):
            try:
                logger.info(f"Loading cached vector store from {index_path}...")
                return self.embeddings_wrapper.load_vector_store(index_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load cached vector store: {e}. Rebuilding..."
                )

        # Create new vector store
        logger.info(f"Creating new vector store from sources in {sources_path}...")
        documents = self.load_documents_from_directory(sources_path)

        if not documents:
            logger.warning("No documents found to create a vector store.")
            return None

        vector_store = self.embeddings_wrapper.create_vector_store(documents)

        # Save the vector store for future use
        self.embeddings_wrapper.save_vector_store(vector_store, index_path)
        logger.info(f"Vector store created and saved to {index_path}")

        return vector_store
