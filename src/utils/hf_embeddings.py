from typing import List, Optional
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HFEmbeddingsWrapper:
    """Wrapper for HuggingFace embeddings with FAISS vector store."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the HuggingFace embeddings wrapper.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings
            model_kwargs: Additional arguments to pass to the model
            encode_kwargs: Additional arguments to pass to the encode method
        """
        if model_kwargs is None:
            model_kwargs = {"device": "cpu"}
        if encode_kwargs is None:
            encode_kwargs = {"normalize_embeddings": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a FAISS vector store from documents."""
        logger.info("Creating FAISS vector store with HuggingFace embeddings...")
        return FAISS.from_documents(documents, self.embeddings)

    def load_vector_store(self, path: str) -> FAISS:
        """Load a FAISS vector store from disk."""
        logger.info(f"Loading FAISS vector store from {path}...")
        return FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    def save_vector_store(self, vector_store: FAISS, path: str):
        """Save a FAISS vector store to disk."""
        logger.info(f"Saving FAISS vector store to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vector_store.save_local(path)
