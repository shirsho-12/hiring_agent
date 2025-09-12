from abc import ABC, abstractmethod
from typing import List
from langchain.schema.document import Document
from src.utils.logger import get_logger


class RAGLoader(ABC):
    """
    Abstract base class for Retrieval-Augmented Generation (RAG) loaders.
    Child classes must implement document loading and vector store creation.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load and split documents from a directory.

        Args:
            directory_path: Path to the directory containing documents.

        Returns:
            List of loaded and split Document objects.
        """
        pass

    @abstractmethod
    def get_vector_store(self, sources_path: str, cache_path: str):
        """
        Create or load a vector store from source documents.

        Args:
            sources_path: Path to source documents.
            cache_path: Path to store or load the vector store.

        Returns:
            A vector store object.
        """
        pass
