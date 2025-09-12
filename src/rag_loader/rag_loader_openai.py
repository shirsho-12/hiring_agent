import os
from typing import List
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from .base_rag_loader import RAGLoader


class OpenAIRAGLoader(RAGLoader):
    """Handles loading documents and creating/loading FAISS vector stores with OpenAI embeddings."""

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Recursively loads all supported files (PDF, Markdown) from a directory."""
        documents = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load_and_split())
                        self.logger.info(f"Successfully loaded and split {filename}")
                    elif filename.endswith(".md"):
                        loader = UnstructuredMarkdownLoader(file_path)
                        documents.extend(loader.load())
                        self.logger.info(f"Successfully loaded {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
        return documents

    def get_vector_store(
        self, sources_path: str, cache_path: str = "data/vector_store"
    ) -> FAISS:
        """Creates or loads a FAISS vector store from all supported documents."""
        index_path = os.path.join(cache_path, "faiss_index")

        if os.path.exists(index_path):
            self.logger.info(f"Loading cached vector store from {index_path}...")
            try:
                vector_store = FAISS.load_local(
                    index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                self.logger.info("Cached vector store loaded successfully.")
                return vector_store
            except Exception as e:
                self.logger.error(
                    f"Failed to load cached vector store: {e}. Rebuilding..."
                )

        self.logger.info(
            f"Creating new vector store from all sources in {sources_path}..."
        )
        documents = self.load_documents_from_directory(sources_path)
        if not documents:
            self.logger.warning("No documents found to create a vector store.")
            return FAISS.from_texts([""], self.embeddings)

        vector_store = FAISS.from_documents(documents, self.embeddings)
        self.logger.info("Vector store created successfully.")

        try:
            self.logger.info(f"Saving vector store to {index_path}...")
            vector_store.save_local(index_path)
            self.logger.info("Vector store saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")

        return vector_store
