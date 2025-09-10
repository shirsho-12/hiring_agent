import os
from src.utils.rag_loader_hf import RAGLoader
from src.utils.logger import get_logger

def rebuild_vector_store():
    logger = get_logger("rebuild_vector_store")
    
    # Define paths
    sources_path = "data/rag_sources"
    cache_path = "data/vector_store"
    
    # Initialize RAG loader
    try:
        logger.info("Initializing RAG loader...")
        rag_loader = RAGLoader(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Force rebuild by removing existing index
        import shutil
        if os.path.exists(cache_path):
            logger.info(f"Removing existing vector store at {cache_path}")
            shutil.rmtree(cache_path)
        
        # Create directory if it doesn't exist
        os.makedirs(cache_path, exist_ok=True)
        
        # Rebuild vector store
        logger.info(f"Rebuilding vector store from {sources_path}")
        vector_store = rag_loader.get_vector_store(sources_path, cache_path)
        
        if vector_store is None:
            logger.error("Failed to rebuild vector store")
            return False
            
        # Verify the vector store
        if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
            logger.info(f"Successfully rebuilt vector store with {vector_store.index.ntotal} vectors")
        else:
            logger.warning("Rebuilt vector store, but couldn't verify vector count")
            
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if rebuild_vector_store():
        print("Vector store rebuilt successfully!")
    else:
        print("Failed to rebuild vector store. Check the logs for details.")
