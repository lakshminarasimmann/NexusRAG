import chromadb
from chromadb.utils import embedding_functions
import os
import sys
from typing import List, Dict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import Config

class VectorEngine:
    def __init__(self, collection_name=Config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=Config.DB_DIR)
        
        # Use SentenceTransformers embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL_NAME
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
    def reset_collection(self):
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_fn
            )
            print(f"Collection '{self.collection.name}' reset.")
        except Exception as e:
            print(f"Error resetting collection: {e}")

    def add_chunks(self, chunks: List[Dict]):
        """
        Adds chunks to the ChromaDB collection.
        chunks: List of specific dict format from Chunker.
        """
        if not chunks:
            return
            
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Upserted {len(chunks)} chunks to collection '{self.collection.name}'.")
        
    def query(self, query_text: str, n_results=Config.TOP_K):
        """
        Performs semantic search.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    ve = VectorEngine()
    print("Vector Engine initialized.")
