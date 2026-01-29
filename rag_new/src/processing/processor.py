import fitz  # PyMuPDF
import os
import sys
from typing import List, Dict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import Config

class PDFProcessor:
    def __init__(self):
        pass
        
    def parse_pdf(self, filepath: str) -> str:
        """Extracts text from a PDF file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class Chunker:
    def __init__(self, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Splits text into overlapping chunks.
        Returns a list of chunk dictionaries with metadata.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "id": f"{os.path.basename(metadata['filepath'])}_{i}",
                "text": chunk,
                "metadata": metadata
            })
            
        return chunked_data

if __name__ == "__main__":
    # Test script
    processor = PDFProcessor()
    chunker = Chunker()
    
    # Create a dummy PDF for testing if needed, or pass an existing one
    print("Processor and Chunker initialized.")
