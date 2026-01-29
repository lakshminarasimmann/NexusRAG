import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PAPERS_DIR = os.path.join(DATA_DIR, "papers")
    DB_DIR = os.path.join(DATA_DIR, "vector_db")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Ingestion
    MAX_PAPERS = 5
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Easy to run locally
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Vector DB
    COLLECTION_NAME = "rag_papers"
    
    # Retrieval
    TOP_K = 5
    RETRIEVAL_WINDOW_SIZE = 50 # Fetch more candidates for re-ranking
    
    # Generation
    LLM_PROVIDER = "ollama" # or "openai"
    OLLAMA_MODEL = "mistral" # Make sure you have this pulled: `ollama pull mistral`
    OPENAI_MODEL = "gpt-4o-mini"
    
    # For OpenAI, ensure OPENAI_API_KEY is in env vars
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.PAPERS_DIR, exist_ok=True)
        os.makedirs(Config.DB_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

Config.ensure_dirs()
