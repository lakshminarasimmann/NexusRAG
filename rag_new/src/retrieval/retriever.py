import sys
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import Config
from src.embedding.vector_store import VectorEngine
# Import LLMClient locally to avoid circular imports if generator imports retriever
# But here we need it for query expansion.
from src.generation.generator import LLMClient

class HybridRetriever:
    def __init__(self, vector_engine: VectorEngine):
        self.vector_engine = vector_engine
        self.bm25 = None
        self.documents = [] # List of text chunks for BM25
        self.doc_metadatas = [] # Parallel list of metadatas
        self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
        self.llm_client = LLMClient()
        
    def fit_bm25(self, chunks: List[Dict]):
        """
        Initialize/Fit BM25 on the current set of chunks.
        """
        self.documents = [chunk['text'] for chunk in chunks]
        self.doc_metadatas = [chunk['metadata'] for chunk in chunks]
        tokenized_corpus = [doc.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index fitted.")

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings): Generates a fake answer to the query.
        This forces the embedding to look for an 'answer-like' vector rather than a 'question-like' vector.
        """
        prompt = f"""
        You are a helpful expert research assistant.
        Provide a hypothetical answer to the following question. 
        Focus on containing keywords and scientific terminology that would appear in a relevant paper.
        Do not say "I don't know". Hallucinate a plausible-sounding scientific answer.
        
        QUESTION: {query}
        
        HYPOTHETICAL ANSWER:
        """
        response = self.llm_client.generate(prompt).strip()
        return response

    def generate_sub_questions(self, query: str) -> List[str]:
        """
        Decomposes a complex query into simple sub-questions.
        """
        prompt = f"""
        You are a helpful expert research assistant.
        Break down the following complex question into 2-3 simple, independent sub-questions that can be searched for in a scientific database.
        
        Examples:
        Complex: "Compare RAG and Fine-tuning for medical QA."
        Sub 1: "What is RAG in medical QA?"
        Sub 2: "What is Fine-tuning in medical QA?"
        
        Complex: "{query}"
        
        SUB-QUESTIONS (one per line):
        """
        response = self.llm_client.generate(prompt)
        sub_questions = [q.strip().strip('- 123.') for q in response.split('\n') if q.strip()]
        return sub_questions[:3]

    def retrieve(self, query: str, top_k=Config.TOP_K, alpha=0.5, strategy="hyde") -> List[Dict]:
        """
        Retrieval Strategy Dispatcher.
        strategies: 'hyde' (default), 'complex' (decomposition), 'naive' (simple vector), 'hybrid' (vector+bm25)
        """
        candidates = []
        
        if strategy == "complex":
            print(f"Strategy: Sub-Question Decomposition for '{query}'")
            sub_qs = self.generate_sub_questions(query)
            print(f"Sub-questions: {sub_qs}")
            
            # Retrieve for each sub-question
            candidate_map = {}
            for sub_q in sub_qs:
                # Recursive call with simple strategy for sub-questions
                sub_results = self.vector_engine.query(sub_q, n_results=top_k * 2)
                v_docs = sub_results['documents'][0]
                v_metas = sub_results['metadatas'][0]
                v_dists = sub_results['distances'][0]
                v_ids = sub_results['ids'][0]
                
                for i in range(len(v_docs)):
                    doc_id = v_ids[i]
                    if doc_id not in candidate_map:
                        candidate_map[doc_id] = {
                            "text": v_docs[i],
                            "metadata": v_metas[i],
                            "score": v_dists[i],
                            "source": "vector"
                        }
            candidates = list(candidate_map.values())
            print(f"Found {len(candidates)} unique candidates from sub-questions.")
            
        elif strategy == "hyde":
            # 1. HyDE Generation
            print(f"Generating hypothetical answer (HyDE) for: '{query}'")
            hyde_vector_query = self.generate_hypothetical_answer(query)
            print(f"Hypothetical Answer Preview: {hyde_vector_query[:100]}...")
            
            # 2. Vector Search 
            window_size = Config.RETRIEVAL_WINDOW_SIZE
            results = self.vector_engine.query(hyde_vector_query, n_results=window_size)
            
            v_docs = results['documents'][0]
            v_metas = results['metadatas'][0]
            v_dists = results['distances'][0]
            
            for i in range(len(v_docs)):
                candidates.append({
                    "text": v_docs[i],
                    "metadata": v_metas[i],
                    "score": v_dists[i],
                    "source": "vector"
                })
        
        else: # Standard/Naive
             results = self.vector_engine.query(query, n_results=Config.RETRIEVAL_WINDOW_SIZE)
             v_docs = results['documents'][0]
             v_metas = results['metadatas'][0]
             v_dists = results['distances'][0]
             for i in range(len(v_docs)):
                 candidates.append({
                    "text": v_docs[i],
                    "metadata": v_metas[i],
                    "score": v_dists[i],
                    "source": "vector"
                 })

        # Common Re-ranking Step
        print(f"Re-ranking {len(candidates)} candidates against original query...")
        reranked = self._rerank(query, candidates)
            
        return reranked[:top_k]

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Uses Cross-Encoder to re-score query-document pairs.
        """
        if not candidates:
            return []
            
        pairs = [[query, doc['text']] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Attach new scores
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(scores[i])
            
        # Sort by rerank score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_candidates

if __name__ == "__main__":
    # Test
    pass
