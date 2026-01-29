import os
import sys
from typing import List, Dict
import requests

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import Config

class LLMClient:
    def __init__(self, provider=Config.LLM_PROVIDER):
        self.provider = provider
        
    def generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _generate_ollama(self, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": Config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error calling Ollama: {str(e)}\nEnsure Ollama is running and model '{Config.OLLAMA_MODEL}' is pulled."

    def _generate_openai(self, prompt: str) -> str:
        # Simplified OpenAI call
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

class RAGGenerator:
    def __init__(self):
        self.llm = LLMClient()
        
    def refine_contexts(self, query: str, context_chunks: List[Dict]) -> List[Dict]:
        """
        Refines the retrieved contexts by asking the LLM to extract relevant information.
        This reduces noise and context window usage.
        """
        refined_chunks = []
        for chunk in context_chunks:
            text = chunk['text']
            prompt = f"""
            You are a helpful assistant. 
            Extract only the sentences from the following text that are directly relevant to the query: "{query}"
            If the text contains no relevant information, output "IRRELEVANT".
            
            TEXT:
            {text}
            
            RELEVANT SENTENCES:
            """
            refined_text = self.llm.generate(prompt).strip()
            
            if "IRRELEVANT" not in refined_text and len(refined_text) > 10:
                refined_chunk = chunk.copy()
                refined_chunk['text'] = refined_text
                refined_chunks.append(refined_chunk)
            else:
                 # Keep original method if refinement fails or is too aggressive (optional fallback)
                 # For now, we only keep if relevant.
                 pass
                 
        # Fallback: If refinement filters everything, return original top k
        if not refined_chunks:
            return context_chunks
            
        return refined_chunks

    def assemble_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_str = "\n\n".join([f"Source: {c['metadata'].get('title', 'Unknown')}\nText: {c['text']}" for c in context_chunks])
        
        prompt = f"""
        You are an expert academic researcher assistant. 
        Your task is to write a literature review section based ONLY on the provided context.
        
        QUERY: {query}
        
        CONTEXT:
        {context_str}
        
        INSTRUCTIONS:
        1. **Analyze**: First, analyze the retrieved context and identify the key themes, methodologies, and findings.
        2. **Synthesize**: Compare and contrast the different approaches.
        3. **Cite**: Use [Author, Year] or [Title] for every claim.
        4. **Structure**: Write a coherent IEEE-style review.
        5. **Reasoning**: Think step-by-step about how the papers relate to the query before writing.
        
        Let's think step by step.
        
        LITERATURE REVIEW:
        """
        return prompt
        
    def generate_review(self, query: str, context_chunks: List[Dict]) -> str:
        print("Refining retrieved context...")
        refined_contexts = self.refine_contexts(query, context_chunks)
        print(f"Refined {len(context_chunks)} chunks into {len(refined_contexts)} relevant segments.")
        
        prompt = self.assemble_prompt(query, refined_contexts)
        response = self.llm.generate(prompt)
        return response
