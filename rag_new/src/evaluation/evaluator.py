from typing import List, Dict
import numpy as np

class Evaluator:
    @staticmethod
    def hit_rate(retrieved_chunks_list: List[List[Dict]], relevant_ids_list: List[List[str]]) -> float:
        """
        Calculates Hit Rate: Proportion of queries where at least one relevant document is retrieved in top-k.
        relevant_ids_list: List of relevant document IDs (filenames or specific chunk IDs) for each query.
        """
        hits = 0
        for retrieved, relevant in zip(retrieved_chunks_list, relevant_ids_list):
            # Check if any retrieved doc matches any relevant doc ID
            # Assuming 'id' or 'filepath' in metadata can be used for matching
            retrieved_ids = [r['metadata'].get('title', '') for r in retrieved]
            if any(rel in retrieved_ids for rel in relevant):
                hits += 1
        return hits / len(retrieved_chunks_list) if retrieved_chunks_list else 0.0

    @staticmethod
    def mrr(retrieved_chunks_list: List[List[Dict]], relevant_ids_list: List[List[str]]) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR).
        """
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_chunks_list, relevant_ids_list):
            retrieved_ids = [r['metadata'].get('title', '') for r in retrieved]
            rank = 0
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant:
                    rank = i + 1
                    break
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_faithfulness(query: str, response: str, context: List[Dict], llm_client) -> Dict:
        """
        G-Eval Faithfulness: Calculates a score (1-5) with reasoning.
        Is the answer derived from the context?
        """
        context_text = "\n".join([c['text'] for c in context])
        prompt = f"""
        You are an impartial judge evaluating a RAG system.
        Task: Evaluate the Faithfulness of the ANSWER to the provided CONTEXT.
        
        CONTEXT:
        {context_text}
        
        ANSWER:
        {response}
        
        INSTRUCTIONS:
        1. Read the Context and the Answer.
        2. Identify any claims in the Answer that contradict or are unsupported by the Context.
        3. Think step by step.
        4. Output a score from 1 to 5, where:
           - 1: Entirely hallucinated or contradicts context.
           - 3: Mostly supported but has minor unverified details.
           - 5: Fully supported by context.
        
        OUTPUT FORMAT:
        Reasoning: [Your step-by-step reasoning]
        Score: [1-5]
        """
        eval_output = llm_client.generate(prompt).strip()
        return Evaluator._parse_geval_output(eval_output)

    @staticmethod
    def evaluate_relevance(query: str, response: str, llm_client) -> Dict:
        """
        G-Eval Relevance: Calculates a score (1-5) with reasoning.
        Does the answer address the query?
        """
        prompt = f"""
        You are an impartial judge evaluating a RAG system.
        Task: Evaluate the Relevance of the ANSWER to the QUERY.
        
        QUERY:
        {query}
        
        ANSWER:
        {response}
        
        INSTRUCTIONS:
        1. Analyze the user Query intent.
        2. Determine if the Answer directly addresses the intent.
        3. Think step by step.
        4. Output a score from 1 to 5, where:
           - 1: Completely irrelevant or off-topic.
           - 3: Partially relevant but misses key aspects.
           - 5: Highly relevant and comprehensive.
        
        OUTPUT FORMAT:
        Reasoning: [Your step-by-step reasoning]
        Score: [1-5]
        """
        eval_output = llm_client.generate(prompt).strip()
        return Evaluator._parse_geval_output(eval_output)

    @staticmethod
    def _parse_geval_output(text: str) -> Dict:
        import re
        score_match = re.search(r"Score:\s*(\d+)", text)
        score = int(score_match.group(1)) if score_match else 0
        return {"reasoning": text, "score": score}
