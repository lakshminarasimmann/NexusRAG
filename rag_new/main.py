import argparse
import sys
import os
import glob

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config
from src.ingestion.ingestor import ArxivIngestor
from src.processing.processor import PDFProcessor, Chunker
from src.embedding.vector_store import VectorEngine
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import RAGGenerator

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Search and download papers")
    ingest_parser.add_argument("--query", required=True, help="Search query for arXiv")
    ingest_parser.add_argument("--max", type=int, default=Config.MAX_PAPERS, help="Max papers to download")
    
    # Index Command
    index_parser = subparsers.add_parser("index", help="Parse and index downloaded papers")
    
    # Retrieve Command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant chunks")
    retrieve_parser.add_argument("--query", required=True, help="Query for retrieval")
    retrieve_parser.add_argument("--k", type=int, default=Config.TOP_K, help="Number of chunks")
    
    # Generate Command
    generate_parser = subparsers.add_parser("generate", help="Generate Literature Review")
    generate_parser.add_argument("--topic", required=True, help="Topic for review")
    
    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--query", required=True, help="Test query")
    
    # Run All Command
    run_parser = subparsers.add_parser("run_all", help="Run full pipeline (Ingest -> Index -> Generate -> Eval)")
    run_parser.add_argument("--query", required=True, help="Topic/Query for the pipeline")
    run_parser.add_argument("--max", type=int, default=Config.MAX_PAPERS, help="Max papers")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        print("--- Mode: Ingestion ---")
        ingestor = ArxivIngestor(max_results=args.max)
        papers = ingestor.search_and_download(args.query)
        print(f"Downloaded {len(papers)} papers.")
        
    elif args.command == "index":
        print("--- Mode: Indexing ---")
        processor = PDFProcessor()
        chunker = Chunker()
        ve = VectorEngine()
        
        # Reset DB to ensure sync with current papers
        ve.reset_collection()
        
        pdf_files = glob.glob(os.path.join(Config.PAPERS_DIR, "*.pdf"))
        print(f"Found {len(pdf_files)} PDFs to process.")
        
        for filepath in pdf_files:
            try:
                print(f"Processing: {os.path.basename(filepath)}")
                text = processor.parse_pdf(filepath)
                # Simple metadata
                metadata = {"filepath": filepath, "title": os.path.basename(filepath)}
                chunks = chunker.chunk_text(text, metadata)
                ve.add_chunks(chunks)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        print("Indexing Complete.")
        
    elif args.command == "retrieve":
        print("--- Mode: Retrieval ---")
        ve = VectorEngine()
        retriever = HybridRetriever(ve)
        
        results = retriever.retrieve(args.query, top_k=args.k)
        for i, res in enumerate(results):
            print(f"\n[Result {i+1}] (Score: {res.get('rerank_score', 'N/A')})")
            print(f"Source: {res['metadata'].get('title')}")
            print(f"Text Snippet: {res['text'][:200]}...")
            
    elif args.command == "generate":
        print("--- Mode: Generation ---")
        ve = VectorEngine()
        retriever = HybridRetriever(ve)
        rag = RAGGenerator()
        
        print(f"Retrieving context for topic: {args.topic}...")
        context = retriever.retrieve(args.topic, top_k=5)
        
        print("Generating Review with LLM...")
        review = rag.generate_review(args.topic, context)
        
        output_path = os.path.join(Config.OUTPUT_DIR, "literature_review.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Literature Review: {args.topic}\\n\\n")
            f.write(review)
            
        print(f"Review saved to: {output_path}")
        print("\n--- Preview ---")
        print(review[:500] + "...")
        
    elif args.command == "evaluate":
        print("--- Mode: Evaluation ---")
        from src.evaluation.evaluator import Evaluator
        from src.generation.generator import LLMClient
        ve = VectorEngine()
        retriever = HybridRetriever(ve)
        llm = LLMClient()
        
        print(f"Evaluating Query: {args.query}")
        
        # 1. Retrieve
        context = retriever.retrieve(args.query, top_k=5)
        print(f"Retrieved {len(context)} chunks.")
        
        # 2. Generate
        rag = RAGGenerator()
        response = rag.generate_review(args.query, context)
        print("Generated response.")
        
        print(f"\nRelevance Score:    {rel_result['score']}/5")
        print(f"Reasoning: {rel_result['reasoning'][:200]}...")
        
def run_interactive_pipeline(topic: str, retrieval_strategy: str):
    """Executes the pipeline with the chosen parameters."""
    print(f"\n--- Running Pipeline: {topic} (Strategy: {retrieval_strategy}) ---")
    
    # 1. Ingest
    print(f"\n[1/4] Ingesting papers...")
    ingestor = ArxivIngestor(max_results=3)
    papers = ingestor.search_and_download(topic)
    
    # 2. Index
    print(f"\n[2/4] Indexing...")
    processor = PDFProcessor()
    chunker = Chunker()
    ve = VectorEngine()
    ve.reset_collection()
    
    pdf_files = glob.glob(os.path.join(Config.PAPERS_DIR, "*.pdf"))
    for filepath in pdf_files:
        try:
            text = processor.parse_pdf(filepath)
            metadata = {"filepath": filepath, "title": os.path.basename(filepath)}
            chunks = chunker.chunk_text(text, metadata)
            ve.add_chunks(chunks)
        except Exception as e:
            print(f"Index Error: {e}")
            
    # 3. Retrieve & Generate
    print(f"\n[3/4] Retrieving & Generating...")
    from src.evaluation.evaluator import Evaluator
    from src.generation.generator import LLMClient
    
    retriever = HybridRetriever(ve)
    rag = RAGGenerator()
    llm = LLMClient()
    
    context = retriever.retrieve(topic, top_k=5, strategy=retrieval_strategy)
    
    print("Refining context...")
    refined_context = rag.refine_contexts(topic, context)
    review = rag.generate_review(topic, refined_context)
    
    output_path = os.path.join(Config.OUTPUT_DIR, f"review_{topic.replace(' ', '_')}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n**Strategy:** {retrieval_strategy}\n\n{review}")
    print(f"Review saved: {output_path}")
    
    # 4. Evaluate
    print(f"\n[4/4] Evaluating...")
    faith_result = Evaluator.evaluate_faithfulness(topic, review, context, llm)
    rel_result = Evaluator.evaluate_relevance(topic, review, llm)
    
    print("\n=== Final Quality Report ===")
    print(f"Faithfulness: {faith_result['score']}/5")
    print(f"Relevance:    {rel_result['score']}/5")
    print("============================")
    input("\nPress Enter to continue...")

def print_menu():
    print("\n" + "="*40)
    print("      ADVANCED RAG SYSTEM v2.0")
    print("="*40)
    print("1. Run RAG Pipeline (Auto-Ingest + Generate)")
    print("2. Evaluation Mode (Test Scoring)")
    print("3. Exit")
    print("="*40)

def main():
    while True:
        print_menu()
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            topic = input("\nEnter Research Topic: ").strip()
            if not topic: continue
            
            print("\nSelect Retrieval Strategy:")
            print("1. HyDE (Best for general queries)")
            print("2. Decomposition (Best for complex/comparison queries)")
            print("3. Standard (Fastest)")
            strat_choice = input("Strategy (1-3): ").strip()
            
            strategy = "hyde"
            if strat_choice == "2": strategy = "complex"
            elif strat_choice == "3": strategy = "standard"
            
            run_interactive_pipeline(topic, strategy)
            
        elif choice == "2":
            topic = input("\nEnter Topic to Evaluate: ").strip()
            # Simply run the eval on existing index? 
            # For simplicity in this menu, we'll assume the user wants full pipeline but mostly cares about numbers.
            # Or we could implement a dedicated eval-only mode if the index exists.
            print("Running quick evaluation pass...")
            run_interactive_pipeline(topic, "hyde")
            
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
