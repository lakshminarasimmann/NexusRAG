# 🚀 ResearchGPT - Agentic RAG Pipeline

> **"One command to convert arXiv chaos into IEEE clarity."**

This is not just a search engine. It is an **Agentic RAG (Retrieval-Augmented Generation)** system that acts as an autonomous research assistant. It fetches papers, reads them, thinks about them, and writes a professional literature review with strict citations.

---

## ⚡ Quick Start (The "One Command" Mode)

If you just want results, run this. It handles ingestion, indexing, generation, and rigorous evaluation in one go.

```bash
python main.py run_all --query "Self-Healing Large Language Models"
```

**What happens when you run this?**
1.  **📥 Ingest**: Deletes old data, searches arXiv for "Self-Healing LLMs", downloads the top 3 PDFs.
2.  **🧠 Index**: Parses PDFs, splits them into semantic chunks, and builds a fresh Vector Database.
3.  **🔎 Retrieve**: Uses **HyDE** to hallucinate a perfect answer, searches the vector space, and uses a **Cross-Encoder** to re-rank findings.
4.  **📝 Generate**: Uses **Chain-of-Thought** reasoning to write a grounded review.
5.  **⚖️ Evaluate**: An LLM Judge reads the review and scores it (1-5) for Faithfulness and Relevance.

---

## 🏗️ Advanced Architecture

We have implemented state-of-the-art RAG techniques to ensure high precision:

### 1. 🔮 HyDE (Hypothetical Document Embeddings)
*   *Problem*: Users ask short questions ("How does LoRA work?"), but papers contain long technical answers. The vectors don't match well.
*   *Solution*: The system first asks the LLM to **hallucinate** a hypothetical scientific abstract answering the question. It embeds *that* hallucination to find real papers with similar semantic patterns.

### 2. 🎯 Cross-Encoder Re-Ranking
*   *Problem*: Vector search is fast but "fuzzy". It retrieves top-50 candidates, but many are irrelevant.
*   *Solution*: We use a **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**. It acts as a strict judge, reading the Query and Document *together* and outputting a relevance score. We keep only the Top-5 highest scorers.

### 3. 🧠 Chain-of-Thought (CoT) Generation
*   *Problem*: LLMs often hallucinate or write generic summaries.
*   *Solution*: We prompt the LLM to **"Think step-by-step"**:
    1.  Analyze retrieved context.
    2.  Identify agreements/disagreements.
    3.  Formulate a synthesis.
    4.  Write the review.
    This intermediate reasoning step drastically improves quality.

### 4. ⚖️ G-Eval (LLM-as-a-Judge)
*   *Problem*: How do we know if the answer is good? BLEU/ROUGE scores are useless for facts.
*   *Solution*: We use a separate LLM pass as a "Judge".
    *   **Faithfulness (1-5)**: "Is every claim in the answer supported by the retrieved text?"
    *   **Relevance (1-5)**: "Does the answer actually address the user's specific question?"

### 5. 🔄 Autonomous Self-Correction
*   *Problem*: Sometimes the first retrieval attempt misses context or the generator hallucinates.
*   *Solution*: The system is **Agentic**. It critiques its own output.
    *   If score < 4.0: It retries with **Deep Retrieval** (Top-10) or **Unfiltered** (No refinement) strategies.
    *   It only saves the result when it passes the quality bar.

### 7. 🧩 Sub-Question Decomposition (New)
*   *Problem*: Complex queries like "Compare X and Y" fail with simple search.
*   *Solution*: The system breaks the question into:
    1.  "What is X?"
    2.  "What is Y?"
    3.  Retrieves answers for *both* independently.
    4.  Synthesizes a comparison.

---

## 🛠️ Usage Guide (Interactive Mode)

We have replaced the complex CLI flags with a simple **Interactive Menu**.

```bash
python main.py
```

**Menu Options:**
1.  **Run RAG Pipeline**:
    - Enter a Topic (e.g., "AI in Healthcare")
    - Select Strategy:
        - `HyDE`: Best for most questions.
        - `Decomposition`: Best for "Compare..." or "How does X affect Y?" logic.
        - `Standard`: Faster, simple vector search.
2.  **Evaluation Mode**: Run the pipeline and focus on the G-Eval scores.
3.  **Exit**.


1.  **Clone & Install**:
    ```bash
    git clone <repo_url>
    cd rag_new
    pip install -r requirements.txt
    ```

2.  **Configure LLM**:
    *   This system is optimized for **Ollama**.
    *   Install [Ollama](https://ollama.com).
    *   Run: `ollama pull mistral`
    *   (Optional) Use OpenAI by editing `config/config.py`.

---

## � CLI Command Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `run_all` | `--query "..."` | **Recommended**. Runs the full pipeline from A to Z. |
| `ingest` | `--query "..." --max 5` | Searches arXiv and downloads PDFs. |
| `index` | *(none)* | Processes PDFs and builds/resets the index. |
| `retrieve` | `--query "..."` | Debug mode. Shows HyDE output, candidates, and Re-ranking scores. |
| `generate` | `--topic "..."` | Generates a review from the current index. |
| `evaluate` | `--query "..."` | Runs the G-Eval metrics on the current index. |

---

## 📊 Evaluation & Accuracy

This system uses **G-Eval (LLM-as-a-Judge)** to score performance. Unlike traditional ML where accuracy is simple (True/False), RAG accuracy is nuanced.

### The Scoring Rubric (1-5)
We use a **Mistral/GPT-4** judge to scrutinize the output:

| Score | Meaning | Action Required |
| :--- | :--- | :--- |
| **5/5** | **Perfect**. Fully grounded, answers the specific question. | ✅ Ready for prod. |
| **4/5** | **Good**. Generally correct but may miss nuance. | ✅ Acceptable. |
| **3/5** | **Acceptable**. hallucinated minor details or vague. | ⚠️ Improve context. |
| **1-2/5** | **Failure**. Hallucination or Irrelevant. | ❌ Fix retrieval. |

### How to Calculate "Accuracy"
To get a single percentage score for your system:

1.  Run `evaluate` or `run_all` on a test set of 10-20 questions.
2.  Average the **Faithfulness** and **Relevance** scores.
3.  **Formula**:
    ```math
    System Accuracy % = ((Avg_Faithfulness + Avg_Relevance) / 10) * 100
    ```
    *Example: Faithfulness 4.5, Relevance 4.5 -> 90% Accuracy.*

### Reading the Logs
```text
Faithfulness Score: 5/5
Reasoning: The answer is entirely derived from the retrieved papers [Author A]...
```
If **Faithfulness** is low: Your `Ingestion` or `Retrieval` is failing (Garbage In, Garbage Out).
If **Relevance** is low: Your `Generation` prompt is bad or the `Retrieval` missed the specific angle of the question.

---

## � Project Structure

```text
rag_new/
├── config/             # Configuration (Models, Top-K, Paths)
├── data/               # Raw PDF storage & VectorDB
├── output/             # Where your reviews are saved
├── src/
│   ├── ingestion/      # arXiv Scraper (Auto-cleaning)
│   ├── processing/     # PDF Parsing & Chunking
│   ├── embedding/      # Vector Store Logic
│   ├── retrieval/      # HyDE + Cross-Encoder Logic
│   ├── generation/     # CoT Prompts & LLM Client
│   └── evaluation/     # G-Eval Metrics
└── main.py             # Master CLI Tool
```

---

> **Note**: This is an "Industry-Ready" template. In a real deployment, you would replace ChromaDB with Pinecone/Weaviate and deploy the API via FastAPI.
