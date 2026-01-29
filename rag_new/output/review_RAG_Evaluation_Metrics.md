# RAG Evaluation Metrics

 In this literature review, we focus on the evaluation metrics employed for Retrieval Augmented Generation (RAG) systems in the context of Fintech Agentic Design, as outlined in [Author, Year]. The authors introduce a novel approach called A-RAG, which addresses the challenge of fragmented enterprise knowledge in fintech domains by synthesizing partial context across sources more effectively than traditional RAG methods (Section I).

Two key RAG systems are compared in the empirical study: B-RAG and A-RAG. The evaluation methodology involves both quantitative and qualitative comparisons, measuring effectiveness through metrics such as retrieval accuracy and answer relevance ([Author, Year]). The authors use a per-question score difference ∆s to assess the performance of each system, with a positive median value indicating that A-RAG outperforms B-RAG in most cases. Specifically, A-RAG achieves [missing data] while B-RAG attains 66.67% ([Author, Year]).

One crucial advantage of A-RAG is its ability to operate within the constraints of a specialized ontology, particularly when semantic cues in documents are minimal or inconsistently structured ([Author, Year]). This superior performance is essential for fintech applications, where public datasets, crowd-sourced relevance judgments, or static ground truth are typically unavailable due to the proprietary and dynamic nature of financial data ([Author, Year]).

In contrast, traditional RAG benchmarks assume publicly available datasets, which makes them infeasible for fintech applications. Designing on-prem, domain-specific RAG applications requires several considerations, including but not limited to: handling sensitive data, ensuring scalability, and integrating with existing financial systems ([Author, Year]).

In conclusion, the evaluation of RAG systems in fintech agentic design involves comparing the performance of different approaches using metrics such as retrieval accuracy, answer relevance, and per-question score difference. A-RAG demonstrates a stronger ability to synthesize partial context across sources and perform well within the constraints of a specialized ontology compared to traditional RAG methods ([Author, Year]). However, designing domain-specific RAG applications for fintech requires careful consideration of various factors due to the sensitive and dynamic nature of financial data.

References:
[Author, Year] Retrieval Augmented Generation RAG for Fintech Agentic Design and Evaluation.pdf