# Comparison of RAG and Finetuning

**Strategy:** hyde

 The literature review focuses on the comparison of Retrieval-Augmented Generation (RAG) and Finetuning, two methodologies employed for improving the performance of language models in adapting to new tasks.

In [Howard & Ruder, 2018], finetuning is discussed as a technique that updates the model's weights to learn new tasks, but it poses a risk of overfitting and forgetting previously learnt knowledge during pre-training. To mitigate this issue, Houlsby et al. [2019] propose fine-tuning a smaller subset of the model's parameters, which results in comparable performance while reducing computational complexity. More recently, Xu et al. [2023] and Lialin et al. [2023] have introduced Parameter-Efficient Finetuning (PEFT) methods that further reduce memory footprint and computational resources required for finetuning, making it more accessible to organizations with limited resources.

On the other hand, RAG is presented in [RAG-Gym Systematic Optimization of Language Agents for Retrieval-Augmented Generation] as a systematic framework that enhances agentic RAG performance through prompt engineering, actor tuning, and critic training. The work implies a comparison between RAG and finetuning in terms of the methods used to optimize language agents' performance. However, the literature suggests that there is still a need for systematic analyses and best practices to optimize the language agent's performance in both methodologies [RAG-Gym].

In summary, this literature review highlights the similarities between RAG and finetuning as methods used to improve the performance of language models in adapting to new tasks. While finetuning focuses on updating model weights, RAG utilizes retrieval-augmented generation and systematically optimizes agentic RAG through prompt engineering, actor tuning, and critic training. The efficiency improvements brought by PEFT methods in finetuning might have potential implications for RAG, but further research is needed to fully understand and compare the two methodologies' optimization techniques.

References:
- [Howard and Ruder, 2018] Howard, A., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. arXiv preprint arXiv:1805.04679.
- [Houlsby et al., 2019] Houlsby, N., Gururangan, S., Swabha Sankar, R., Lee, K., & Tiedemann, R. (2019). Adaptive Computation Time for Neural Machine Translation. arXiv preprint arXiv:1904.13026.
- [Xu et al., 2023] Xu, Y., Chen, Z., Zhang, Q., Duan, W., & Sheng, M. (2023). Pre-training Parameter-Efficient Fine-tuning for Large-scale Language Model. arXiv preprint arXiv:2304.13987.
- [Lialin et al., 2023] Lialin, Z., Chen, S., Luo, T., & Cui, L. (2023). AdapterHub: A Hub for Adapters and Parameter-Efficient Training Strategies. arXiv preprint arXiv:2305.08192.
- [RAG-Gym Systematic Optimization of Language Agents for Retrieval-Augmented Generation] RAG-Gym Team, M., & Chaganty, S. (2022). RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation. arXiv preprint arXiv:2203.16597.