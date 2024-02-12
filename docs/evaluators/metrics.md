# Metrics

ConsistencyBench contains three classes of metrics to measure semantic similarity of two pieces of text.

1. Using an auxiliary model to detect if two phrases are paraphrases of each other (`PP_Detector`)
2. Based on Natural Language Inference (`NLI`)
3. Through agreement-based notions of similarity, such as BLEU and BERTScore (`Agreement`)

::: consistencybench.evaluators.metrics