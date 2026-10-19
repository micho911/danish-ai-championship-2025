# Emergency Healthcare RAG

**Task** ([full description](TASK.md)): given a short medical statement, predict (1) whether it is true or false and (2) which of 115 emergency-medicine topics it belongs to. The reference corpus is a set of StatPearls articles. Inference had to run locally (no cloud LLM APIs) under a time limit.

## Approach

### v2: retrieval + meta-classifier (main attempt)

1. **Indexing** ([`build_index.py`](build_index.py)): clean the StatPearls markdown, chunk it with overlap, embed the chunks with `intfloat/e5-large-v2` on GPU and store them in an **HNSW** index (`hnswlib`) along with passage→topic metadata.
2. **Topic prediction** ([`rag_model2.py`](rag_model2.py)): retrieve the top-k passages and assign the topic by confidence-weighted majority vote over the topics of the retrieved passages.
3. **Truth prediction** ([`truth_features2.py`](truth_features2.py), [`truth_meta2.py`](truth_meta2.py)): build hand-engineered features from the statement and retrieved evidence (retrieval similarities at several chunk levels, negation and uncertainty cues, a true/false verdict from a local LLM via **Ollama**, `deepseek-r1:32b` by default) and pass them to a calibrated meta-classifier.
4. **Training** ([`train_truth_meta2.py`](train_truth_meta2.py)): logistic regression with feature selection, gradient boosting and a voting ensemble, evaluated with stratified cross-validation.

### v1: fine-tuned PubMedBERT ([`v1-pubmedbert-finetune/`](v1-pubmedbert-finetune/))

My first attempt was a multi-task classifier on `microsoft/BiomedNLP-PubMedBERT`, with a truth head and a topic head, class weighting, augmentation and LoRA ([`optimized_training.py`](optimized_training.py)). With only 200 labelled training statements, I switched to an approach that leans on the reference corpus instead.

## Status

Unfinished. The scripts import modules under their in-progress names (`enhanced_truth_features`, `enhanced_rag_model`), while the files on disk carry the `*2.py` names, so the pipeline won't run end-to-end as it is.

## What I learned

- Building a dense-retrieval pipeline: chunking, embedding models and approximate nearest-neighbour search.
- The trade-off between fine-tuning a domain model and grounding predictions in retrieved evidence when labelled data is scarce.
- Running local LLMs (Ollama) on HPC GPUs under a latency budget.
