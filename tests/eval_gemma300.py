"""Evaluate retrieval quality of original and trimmed model using qrels."""
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

ORIGINAL_MODEL = 'google/embeddinggemma-300m'
TRIMMED_MODEL = 'model/gemma300-vi-trimmed'
QRELS_PATH = 'crosslingual/synthetic/cross_queries.parquet'
CORPUS_PATH = 'crosslingual/eval/filtered_corpus.json'
TOP_K = 10
SAMPLE = 200

# load qrels
df = pd.read_parquet(QRELS_PATH).head(SAMPLE)
queries = df['query'].tolist()
relevant_docs = df['pos'].tolist()  # list of lists

# load eval corpus
with open(CORPUS_PATH) as f:
    import json
    corpus = list(json.load(f).values())
doc_to_idx = {doc: idx for idx, doc in enumerate(corpus)}
print(f"Queries: {len(queries)}, Corpus: {len(corpus)}")


def evaluate(model_path, queries, corpus, relevant_docs, top_k):
    model = SentenceTransformer(model_path)
    q_emb = model.encode_query(queries, show_progress_bar=True)
    d_emb = model.encode_document(corpus, show_progress_bar=True)
    scores = model.similarity(q_emb, d_emb)

    recall, mrr, ndcg = [], [], []
    for i, rel_docs in enumerate(relevant_docs):
        rel_indices = {doc_to_idx[d] for d in rel_docs if d in doc_to_idx}
        top_k_indices = torch.topk(scores[i], min(top_k, len(corpus))).indices.tolist()

        # Recall@K
        hits = len(rel_indices & set(top_k_indices))
        recall.append(hits / len(rel_indices) if rel_indices else 0)

        # MRR
        mrr_score = 0
        for rank, idx in enumerate(top_k_indices):
            if idx in rel_indices:
                mrr_score = 1 / (rank + 1)
                break
        mrr.append(mrr_score)

        # NDCG@K
        dcg = sum(1 / (torch.log2(torch.tensor(rank + 2.0))) for rank, idx in enumerate(top_k_indices) if idx in rel_indices)
        idcg = sum(1 / (torch.log2(torch.tensor(rank + 2.0))) for rank in range(min(len(rel_indices), top_k)))
        ndcg.append((dcg / idcg).item() if idcg > 0 else 0)

    print(f"  Recall@{top_k}:  {sum(recall) / len(recall) * 100:.1f}%")
    print(f"  MRR@{top_k}:     {sum(mrr) / len(mrr) * 100:.1f}%")
    print(f"  NDCG@{top_k}:    {sum(ndcg) / len(ndcg) * 100:.1f}%")


print("\n--- Original model ---")
evaluate(ORIGINAL_MODEL, queries, corpus, relevant_docs, TOP_K)

print("\n--- Trimmed model ---")
evaluate(TRIMMED_MODEL, queries, corpus, relevant_docs, TOP_K)
