import json
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
# ---------- Load data ----------
with open("final_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

texts = [doc["text"] for doc in corpus]
urls = [doc["url"] for doc in corpus]

# Load FAISS embeddings we had created
index = faiss.read_index("faiss.index")
# ---------- Models ----------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
generator = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# ---------- BM25 ----------
tokenized_texts = [nltk.word_tokenize(t.lower()) for t in texts]
bm25 = BM25Okapi(tokenized_texts)


# ---------- Retrieval ----------

def dense_search(query, top_k=10):
    q_emb = embedder.encode(query).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))

    scores, idx = index.search(q_emb.reshape(1, -1), top_k)
    return idx[0], scores[0]

def bm25_search(query, top_k=10):
    tokens = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokens)
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores[idx]


def rrf_fusion(dense_idx, bm25_idx, k=60):
    scores = {}
    for rank, idx in enumerate(dense_idx):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    for rank, idx in enumerate(bm25_idx):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# def hybrid_search(query, top_k=10, rerank_k=20):
#     dense_idx, _ = dense_search(query, rerank_k)
#     bm25_idx, _ = bm25_search(query, rerank_k)

#     fused_idx = rrf_fusion(dense_idx, bm25_idx)
#     fused_idx = fused_idx[:rerank_k]

#     pairs = [(query, texts[i]) for i in fused_idx]
#     rerank_scores = reranker.predict(pairs)

#     ranked = sorted(
#         zip(fused_idx, rerank_scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     results = []
#     for idx, score in ranked[:top_k]:
#         results.append({
#             "text": texts[idx],
#             "url": urls[idx],
#             "score": float(score)
#         })

#     return results

def hybrid_search(query, top_k=10, rerank_k=20):
    # --- Dense search ---
    dense_idx, dense_scores = dense_search(query, rerank_k)
    dense_score_map = {
        idx: float(score)
        for idx, score in zip(dense_idx, dense_scores)
    }

    # --- BM25 search ---
    bm25_idx, bm25_scores = bm25_search(query, rerank_k)
    bm25_score_map = {
        idx: float(score)
        for idx, score in zip(bm25_idx, bm25_scores)
    }

    # --- RRF fusion (compute AND store scores) ---
    rrf_scores = {}
    k = 60

    for rank, idx in enumerate(dense_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(bm25_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)

    fused_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_idx = fused_idx[:rerank_k]

    # --- Reranking ---
    pairs = [(query, texts[i]) for i in fused_idx]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(fused_idx, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # --- Build final results with ALL scores ---
    results = []
    for idx, rerank_score in ranked[:top_k]:
        results.append({
            "text": texts[idx],
            "url": urls[idx],
            "dense_score": dense_score_map.get(idx),
            "bm25_score": bm25_score_map.get(idx, 0.0),
            "rrf_score": rrf_scores.get(idx),
            "rerank_score": float(rerank_score),
        })

    return results

def generate_answer(query, top_k=3):
    chunks = hybrid_search(query, top_k=top_k)

    context = "\n\n".join([c["text"] for c in chunks])

    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator.generate(
        **inputs,
        max_new_tokens=256
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, chunks
