# 📚 MechRabot — Retrieval Evaluation Notebook: Full Learning Guide

> **Goal of this notebook:** After you build a RAG system (documents stored in Qdrant), you need to *test* how good the retrieval actually is. This notebook does exactly that — it runs 30 test queries against Qdrant and mathematically measures the quality of what comes back.

---

## 🗺️ Big Picture: What Are We Doing?

```
Evaluation Dataset (30 labeled queries)
         │
         ▼
 Embed queries with BGE-M3
         │
         ▼
 Send to Qdrant → get Top-10 results per query
         │
         ▼
 Compare results vs. ground truth labels
         │
         ▼
 Compute MRR@10, NDCG@10, Recall@K
         │
         ▼
 Break down by language (en / ar / ar_slang)
```

---

## ⚙️ Cell 1 — Kaggle Environment Setup

```python
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

### What it does
- Standard Kaggle boilerplate — prints all available dataset files so you know what paths to use.
- `numpy` → math/arrays. `pandas` → data tables. `os` → file system traversal.

> [!NOTE]
> This cell runs in Kaggle. When you remake locally, replace `/kaggle/input/...` paths with your actual local paths.

---

## ⚙️ Cell 2 — Install Dependencies

```python
!pip install "numpy<2.0.0" "scipy<1.14.0" "scikit-learn<1.5.0" \
    "transformers>=4.41.0,<4.44.0" FlagEmbedding qdrant-client accelerate -U --quiet
```

### Why these specific versions?
FlagEmbedding (BGE-M3) is sensitive to numpy and transformers versions. The version pins prevent silent math errors or crashes. Key libraries:

| Library | Purpose |
|---|---|
| `FlagEmbedding` | Provides the BGE-M3 model (dense + sparse + colbert) |
| `qdrant-client` | Python SDK to talk to Qdrant vector database |
| `transformers` | HuggingFace, required by FlagEmbedding |
| `accelerate` | Speeds up model inference |

---

## ⚙️ Cell 3 — Connect to Qdrant

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="https://935c3158-...",
    api_key="eyJhbG..."
)
col_name = "mechrabot_Vdb_1"
print("Qdrant portal is ready ✅")
```

### What it does
Creates a connection object (`client`) to the cloud Qdrant instance where your MechRabot documents are already stored.

- `url` → The Qdrant cloud cluster address
- `api_key` → Authentication token
- `col_name` → The collection that holds your embedded document chunks

> [!IMPORTANT]
> The API key here is embedded in the notebook. When you remake this, store it in an environment variable or a config file — **never commit secrets to Git**.

---

## ⚙️ Cell 4 — Load Evaluation Data

```python
import json

file_path = "/kaggle/input/.../evaluation_30_V1.json"
with open(file_path, "r", encoding="utf-8") as file:
    eval_data = json.load(file)
    for key, value in eval_data.items():
        for i in value:
            print("-" * 30)
            print(f"{key}: {i}\n")
```

### What is inside `evaluation_30_V1.json`?

This is your **ground truth file** — a manually labeled test set. Its structure looks like:

```json
{
  "queries": [
    {
      "id": "EV-001",
      "query": "What is the torque of the motor?",
      "language": "en",
      "relevant_chunk_ids": ["chunk_042", "chunk_117"]
    },
    {
      "id": "EV-002",
      "query": "إيه الكوبلنج المناسب؟",
      "language": "ar",
      "relevant_chunk_ids": ["chunk_088"]
    },
    ...
  ]
}
```

Each query has:
- **`id`** → unique identifier like `EV-001`
- **`query`** → the actual question text
- **`language`** → `en`, `ar`, or `ar_slang`
- **`relevant_chunk_ids`** → the chunk IDs a human labeled as "correct answer"

---

## ⚙️ Cell 5 — Extract Query List

```python
query_list = []
for chunk in eval_data["queries"]:
    query_list.append(chunk["query"])

print(f"{len(query_list)} loaded...")
```

### What it does
Extracts just the **query text strings** into a flat list, like:
```python
query_list = [
    "What is the torque of the motor?",
    "إيه الكوبلنج المناسب؟",
    ...
]
```
This list will be fed to the embedding model. There are 30 queries total.

---

## ⚙️ Cell 6 — Initialize BGE-M3 and Embed Queries

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

embedding_docu = model.encode(
    query_list,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
    batch_size=64,
    max_length=512
)
```

### What is BGE-M3?

BGE-M3 is a **multi-representation embedding model** from BAAI. "M3" means it can produce 3 types of embeddings simultaneously from one pass:

| Type | Name in output | What it captures |
|---|---|---|
| Dense | `dense_vecs` | Overall semantic meaning (one vector per query) |
| Sparse | `lexical_weights` | Keyword importance weights (like BM25 but neural) |
| ColBERT | `colbert_vecs` | Per-token vectors for fine-grained matching |

**Why 3 types?** Each catches different retrieval signals. Hybrid search combines them for better results.

### Parameters explained
- `use_fp16=True` → Use half-precision floats (faster, less memory, negligible quality loss)
- `batch_size=64` → Process 64 queries at a time (memory efficiency)
- `max_length=512` → Truncate queries to 512 tokens max

### Output shape (for 30 queries)
- `dense_vecs` → shape `(30, 1024)` — 30 queries × 1024-dim vector each
- `lexical_weights` → list of 30 dicts `{token_id: weight}`  
- `colbert_vecs` → list of 30 matrices, each `(n_tokens, 1024)`

---

## ⚙️ Cell 7 — Sanity Check Embeddings

```python
sample_idx = 0
dense_output  = embedding_docu['dense_vecs']
sparse_output = embedding_docu['lexical_weights']
colbert_output= embedding_docu['colbert_vecs']

print(f"Dense Shape: {dense_output.shape}")          # (30, 1024)
print(f"Sparse tokens: {len(sparse_output[0])}")     # number of active tokens
print(f"ColBERT matrix: {colbert_output[0].shape}")  # (n_tokens, 1024)
```

### Why do a sanity check?
Before querying thousands of documents, you verify the embeddings actually look right. Common issues caught here:
- Wrong shape → model setting error
- All zeros → encoding failed silently
- Single vector when expecting batched → indexing bug

---

## ⚙️ Cell 8 — Query Qdrant with Hybrid Search (Step 1)

This is the **core retrieval cell**.

```python
K = 10  # evaluate at Top-10

results = {}  # { "EV-001": {"chunk_id": score, ...}, ... }
qrels   = {}  # { "EV-001": {"chunk_id": 1, ...}, ... }  ← ground truth
queries = {}  # { "EV-001": "query text", ... }

for i, q in enumerate(eval_data["queries"]):
    qid = q["id"]
    queries[qid] = q["query"]
    
    # Ground truth: which chunks are relevant?
    qrels[qid] = {cid: 1 for cid in q["relevant_chunk_ids"]}
```

### Building `qrels` (Query Relevance Judgments)

`qrels` is the standard IR (Information Retrieval) structure for ground truth:
```python
qrels = {
    "EV-001": {"chunk_042": 1, "chunk_117": 1},  # binary: 1 = relevant
    "EV-002": {"chunk_088": 1},
    ...
}
```

The `1` means "this chunk is relevant to this query". In binary evaluation there are only 0s and 1s.

### The Hybrid Search Query

```python
hits = client.query_points(
    collection_name=col_name,
    prefetch=[
        # Dense retrieval: semantic similarity
        models.Prefetch(
            query=embedding_docu["dense_vecs"][i].tolist(),
            using="dense",
            limit=K * 2,  # fetch 20 candidates
        ),
        # Sparse retrieval: keyword matching
        models.Prefetch(
            query=models.SparseVector(
                indices=[int(k) for k in embedding_docu["lexical_weights"][i].keys()],
                values=[float(v) for v in embedding_docu["lexical_weights"][i].values()],
            ),
            using="sparse",
            limit=K * 2,
        ),
    ],
    query=embedding_docu["dense_vecs"][i].tolist(),  # final re-rank by dense
    using="dense",
    limit=K,
)
```

### 🔍 How Hybrid Search Works (Step by Step)

**Stage 1 — Prefetch (parallel candidate generation):**
- Dense search: finds 20 chunks most semantically similar
- Sparse search: finds 20 chunks with most keyword overlap
- Result: up to 40 candidate chunks (with duplicates)

**Stage 2 — Fusion (Qdrant RRF internally):**
- Qdrant merges the two candidate lists using **Reciprocal Rank Fusion (RRF)**
- A chunk that ranks high in BOTH lists gets a big boost

**Stage 3 — Final Re-rank:**
- The fused candidates are re-scored by dense similarity
- Top `K=10` are returned

### Storing results

```python
results[qid] = {}
for rank, point in enumerate(hits.points):
    results[qid][point.id] = point.score
```

After the loop, `results` looks like:
```python
{
    "EV-001": {"chunk_042": 0.91, "chunk_203": 0.87, "chunk_055": 0.84, ...},
    "EV-002": {"chunk_088": 0.93, ...},
    ...
}
```

---

## ⚙️ Cell 9 — Compute Metrics (Step 2)

This cell implements **3 standard IR evaluation metrics from scratch**, with no external library.

---

### 📐 Metric 1: MRR@K (Mean Reciprocal Rank)

**The question it answers:** *"On average, how high up in the ranked list does the first correct answer appear?"*

```python
def compute_mrr(qrels, results, k=10):
    mrr_scores = []
    for qid in qrels:
        ranked = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        rr = 0.0
        for rank, (doc_id, score) in enumerate(ranked, 1):  # rank starts at 1
            if doc_id in qrels[qid]:
                rr = 1.0 / rank   # Reciprocal Rank
                break
        mrr_scores.append(rr)
    return sum(mrr_scores) / len(mrr_scores)
```

**The Math:**

$$MRR@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{rank_q}$$

Where `rank_q` = position of first relevant doc for query `q`.

**Worked Example:**

| Query | First relevant doc at rank | Reciprocal Rank |
|---|---|---|
| EV-001 | 1 | 1/1 = 1.000 |
| EV-002 | 3 | 1/3 = 0.333 |
| EV-003 | Not found | 0/0 = 0.000 |

MRR@10 = (1.000 + 0.333 + 0.000) / 3 = **0.444**

**Interpretation:**
- MRR = 1.0 → Perfect (relevant always at rank 1)
- MRR = 0.5 → Relevant typically at rank 2
- MRR = 0.0 → Never found relevant docs

> [!TIP]
> MRR is ideal for single-answer scenarios — like "what is the formula for gear ratio?" where there's one correct chunk.

---

### 📐 Metric 2: Recall@K

**The question it answers:** *"Out of all the relevant docs that exist, what fraction did we find in our top K results?"*

```python
def compute_recall(qrels, results, k=10):
    recall_scores = []
    for qid in qrels:
        ranked = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        retrieved_ids = {doc_id for doc_id, _ in ranked}
        relevant_ids = set(qrels[qid].keys())
        
        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        recall_scores.append(recall)
    return sum(recall_scores) / len(recall_scores)
```

**The Math:**

$$Recall@K = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Relevant}|}$$

**Worked Example:**

| Query | Relevant chunks | Found in top-10 | Recall |
|---|---|---|---|
| EV-001 | {chunk_042, chunk_117} | {chunk_042} | 1/2 = 0.50 |
| EV-002 | {chunk_088} | {chunk_088} | 1/1 = 1.00 |

Avg Recall@10 = (0.50 + 1.00) / 2 = **0.75**

**Why K=1, K=5, K=10?**
The notebook computes recall at multiple cutoffs:
- `Recall@1` → Does the single best result contain a relevant doc?
- `Recall@5` → Any relevant doc in first 5?
- `Recall@10` → Any relevant doc in first 10?

Useful for understanding where performance drops off.

> [!TIP]
> Recall is ideal when there are **multiple relevant chunks** per query, and you want to know if you found them all.

---

### 📐 Metric 3: NDCG@K (Normalized Discounted Cumulative Gain)

**The question it answers:** *"How well did we rank the relevant docs? Higher rank = better, and we penalize for putting them lower."*

```python
def compute_ndcg(qrels, results, k=10):
    ndcg_scores = []
    for qid in qrels:
        ranked = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        
        # DCG: reward for finding relevant docs, discounted by position
        dcg = 0.0
        for rank, (doc_id, score) in enumerate(ranked, 1):
            rel = qrels[qid].get(doc_id, 0)  # 1 if relevant, 0 if not
            dcg += rel / math.log2(rank + 1)
        
        # IDCG: best possible DCG (all relevant docs at top)
        ideal_rels = sorted(qrels[qid].values(), reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:k], 1):
            idcg += rel / math.log2(rank + 1)
        
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcg_scores) / len(ndcg_scores)
```

**The Math:**

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

**Why the log discount?**

| Rank | `log2(rank+1)` | Discount |
|---|---|---|
| 1 | log2(2) = 1.0 | No discount (best position) |
| 2 | log2(3) = 1.58 | Small discount |
| 5 | log2(6) = 2.58 | Medium discount |
| 10 | log2(11) = 3.46 | Large discount |

Relevant docs found at rank 1 contribute `1/1 = 1.0`. Found at rank 10 contribute `1/3.46 = 0.29`. This mimics real user behavior — users rarely look past rank 3-4.

**Worked Example (1 query, 2 relevant chunks):**

Relevant = `{chunk_042: 1, chunk_117: 1}`

Retrieved order: `[chunk_203, chunk_042, chunk_117, ...]`

| Rank | Doc | Relevant? | DCG contribution |
|---|---|---|---|
| 1 | chunk_203 | No (0) | 0 / log2(2) = 0 |
| 2 | chunk_042 | Yes (1) | 1 / log2(3) = 0.631 |
| 3 | chunk_117 | Yes (1) | 1 / log2(4) = 0.500 |

DCG = 0 + 0.631 + 0.500 = **1.131**

Ideal (both at top): rank 1 + rank 2 = 1.0 + 0.631 = **1.631**

NDCG = 1.131 / 1.631 = **0.693**

> [!TIP]
> NDCG is the most complete metric because it rewards both **finding** relevant docs AND **ranking them high**.

---

### Final Metric Print

```python
mrr_10    = compute_mrr(qrels, results, k=10)
ndcg_10   = compute_ndcg(qrels, results, k=10)
recall_1  = compute_recall(qrels, results, k=1)
recall_5  = compute_recall(qrels, results, k=5)
recall_10 = compute_recall(qrels, results, k=10)
```

Example output you'd aim for:
```
MRR@10    = 0.7833
NDCG@10   = 0.8012
Recall@1  = 0.6333
Recall@5  = 0.8667
Recall@10 = 0.9000
```

---

## ⚙️ Cell 10 — Language Breakdown (Step 3)

```python
lang_groups = {}
for q in eval_data["queries"]:
    lang = q["language"]
    if lang not in lang_groups:
        lang_groups[lang] = {"qrels": {}, "results": {}}
    qid = q["id"]
    lang_groups[lang]["qrels"][qid] = qrels[qid]
    lang_groups[lang]["results"][qid] = results[qid]
```

### What it does
Groups the 30 queries by their language label and **re-runs all 3 metrics per group**.

```
Language     Count  MRR@10     NDCG@10    Recall@10
------------------------------------------------------------
en           10     0.9000     0.9200     1.0000
ar           10     0.8500     0.8700     0.9500
ar_slang     10     0.6500     0.6800     0.7000
```

### Why is this the most important analysis?

Your RAG system serves Arabic-speaking engineers who write in Egyptian dialect (slang). BGE-M3 is multilingual but was trained mostly on formal text. The breakdown reveals:

- **If `en >> ar_slang`** → The model struggles with dialect. You need fine-tuning or query expansion.
- **If `en ≈ ar_slang`** → BGE-M3 handles it natively. No extra work needed.

The language breakdown directly informs your **next development step**.

---

## ⚙️ Cell 11 — Per-Query Hit/Miss Table (Step 4)

```python
for q in eval_data["queries"]:
    qid = q["id"]
    ranked = sorted(results[qid].items(), key=lambda x: -x[1])[:10]
    
    rank_found = "-"
    hit = "❌"
    for rank, (doc_id, score) in enumerate(ranked, 1):
        if doc_id in qrels[qid]:
            rank_found = str(rank)
            hit = "✅"
            break
    
    print(f"{qid:<8} {lang:<10} {hit:<6} {rank_found:<6} {q['query'][:50]}")
```

### Sample output

```
ID       Lang       Hit?   Rank   Query
--------------------------------------------------------------------
EV-001   en         ✅     1      What is the torque of the motor?
EV-002   ar         ✅     3      ما هو الكوبلنج المناسب؟
EV-003   ar_slang   ❌     -      إيه إللي بيخلي الموتور يسخن؟
EV-004   en         ✅     2      How to calculate shaft diameter?
```

### Why do this?

Aggregate metrics like MRR=0.78 hide which specific queries are failing. This table lets you:
1. **Spot patterns in failures** — Are all misses Arabic slang? Multi-hop questions?
2. **Manually inspect** — Go look at the retrieved chunks for ❌ queries and understand *why* they failed
3. **Prioritize fixes** — Fix the highest-impact failures first

> [!NOTE]
> `rank_found = "-"` means the relevant chunk was NOT in the top 10 at all. This is the worst case.

---

## 🎯 Overall Workflow Summary

```
Step 0: Setup  →  Install libraries, connect Qdrant
Step 1: Load   →  Read evaluation JSON (30 labeled queries)
Step 2: Embed  →  BGE-M3 encodes all queries (dense + sparse + colbert)
Step 3: Query  →  Hybrid search per query → ranked results dict
Step 4: Eval   →  Compute MRR@10, NDCG@10, Recall@K from scratch
Step 5: Drill  →  Language breakdown (en / ar / ar_slang)
Step 6: Debug  →  Per-query hit/miss table
```

---

## 📖 Concepts Glossary

| Term | Meaning |
|---|---|
| **qrels** | Query Relevance Judgments — the ground truth map of query → relevant doc IDs |
| **Dense vector** | Single fixed-size vector capturing overall semantic meaning |
| **Sparse vector** | Dictionary of `{token_id: weight}` — good for keyword matching |
| **ColBERT** | Per-token vectors — enables late interaction for fine-grained matching |
| **Hybrid search** | Combining dense + sparse retrieval for best of both worlds |
| **RRF** | Reciprocal Rank Fusion — merges ranked lists by rewarding docs ranked high in multiple lists |
| **MRR** | Mean Reciprocal Rank — rewards finding the first correct answer as early as possible |
| **NDCG** | Normalized Discounted Cumulative Gain — rewards finding relevant docs AND ranking them correctly |
| **Recall@K** | Fraction of all relevant docs found in the top K results |
| **K** | The cutoff — e.g. top-10 means you only look at the first 10 results |

---

## 🔄 When You Remake This Notebook

### Things to change

1. **Paths** → Replace `/kaggle/input/...` with your local file path to the JSON
2. **Qdrant credentials** → Use `os.environ["QDRANT_API_KEY"]` instead of hardcoding
3. **Collection name** → Use your actual collection name
4. **K value** → Start with K=10, experiment with K=5 and K=20

### Things to keep identical

- The `compute_mrr`, `compute_ndcg`, `compute_recall` math — these are correct standard implementations
- The `qrels` structure — this is the IR standard format
- The hybrid search pattern with `prefetch` — this is the right way to do hybrid in Qdrant

### Recommended order to build

```
1. Load JSON → print a few items, understand structure
2. Build query_list → just the text strings
3. Initialize BGE-M3 → embed one test query first
4. Connect Qdrant → test with client.get_collections()
5. Single query test → manually check hits for one query
6. Full loop → all 30 queries
7. MRR function → test on toy example before running
8. NDCG function → same
9. Recall functions → same
10. Language breakdown → last, after everything works
```
