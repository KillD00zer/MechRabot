# 🔍 The Complete Retrieval Paradigms Guide

## Dense · Sparse · ColBERT · Hybrid — and MechRabot Recommendations

---

## Table of Contents

1. [The Core Problem: What Are We Retrieving?](#1-the-core-problem)
2. [Paradigm 1: Sparse Retrieval (BM25 / TF-IDF)](#2-sparse-retrieval)
3. [Paradigm 2: Dense Retrieval (Bi-Encoders)](#3-dense-retrieval)
4. [Paradigm 3: ColBERT — Multi-Vector Late Interaction](#4-colbert-multi-vector)
5. [Hybrid Retrieval — Fusing All Three](#5-hybrid-retrieval)
6. [BGE-M3: The Unified Three-in-One Model](#6-bge-m3-unified)
7. [Decision Matrix: When to Use What](#7-decision-matrix)
8. [MechRabot-Specific Strategy](#8-mechrabot-strategy)
9. [Code Examples for Qdrant](#9-qdrant-code)

---

## 1. The Core Problem

Before any model architecture, understand the core retrieval question:

> **Given a user query `q`, find the top-K chunks from a corpus `C` that are most relevant.**

The three paradigms answer this with fundamentally different mathematical strategies:

| Paradigm    | Core Math                          |    Speed     |     Quality     |
| :---------- | :--------------------------------- | :----------: | :-------------: |
| **Sparse**  | Exact term overlap (TF-IDF / BM25) |    ⚡⚡⚡    |     Medium      |
| **Dense**   | Single vector cosine similarity    |     ⚡⚡     | High (semantic) |
| **ColBERT** | Token-level MaxSim over matrix     | ⚡ (re-rank) |     Highest     |

---

## 2. Sparse Retrieval (BM25 / TF-IDF)

### What it is

Sparse retrieval represents text as a **high-dimensional, mostly-zero vector** where each dimension is a vocabulary term. The "sparse" name comes from the fact that most dimensions are zero — a sentence only activates the ~20 vocab tokens it actually uses out of a 50,000-term vocabulary.

### The Math: BM25 Score

```
BM25(q, d) = Σᵢ IDF(tᵢ) · [ f(tᵢ,d) · (k₁+1) ] / [ f(tᵢ,d) + k₁·(1 - b + b·|d|/avgdl) ]
```

Where:

- `IDF(tᵢ)` = Inverse Document Frequency — rare terms score higher
- `f(tᵢ, d)` = term frequency in document `d`
- `k₁` = saturation constant (~1.2–2.0), diminishing returns on repeated terms
- `b` = length normalization (~0.75), penalizes long documents
- `avgdl` = average document length in corpus

### How It Works Step-by-Step

```
Query: "timing belt torque 45 Nm"
         ↓
1. Tokenize → ["timing", "belt", "torque", "45", "nm"]
2. Compute IDF for each token from corpus statistics
3. For each document: count token overlaps, apply BM25 formula
4. Rank by score — exact matches WIN
```

### Strengths 💪

- **Exact term matching**: `10.5 Nm` ≠ `10 Nm` — perfect for safety specs
- **Zero training required**: purely statistical, works on day-1 data
- **Blazing fast**: inverted index lookup is O(log N)
- **Interpretable**: you can explain _why_ a document ranked high
- **Part numbers**: `OEM-44305-06050` matched exactly, no ambiguity

### Weaknesses 😓

- **Vocabulary mismatch**: "engine head gasket" ≠ "head sealing component"
- **No semantics**: cannot infer meaning, only surface form
- **Language boundary**: Arabic query misses English document unless same word appears
- **Synonyms fail**: "oil" ≠ "lubricant" unless both are in the text

### Use Cases ✅

- Legal document retrieval (exact clause numbers)
- Medical records (ICD codes, drug names)
- **Safety-critical specs** (torque values, tolerances)
- Part number and code lookups
- FAQ systems with templated answers

---

## 3. Dense Retrieval (Bi-Encoders)

### What it is

Dense retrieval encodes **both query and document into a single fixed-size vector** (typically 768–1024 dimensions) using a transformer encoder. Similarity is computed as the cosine or dot product between these two vectors.

The "dense" name: all ~1024 dimensions carry meaningful information simultaneously.

### Architecture: The Bi-Encoder

```
                ┌──────────────┐
Query ──────►   │  Transformer │  ──► q_vec  [1024d]
                │  (shared or  │              ╲
                │   separate)  │               cosine_sim(q_vec, d_vec) → score
                │              │              ╱
Document ──►    │              │  ──► d_vec  [1024d]
                └──────────────┘
```

The key property: **documents are pre-encoded offline and stored as vectors**. At query time, only the query is encoded, then ANN (Approximate Nearest Neighbor) search finds the closest document vectors in milliseconds.

### The Math

```python
score = cosine_similarity(encode(query), encode(document))
      = (q_vec · d_vec) / (|q_vec| · |d_vec|)
```

### Training: Contrastive Learning

Dense models are trained to pull **positive pairs** together and push **negatives** apart in vector space:

```python
# Contrastive loss (InfoNCE / in-batch negatives)
loss = -log( exp(sim(q, pos)) / Σⱼ exp(sim(q, neg_j)) )
```

Training data format:

```json
{
  "query": "timing belt replacement",
  "positive": "Replace timing belt every 60,000 km...",
  "negative": "The fuel pump is located..."
}
```

### The Semantic Leap

```
User Query: "سير كاتينة"  (Egyptian slang for timing belt)
              ↓
         [BGE-M3 Dense Encoder]
              ↓
         q_vec: [0.23, -0.71, 0.45, ...]  (1024d)
              ↓  cosine search
         d_vec: [0.21, -0.69, 0.47, ...]  ← "timing belt synchronizes crankshaft"
              ↓
         score: 0.94  ← HIGH MATCH! (if fine-tuned)
```

This is the **semantic bridge** — the model maps conceptually equivalent content close in vector space, regardless of language or surface form.

### ANN Index: How It Scales

Computing exact cosine similarity against millions of vectors is O(N). Instead:

| Algorithm                 | Strategy                                 | Trade-off            |
| :------------------------ | :--------------------------------------- | :------------------- |
| **HNSW** (Qdrant default) | Hierarchical Navigable Small World graph | 99%+ recall, fast    |
| **IVF** (FAISS)           | Cluster vectors, search nearest clusters | Tunable speed/recall |
| **PQ**                    | Product Quantization — compress vectors  | Memory efficient     |

### Strengths 💪

- **Semantic understanding**: synonyms, paraphrases, concept mapping
- **Cross-lingual**: multilingual models (BGE-M3) bridge Arabic ↔ English naturally
- **ANN scales to billions**: sub-millisecond search at scale
- **Context-aware**: entire sentence meaning captured, not just keywords

### Weaknesses 😓

- **Exact match blind spot**: `10.5 Nm` and `10 Nm` can have very similar dense vectors
- **Training dependency**: quality degrades on out-of-domain data
- **Black box**: hard to explain _why_ a document was ranked high
- **Embedding drift on rare terms**: OEM part codes may not generalize well

### Use Cases ✅

- Semantic Q&A systems
- Cross-lingual retrieval
- Concept search ("find all procedures related to engine cooling")
- Customer support (paraphrase-tolerant)
- **MechRabot**: Arabic slang → English manual mapping

---

## 4. ColBERT — Multi-Vector Late Interaction

### What it is

ColBERT (**Col**umnar **BERT**) is the most sophisticated paradigm. Instead of compressing the full document into _one_ vector, ColBERT generates **one vector per token**. This preserves token-level granularity, enabling more precise matching.

### Architecture: Late Interaction

```
                 ┌─────────────────────────────┐
Query:  "What   is   timing   belt   torque"   │
         ↓       ↓     ↓        ↓      ↓       │
        [q1]   [q2]  [q3]    [q4]   [q5]       │   → Query Matrix Q [5 × 128d]
                                                │
Document: "Timing   belt   installation   requires   45 Nm   torque"
             ↓        ↓         ↓            ↓        ↓       ↓
           [d1]     [d2]      [d3]          [d4]    [d5]    [d6]  → Doc Matrix D [6 × 128d]
                                                │
                           MaxSim(Q, D)         │
                           ─────────────────────┘
                           For each query token qᵢ:
                             find max similarity to ANY document token dⱼ
                           Sum all per-token max scores → final relevance
```

### The MaxSim Formula

```
ColBERT_score(q, d) = Σᵢ max_j ( qᵢ · dⱼᵀ )
```

This means:

- Each query token **independently** finds its best matching document token
- Even if "timing" appears in position 5 of the document, it will be found by the "timing" query token
- Result: **soft, positional-invariant, exact-ish matching**

### Why This Outperforms Both Dense and Sparse

| Scenario                                      | Sparse | Dense | ColBERT |
| :-------------------------------------------- | :----: | :---: | :-----: |
| Exact keyword match                           |   ✅   |  ❌   |   ✅    |
| Semantic synonym match                        |   ❌   |  ✅   |   ✅    |
| Part-of-word match ("Nm" ↔ "torque Nm value") |   ❌   |  🟡   |   ✅    |
| Query token gets lost in dense compression    |  N/A   |  ❌   |   ✅    |
| Arabic-English code switching in same query   |   ❌   |  🟡   |   ✅    |

### Storage Consideration: The ColBERT Tax

Every document token becomes a vector. For a 256-token chunk:

- **Dense**: 1 × 1024 floats = **4 KB per chunk**
- **ColBERT** (128d per token): 256 × 128 floats = **131 KB per chunk**

For 2,790 MechRabot chunks: Dense = ~11 MB vs ColBERT = ~366 MB. Manageable, but significant.

### Two Deployment Modes

#### Mode A: Full ColBERT (Index + Search)

Indexes all document token vectors. Used in `PLAID` engine. Best recall, highest storage.

#### Mode B: ColBERT as Re-ranker (Most Common)

```
1. Retrieve top-100 candidates with Dense (fast ANN)
2. Re-rank top-100 using ColBERT MaxSim (precise, small set)
3. Return top-10 final results
```

This is the **dominant production pattern** — get speed from dense, get precision from ColBERT.

### Strengths 💪

- **Best MRR/NDCG** on most benchmarks (BEIR, MTEB)
- **No information collapse**: every token retains its own signal
- **Handles abbreviations**: "N·m" token matched to context perfectly
- **Part codes**: `44305-06050` — each digit segment is a token with its own vector
- **Code-switching**: Arabic-English mixed queries handled at token level

### Weaknesses 😓

- **Storage**: 30–100× more than dense
- **Raw search is slow**: must compute MaxSim during query, not precomputed cosine
- **Typically used as re-ranker**, not first-stage retriever
- **Training complexity**: requires curated positive/negative pairs

### Use Cases ✅

- Re-ranking top-K dense results for maximum precision
- Technical documentation retrieval
- **Safety-critical specs** where token-level precision matters
- **MechRabot**: re-ranking candidates before presenting torque specs

---

## 5. Hybrid Retrieval — Fusing All Three

### Why Hybrid?

No single paradigm is universally best. The sweet spot is **fusing multiple signals**:

```
Query: "سير كاتينة torque 45 nm"

Sparse gives: chunks with exact "45 nm"          → precision ✅
Dense gives:  chunks about timing belt in Arabic  → recall ✅
ColBERT gives: token-level verification of specs  → safety ✅
```

### Fusion Strategy 1: Reciprocal Rank Fusion (RRF)

RRF is the most common, robust, and parameter-free fusion method:

```
RRF_score(d) = Σₛ 1 / (k + rank_s(d))
```

Where:

- `rank_s(d)` = the rank of document `d` in retrieval system `s`
- `k` = smoothing constant (typically 60)
- Sum is over all retrieval systems (dense, sparse, colbert)

**Example:**

| Chunk                                    | Dense Rank | Sparse Rank |            RRF Score            |
| :--------------------------------------- | :--------: | :---------: | :-----------------------------: |
| "Timing belt torque 45 Nm"               |     3      |      1      | 1/(60+3) + 1/(60+1) =**0.032**  |
| "Timing belt synchronizes crankshaft..." |     1      |      8      | 1/(60+1) + 1/(60+8) =**0.031**  |
| "Fuel pump pressure 3.5 bar"             |     5      |     12      | 1/(60+5) + 1/(60+12) =**0.015** |

RRF naturally promotes documents that **rank well in multiple systems**.

### Fusion Strategy 2: Weighted Linear Score Combination

```
final_score = α · dense_score + β · sparse_score + γ · colbert_score
```

Requires calibration:

- `α + β + γ = 1`
- Tune per query type (spec queries → boost β, semantic queries → boost α)

**Problem**: Dense and sparse scores are on different scales. BM25 scores can be 0–20+, cosine similarity is 0–1. Requires normalization first.

### Fusion Strategy 3: Query-Type Detection → Dynamic Weighting

The most sophisticated approach — detect query intent and adapt weights:

```python
def adaptive_fusion(query):
    if contains_number(query) or contains_part_code(query):
        # Safety-critical: boost sparse + colbert
        return alpha=0.3, beta=0.5, gamma=0.2
    elif is_arabic(query):
        # Cross-lingual: boost dense
        return alpha=0.7, beta=0.1, gamma=0.2
    else:
        # General semantic query
        return alpha=0.5, beta=0.3, gamma=0.2
```

### Qdrant's Native Hybrid: Prefetch + Fusion

Qdrant has first-class hybrid search support:

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="mechrabot_v2",
    prefetch=[
        # Stage 1: Dense retrieval (top-50)
        Prefetch(
            query=dense_vec,          # [1024d] float vector
            using="dense",
            limit=50
        ),
        # Stage 1: Sparse retrieval (top-50)
        Prefetch(
            query=SparseVector(       # BM25-style sparse
                indices=[23, 445, 7890],
                values=[0.8, 0.6, 0.4]
            ),
            using="sparse",
            limit=50
        ),
    ],
    # Stage 2: Fuse results
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    with_payload=True
)
```

---

## 6. BGE-M3: The Unified Three-in-One Model

BGE-M3 is uniquely positioned because it outputs **all three vector types from a single forward pass**. This is architecturally novel.

### Internal Architecture

```
Input Text: "timing belt torque"
               ↓
    [XLM-RoBERTa Backbone] (105 languages, 8192 token context)
               ↓
    Hidden States: [T × 1024]  (T = number of tokens)
               ↓
    ┌──────────────────────────────────────────────────────┐
    │                 Three Output Heads                   │
    │                                                      │
    │  [CLS] token →  Dense Head  →  dense_vec  [1024d]  │
    │                                                      │
    │  SPLADE Head →  Sparse Vec →  sparse_vec [30522d]  │  (vocab-sized, sparse)
    │  (term weights from                                  │
    │   vocab projection)                                  │
    │                                                      │
    │  All tokens  →  ColBERT Head → token_vecs [T × 128d]│
    └──────────────────────────────────────────────────────┘
```

### Unified Training Loss

BGE-M3 was trained with a **multi-objective unified loss**:

```
L_total = L_dense + L_sparse + L_colbert

where each Lₓ = InfoNCE contrastive loss for that head
```

This forces the backbone to learn representations that are simultaneously good for all three tasks — a powerful inductive bias.

### The SPLADE Sparse Head (How BGE-M3 Sparse Works)

Unlike BM25 (which is statistical), BGE-M3's sparse head is **neural**:

```
sparse_weight(term_t) = ReLU(log(1 + W·hidden_state))
```

Result: The model can assign weight to **terms NOT in the original text** if they are contextually implied. This is "learned BM25" — it understands "oil" and "lubricant" as the same index term.

### BGE-M3 in Practice

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Single call → all three vector types
output = model.encode(
    ["timing belt torque 45 Nm"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
    max_length=512
)

output['dense_vecs']       # shape: [1, 1024]
output['lexical_weights']  # dict: {"timing": 0.8, "belt": 0.7, ...}
output['colbert_vecs']     # shape: [1, T, 128]
```

### Performance vs. Specialist Models

| Model              | Dense MTEB | Sparse MAP | ColBERT nDCG |  Params  |
| :----------------- | :--------: | :--------: | :----------: | :------: |
| BM25 (statistical) |     —      |    23.1    |      —       |    0     |
| E5-large-v2        |    64.2    |     —      |      —       |   335M   |
| SPLADE++           |     —      |    29.8    |      —       |   110M   |
| ColBERT v2         |     —      |     —      |     69.8     |   110M   |
| **BGE-M3**         |  **64.0**  |  **28.3**  |   **65.1**   | **568M** |

BGE-M3 is not the #1 specialist in any single category but is **top-tier in all three simultaneously**.

---

## 7. Decision Matrix: When to Use What

```
                        FULL DECISION MATRIX
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Query Type               Sparse  Dense  ColBERT  Hybrid           │
│  ─────────────────────────────────────────────────────────────────  │
│  "10.5 Nm torque"           ★★★    ★☆☆    ★★☆      ★★★            │  ← NUMBERS
│  "timing belt procedure"    ★☆☆    ★★★    ★★☆      ★★★            │  ← CONCEPTS
│  "سير كاتينة" (Arabic)     ★☆☆    ★★★    ★★☆      ★★★            │  ← CROSS-LINGUAL
│  "OEM-44305-06050"          ★★★    ★☆☆    ★★★      ★★★            │  ← PART CODES
│  "head gasket sealing..."   ★☆☆    ★★★    ★★★      ★★★            │  ← DIAGNOSTIC
│  "replace سير & check Nm"  ★★☆    ★★☆    ★★★      ★★★            │  ← CODE-SWITCH
│  ─────────────────────────────────────────────────────────────────  │
│  Conclusion: Hybrid with adaptive weights ALWAYS wins              │
└─────────────────────────────────────────────────────────────────────┘
```

### Storage Comparison (MechRabot scale: 2,790 chunks)

| Vector Type              |     Dimensions     | Size per chunk | Total (2,790 chunks) |
| :----------------------- | :----------------: | :------------: | :------------------: |
| Dense                    |       1,024        |     ~4 KB      |        ~11 MB        |
| Sparse (BGE-M3)          | ~30K (but sparse!) |   ~0.5–2 KB    |        ~5 MB         |
| ColBERT (avg 150 tokens) |     150 × 128      |     ~77 KB     |       ~215 MB        |

**MechRabot is small enough to store ALL THREE** in Qdrant without issue.

---

## 8. MechRabot-Specific Strategy

### The Problem Space Mapping

MechRabot has **five distinct query archetypes**, each requiring different retrieval emphasis:

#### Archetype 1: 🔴 Safety-Critical Spec Query

```
User: "What is the cylinder head bolt torque?"
Arabic: "عزم ربط برغي رأس السيلندر كام؟"
```

**Risk**: Dense retrieval alone may return "68 Nm" instead of "65 Nm" — catastrophic

**Strategy**: `Sparse FIRST (0.6) + Dense (0.3) + ColBERT re-rank (final check)`

---

#### Archetype 2: 🟡 Procedural/Diagnostic Query

```
User: "How do I replace the timing belt on this engine?"
Arabic: "ازاي أغير سير الكاتينة؟"
```

**Risk**: None safety-critical, need full procedure steps

**Strategy**: `Dense FIRST (0.6) + Sparse (0.2) + ColBERT re-rank + chunk linked-list fetch`

The **linked-list chain** (`previous_chunk_id` / `next_chunk_id`) is the killer feature here — once you find step 3 of a procedure, fetch steps 1-5 automatically.

---

#### Archetype 3: 🟢 Cross-Lingual Concept Query

```
User: "سير كاتينة" or "جلبة مقص" or "بلوف مكيف"
```

**Risk**: BM25 returns zero results (no word overlap with English manual)

**Strategy**: `Dense DOMINANT (0.8) + ColBERT re-rank (0.2)` — fine-tuned BGE-M3 handles this natively

---

#### Archetype 4: 🔵 Part Code / OEM Number Query

```
User: "OEM 44305-06050 specs"
```

**Risk**: Dense embedding may generalize nearby part codes

**Strategy**: `Sparse DOMINANT (0.7) + ColBERT token match (0.3)` — exact token matching is essential

---

#### Archetype 5: 🟣 Code-Switching Mixed Query (Arabic + English)

```
User: "سير كاتينة timing belt 45 Nm replace procedure"
```

**Risk**: None of the pure approaches handles ALL signals

**Strategy**: `Full Hybrid Equal (Dense 0.4 + Sparse 0.35 + ColBERT 0.25)` + query language detection

---

### Recommended Qdrant Collection Schema

```python
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    SparseIndexParams
)

client.create_collection(
    collection_name="mechrabot_v3",
    vectors_config={
        # Dense: semantic / cross-lingual
        "dense": VectorParams(size=1024, distance=Distance.COSINE),
        # ColBERT: token-level precision (multi-vector)
        "colbert": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        ),
    },
    sparse_vectors_config={
        # Sparse: keyword / number exact matching
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)
```

### Adaptive Query Router

```python
import re
from dataclasses import dataclass
from qdrant_client.models import Prefetch, FusionQuery, Fusion

@dataclass
class QueryWeights:
    dense: float
    sparse: float
    colbert_rerank: bool

def detect_query_type(query: str) -> QueryWeights:
    has_number = bool(re.search(r'\d+\.?\d*\s*(nm|bar|°c|mm|kg)', query, re.I))
    has_oem_code = bool(re.search(r'[A-Z]{2,}-\d{4,}', query))
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    has_arabic_and_english = is_arabic and bool(re.search(r'[a-zA-Z]', query))

    if has_number or has_oem_code:
        return QueryWeights(dense=0.25, sparse=0.55, colbert_rerank=True)
    elif has_arabic_and_english:
        return QueryWeights(dense=0.45, sparse=0.3, colbert_rerank=True)
    elif is_arabic:
        return QueryWeights(dense=0.75, sparse=0.1, colbert_rerank=True)
    else:
        return QueryWeights(dense=0.55, sparse=0.3, colbert_rerank=True)


def mechrabot_hybrid_search(
    client, model, query: str,
    section_filter=None, top_k=10
):
    weights = detect_query_type(query)

    # Encode query with BGE-M3 (all three in one call)
    encoded = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )

    dense_vec = encoded['dense_vecs'][0].tolist()
    sparse_vec = encoded['lexical_weights'][0]   # dict {token_id: weight}
    colbert_vecs = encoded['colbert_vecs'][0]    # [T, 128]

    # Build payload filter for section_path narrowing
    query_filter = None
    if section_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(
                key="meta.section_path",
                match=MatchValue(value=section_filter)
            )]
        )

    # Prefetch: dense + sparse candidates
    prefetch = [
        Prefetch(
            query=dense_vec,
            using="dense",
            limit=50,
            filter=query_filter
        ),
        Prefetch(
            query=SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values())
            ),
            using="sparse",
            limit=50,
            filter=query_filter
        ),
    ]

    # First stage: RRF fusion of dense + sparse → top 30
    stage1_results = client.query_points(
        collection_name="mechrabot_v3",
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=30,
        with_payload=True
    )

    # Second stage: ColBERT re-rank (if enabled)
    if weights.colbert_rerank:
        reranked = colbert_rerank(stage1_results, colbert_vecs, model)
        return reranked[:top_k]

    return stage1_results[:top_k]
```

### The Linked-List Post-Retrieval Expansion

After finding the best chunk, expand context via the linked list:

```python
def expand_with_context(client, chunk, window=1):
    """Fetch neighboring chunks without an extra vector search"""
    neighbors = []

    prev_id = chunk.payload['meta'].get('previous_chunk_id')
    next_id = chunk.payload['meta'].get('next_chunk_id')

    if prev_id:
        prev = client.retrieve("mechrabot_v3", ids=[prev_id], with_payload=True)
        neighbors.extend(prev)

    neighbors.append(chunk)

    if next_id:
        nxt = client.retrieve("mechrabot_v3", ids=[next_id], with_payload=True)
        neighbors.extend(nxt)

    return neighbors
```

### Fine-Tuning Impact on Retrieval Modes

| Fine-Tuning Goal               | Affected Vector             | Expected Gain                           |
| :----------------------------- | :-------------------------- | :-------------------------------------- |
| Arabic slang → English bridge  | **Dense** (CLS)             | +30–50% recall on Arabic queries        |
| Part code recognition          | **Sparse** + **ColBERT**    | Better token weights for OEM codes      |
| Spec number precision          | **Sparse** (SPLADE weights) | Stronger IDF for "Nm", "bar" etc.       |
| Section context (section_path) | **Dense**                   | Better hierarchical query understanding |

Use `FlagEmbedding` unified fine-tuning so **all three heads benefit** from your training data simultaneously.

---

## 9. Qdrant Code Examples

### Upsert All Three Vectors per Chunk

```python
from qdrant_client.models import PointStruct, SparseVector

def upsert_mechrabot_chunk(client, model, chunk: dict):
    text = chunk['content']

    # Single BGE-M3 call → all vectors
    encoded = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
        max_length=512
    )

    point = PointStruct(
        id=chunk['chunk_id'],
        vector={
            "dense": encoded['dense_vecs'][0].tolist(),
            "colbert": encoded['colbert_vecs'][0].tolist(),  # [[128d], [128d], ...]
        },
        # Sparse goes separately
        payload=chunk['meta']
    )

    # Qdrant requires sparse vectors in a separate call or via named vector syntax
    client.upsert(
        collection_name="mechrabot_v3",
        points=[point]
    )

    # Update sparse separately (current Qdrant API)
    sparse_indices = list(encoded['lexical_weights'][0].keys())
    sparse_values  = list(encoded['lexical_weights'][0].values())

    client.update_vectors(
        collection_name="mechrabot_v3",
        points=[
            PointVectors(
                id=chunk['chunk_id'],
                vector={"sparse": SparseVector(indices=sparse_indices, values=sparse_values)}
            )
        ]
    )
```

### Full Hybrid Query with Section Filter

```python
# Example: Find cylinder head bolt torque in ENGINE section
results = mechrabot_hybrid_search(
    client=client,
    model=bge_m3_model,
    query="عزم ربط برغي رأس السيلندر",  # Arabic query
    section_filter="ENGINE",
    top_k=5
)

for r in results:
    print(f"[{r.score:.3f}] {r.payload['content'][:100]}...")
    # Expand to include neighboring procedure steps
    context_chunks = expand_with_context(client, r, window=1)
```

---

## Summary: MechRabot's Retrieval Advantage

Your project already has the **correct architecture** — BGE-M3's three-in-one output is the exact right tool. The refinement is in:

1. **Adaptive query routing**: Detect numbers/Arabic/codes → shift weights dynamically
2. **ColBERT as final re-ranker**: Run MaxSim on top-30 RRF results before returning to LLM
3. **Linked-list expansion**: Always fetch `prev + chunk + next` for procedural queries
4. **Section-path pre-filter**: Narrow vector search to the relevant section BEFORE RRF fusion (reduces noise, speeds up search)
5. **Fine-tuning all three heads**: Use FlagEmbedding unified training — your Egyptian slang pairs improve the Dense head, which in turn improves the ColBERT token representations from the same backbone

The combination of these five strategies positions MechRabot at the frontier of domain-specific RAG systems.

---

_Guide generated for MechRabot project — 2026-03-29_
