# 📊 MechRabot Evaluation Dataset — V1

> **Goal**: Measure the retrieval quality of BGE-M3 (via FlagEmbedding) on the Chery A3 mechanical corpus before and after fine-tuning on Egyptian mechanical slang.

---

## What This Dataset Is

`evaluation_30_V1.json` is the **labeled ground truth** (qrels) for offline retrieval evaluation.

It contains **30 queries** paired with the **real chunk IDs** from `final_chunks_v2.json` (2,790 chunks) that correctly answer each query.

The evaluator never sees the expected answer text — it only uses the `relevant_chunk_ids` to score whether the retrieval system returned those chunks near the top of its ranked list.

---

## Dataset Structure

```json
{
  "metadata": { ... },
  "queries": [
    {
      "id": "EV-001",
      "language": "en | ar | ar_slang",
      "category": "spec | procedural | diagnostic | diagnostic_electrical",
      "topic": "descriptive topic name",
      "difficulty": "easy | medium | hard",
      "query": "the question text",
      "query_translation": "English gloss (Arabic queries only)",
      "expected_answer": "human-readable correct answer",
      "relevant_chunk_ids": ["full-uuid-from-corpus"],
      "notes": "what this query is specifically testing"
    }
  ]
}
```

---

## Query Distribution

### By Language

| Language | Code | Count | Purpose |
|---|---|---|---|
| English | `en` | 20 | Baseline — model should perform best here |
| Formal Arabic (MSA) | `ar` | 3 | Cross-lingual dense retrieval test |
| Egyptian Colloquial | `ar_slang` | 7 | Core fine-tuning validation target |

### By Category

| Category | Count | Block |
|---|---|---|
| `spec` — English specs | 10 | Block A (EV-001 → EV-010) |
| `procedural` + `diagnostic` — English | 6 | Block B (EV-011 → EV-016) |
| `diagnostic_electrical` — English | 4 | Block B (EV-017 → EV-020) |
| `spec` — Arabic | 5 | Block C (EV-021 → EV-025) |
| `procedural` + `diagnostic` — Arabic | 5 | Block D (EV-026 → EV-030) |

### By Difficulty

| Difficulty | Count | What makes it hard |
|---|---|---|
| `easy` | 11 | Query terms appear verbatim in the chunk |
| `medium` | 14 | Paraphrasing required, or answer spans multiple chunks |
| `hard` | 5 | Egyptian slang with semantic drift, or ambiguous terminology |

---

## Topic Coverage

The 30 queries span **8 mechanical topics** and **1 electrical section**:

| Topic Group | Queries |
|---|---|
| Engine oil pressure | EV-001, EV-002, EV-021 |
| Compression pressure | EV-003 |
| Cylinder head cover | EV-004, EV-014, EV-022, EV-028 |
| Crankshaft & timing belt | EV-005, EV-010, EV-023 |
| Throttle body | EV-006, EV-024 |
| Exhaust manifold | EV-007 |
| Drive belt & pulleys | EV-008, EV-009, EV-011, EV-025 |
| Air cleaner | EV-012, EV-026 |
| Cylinder head cover removal (procedure) | EV-013, EV-027 |
| Engine no-start diagnostics | EV-015, EV-029 |
| CO exhaust hazard (safety) | EV-016 |
| DTC overview | EV-017 |
| Intermittent DTC | EV-018 |
| MAF sensor DTC (P0103) | EV-019, EV-030 |
| Misfire DTC (P0302-304) | EV-020 |

---

## Evaluation Metrics

### MRR@10 — Mean Reciprocal Rank
**Question it answers**: *"How early does the first correct chunk appear?"*

```
RR per query = 1 / rank_of_first_relevant_chunk
MRR@10 = average of RR across all 30 queries
```

- Score of **1.0** → correct chunk is always rank #1
- Score of **0.5** → correct chunk is usually around rank #2
- A chunk ranked **below #10** contributes **0.0**

Best for catching: *the model returns the wrong chunk at the very top.*

---

### NDCG@10 — Normalized Discounted Cumulative Gain
**Question it answers**: *"Is the overall ranking order correct, especially for multi-chunk queries?"*

```
DCG@10   = sum of (relevance / log2(rank+1)) for positions 1..10
NDCG@10  = DCG@10 / ideal_DCG@10
```

- Used for queries with **multiple relevant chunks** (e.g., EV-014, EV-028)
- A chunk at rank #1 contributes **much more** than same chunk at rank #5
- Result is always between 0 and 1

Best for catching: *model finds relevant chunks but buries them below irrelevant ones.*

---

### Recall@K
**Question it answers**: *"Did we even find ALL the relevant chunks?"*

```
Recall@K = (relevant chunks found in Top-K) / (total relevant chunks)
```

| K | Interpretation |
|---|---|
| Recall@5  | Are most answers in the 5 results shown to the LLM? |
| Recall@10 | Safety net — did the retriever at least find them? |
| Recall@50 | Upper-bound check for the corpus |

Best for catching: *model consistently misses a relevant chunk entirely.*

---

## Interpreting Results

### Target Thresholds (pre-fine-tuning baseline)

| Metric | 🟡 Acceptable | 🟢 Good | 🔵 Excellent |
|---|---|---|---|
| MRR@10 | > 0.55 | > 0.70 | > 0.85 |
| NDCG@10 | > 0.50 | > 0.65 | > 0.80 |
| Recall@10 | > 0.65 | > 0.80 | > 0.90 |

### Expected Pattern Per Language

| Language | Expected Score (pre-tuning) | Reason |
|---|---|---|
| English | Highest | BGE-M3 is trained heavily on English |
| Formal Arabic (MSA) | Medium | BGE-M3 covers Arabic in MIRACL |
| Egyptian Slang | Lowest → should rise after fine-tuning | Out-of-distribution for base model |

**Key insight**: If English scores are high but Arabic slang scores are low, fine-tuning is working correctly when Arabic scores improve without English dropping.

---

## How to Convert This File to `qrels` Format

The evaluation framework (BEIR / FlagEmbedding) needs the data reshaped into two dicts:

```python
import json

with open("evaluation_30_V1.json") as f:
    data = json.load(f)

# queries dict
queries = {
    item["id"]: item["query"]
    for item in data["queries"]
}

# qrels dict — the actual ground truth
qrels = {
    item["id"]: {cid: 1 for cid in item["relevant_chunk_ids"]}
    for item in data["queries"]
    if "relevant_chunk_ids" in item
}
```

**Output shape:**
```python
queries = {"EV-001": "What is the engine oil pressure at idle speed?", ...}
qrels   = {"EV-001": {"e53006fd-7590-c882-6ba5-fef1923f37cf": 1}, ...}
```

---

## Files in This Directory

| File | Description |
|---|---|
| `evaluation_30_V1.json` | The full labeled evaluation dataset (this file) |
| `eval_dataset.json` | Earlier draft — kept for reference |
| `README.md` | This file |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| V1 | 2026-04-03 | 30 queries with real chunk IDs from `final_chunks_v2.json`. Stratified by language, topic, and difficulty. |
