# 🔬 RAG Evaluation Frameworks — SOTA Comparison Report

> **Context:** MechRabot needs to evaluate its hybrid retrieval (Dense + Sparse + ColBERT) and future LLM generation layer. This report compares the leading tools so we pick the right one for each evaluation phase.

---

## The Two Phases of RAG Evaluation

Before comparing tools, understand that RAG has **two completely different things to evaluate**:

```
Phase 1: RETRIEVAL EVALUATION          Phase 2: GENERATION EVALUATION
"Did we find the right chunks?"        "Did the LLM answer correctly?"
─────────────────────────────          ──────────────────────────────
Metrics:                               Metrics:
• MRR@10, NDCG@10, Recall@K           • Faithfulness (hallucination check)
• Context Precision                    • Answer Relevancy
• Context Recall                       • Correctness

No LLM needed to evaluate!            Needs LLM-as-a-judge!
Uses your ground truth labels.         Compares LLM output vs context.
```

**MechRabot right now:** You are in Phase 1. You don't have an LLM layer yet. Most of the tools below are built for Phase 2. Keep this distinction in mind.

---

## The 7 Major Tools at a Glance

| Tool | Type | Needs LLM to Evaluate? | Free? | Best For |
|---|---|---|---|---|
| **Ragas** | Metrics Library | ✅ Yes (LLM-as-judge) | ✅ Open Source | RAG-specific metric standard |
| **DeepEval** | Unit Testing | ✅ Yes (LLM-as-judge) | ✅ Open Source | CI/CD pipeline testing |
| **TruLens** | Feedback + Dashboard | ✅ Yes (feedback functions) | ✅ Open Source | Iterative experiment tracking |
| **Arize Phoenix** | Observability | ❌ Optional | ✅ Open Source | Visual debugging, embedding analysis |
| **LangSmith** | Tracing + Eval | ✅ Yes | ⚠️ Freemium | LangChain ecosystem deep tracing |
| **UpTrain** | Monitoring | ✅ Yes | ✅ Open Source | Production guardrails |
| **ARES** | Automated Eval | ⚠️ Uses trained judge | ✅ Open Source | Minimal labeled data evaluation |

---

## Detailed Comparison

### 1. 🏆 Ragas — The Industry Standard

**Philosophy:** Research-backed, transparent metrics for RAG.

**Core Metrics:**

| Metric | What It Measures | Phase |
|---|---|---|
| `Context Precision` | Are relevant chunks ranked higher than irrelevant ones? | Retrieval |
| `Context Recall` | Does the retrieved context contain ALL necessary info? | Retrieval |
| `Faithfulness` | Is the LLM answer grounded in the context (no hallucination)? | Generation |
| `Answer Relevancy` | Is the answer actually addressing the user's question? | Generation |

**How it works:**
```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

# Ragas needs this data structure:
dataset = {
    "question": ["What is the torque for cylinder head bolts?"],
    "contexts": [["Cylinder head bolts to 11 N·m..."]],  # retrieved chunks
    "answer": ["The torque is 11 N·m."],                  # LLM's response
    "ground_truth": ["11 N·m"]                            # your label
}

result = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness])
```

**Strengths:**
- ✅ Most widely cited in RAG research papers
- ✅ Framework-agnostic (works with anything, not locked to LangChain)
- ✅ Can generate synthetic test datasets when you don't have enough labeled data
- ✅ Metrics are transparent and explainable

**Weaknesses:**
- ❌ Uses LLM-as-judge → costs money per evaluation (OpenAI API calls)
- ❌ No built-in dashboard — evaluation only, not monitoring
- ❌ Retrieval metrics (precision/recall) use LLM judgment, NOT your ground truth chunk IDs directly

---

### 2. 🧪 DeepEval — Unit Testing for LLMs

**Philosophy:** Treat RAG evaluation like `pytest` — pass/fail assertions.

**Core Metrics:**

| Metric | What It Measures | Phase |
|---|---|---|
| `ContextualPrecisionMetric` | Relevant chunks ranked high? | Retrieval |
| `ContextualRecallMetric` | All needed info retrieved? | Retrieval |
| `ContextualRelevancyMetric` | Is retrieved context relevant to the query? | Retrieval |
| `FaithfulnessMetric` | Hallucination check | Generation |
| `AnswerRelevancyMetric` | Is answer on-topic? | Generation |
| `HallucinationMetric` | Direct hallucination detection | Generation |
| `ToxicityMetric` | Safety check | Safety |

**How it works:**
```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

test_case = LLMTestCase(
    input="What is the torque for cylinder head bolts?",
    actual_output="The torque is 11 N·m.",
    expected_output="11 N·m",
    retrieval_context=["Cylinder head bolts to 11 N·m..."]
)

metric = ContextualRecallMetric(threshold=0.7)
assert_test(test_case, [metric])  # PASSES or FAILS like pytest!
```

**Strengths:**
- ✅ Native `pytest` integration → runs in CI/CD pipelines
- ✅ 50+ metrics (largest library)
- ✅ Built-in red-teaming and safety metrics
- ✅ Best developer experience for test-driven development

**Weaknesses:**
- ❌ Heavy reliance on LLM-as-judge → expensive at scale
- ❌ More setup needed for dashboard visualization vs TruLens

---

### 3. 📊 TruLens — The Experiment Tracker

**Philosophy:** The "RAG Triad" — evaluate the triangle of (Query ↔ Context ↔ Answer).

**The RAG Triad:**
```
         Query
        /     \
       /       \
  Answer ──── Context
  Relevance    Relevance
       \       /
        \     /
      Groundedness
    (= Faithfulness)
```

**How it works:**
```python
from trulens_eval import TruChain, Feedback, OpenAI

# Define feedback functions
provider = OpenAI()
f_relevance = Feedback(provider.relevance).on_input_output()
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)

# Wrap your RAG chain
tru_recorder = TruChain(rag_chain, feedbacks=[f_relevance, f_groundedness])

# Every call is now tracked!
with tru_recorder as recording:
    answer = rag_chain.invoke("What is the torque?")

# Launch dashboard to visualize results
tru.run_dashboard()
```

**Strengths:**
- ✅ Beautiful visual dashboard for comparing experiments
- ✅ Best for A/B testing (baseline vs fine-tuned)
- ✅ Tracks performance over time across versions
- ✅ Native LangChain and LlamaIndex integration

**Weaknesses:**
- ❌ More suited for full-pipeline monitoring than isolated retrieval testing
- ❌ Requires a running LLM chain to evaluate (not just embeddings)

---

### 4. 🔍 Arize Phoenix — Visual Debugger

**Philosophy:** See your embeddings and traces visually.

**Strengths:**
- ✅ **Embedding visualizer** — plots your 1024-dim vectors in 2D/3D clusters
- ✅ Built on OpenTelemetry (vendor-neutral)
- ✅ Can visually show WHERE queries land relative to your chunks
- ✅ No LLM needed for embedding analysis!

**Weaknesses:**
- ❌ More observability tool than evaluation framework
- ❌ Doesn't compute MRR/NDCG/Recall natively

---

### 5. 🔗 LangSmith — LangChain's Native Tool

**Philosophy:** Deep tracing for LangChain apps.

**Strengths:**
- ✅ Best-in-class tracing for LangChain/LangGraph pipelines
- ✅ Annotation queues for human review
- ✅ Prompt playground

**Weaknesses:**
- ❌ **Framework lock-in** — designed for LangChain, awkward without it
- ❌ Freemium model (free tier limited)

---

### 6. 🛡️ UpTrain — Production Guardrails

**Strengths:**
- ✅ Pre-built checks for hallucination, toxicity, missing context
- ✅ Designed for production monitoring

**Weaknesses:**
- ❌ Less established community than Ragas/DeepEval
- ❌ LLM-dependent

---

### 7. 🧠 ARES — Minimal Labeling

**Philosophy:** Train a small "judge model" to evaluate so you don't need a big LLM.

**Strengths:**
- ✅ Works with very few labeled examples
- ✅ Cheaper than LLM-as-judge (uses a fine-tuned small model)

**Weaknesses:**
- ❌ More academic/research-oriented
- ❌ Requires training a judge model first

---

## The Similarity Matrix

How much do these tools overlap?

```
                Ragas  DeepEval  TruLens  Phoenix  LangSmith  UpTrain  ARES
Ragas            ██     🟡 70%   🟡 50%   🔵 20%   🔵 30%    🟡 60%   🟡 55%
DeepEval              ██        🟡 50%   🔵 20%   🔵 30%    🟡 55%   🟡 40%
TruLens                          ██      🟡 40%   🟡 60%    🟡 50%   🔵 30%
Phoenix                                   ██      🟡 50%    🔵 20%   🔵 15%
LangSmith                                          ██       🔵 30%   🔵 20%
UpTrain                                                      ██      🟡 40%
ARES                                                                  ██
```

- 🟡 = Significant overlap (can replace each other for some tasks)
- 🔵 = Minimal overlap (complementary tools)

**Key insight:** Ragas and DeepEval have **70% overlap** — they measure almost the same things, just with different developer experiences. You typically pick ONE of them, not both.

---

## What Each Metric Actually Measures (Unified View)

| Concept | Ragas Name | DeepEval Name | TruLens Name | Phase |
|---|---|---|---|---|
| "Right chunks found?" | Context Precision | ContextualPrecision | Context Relevance | Retrieval |
| "All chunks found?" | Context Recall | ContextualRecall | Context Relevance | Retrieval |
| "No hallucination?" | Faithfulness | Faithfulness | Groundedness | Generation |
| "Answer on-topic?" | Answer Relevancy | AnswerRelevancy | Answer Relevance | Generation |
| "Answer correct?" | Answer Correctness | GEval (custom) | — | Generation |

They're measuring the **same things** with different names and slightly different math.

---

## 🎯 Recommendation for MechRabot

### Right Now (Phase 1 — Retrieval Only)

You don't have an LLM layer yet. You only need to evaluate **retrieval quality**.

**None of these tools are needed yet!** 

Your simplest evaluation stack is:
- **`pytrec_eval`** or **manual Python** → Computes MRR@10, NDCG@10, Recall@K
- **Your `evaluation_30_V1.json`** → The ground truth
- **No LLM costs** → Just pure math on vectors

```python
# This is all you need for Phase 1:
from collections import defaultdict

def mrr_at_k(qrels, results, k=10):
    """Mean Reciprocal Rank — no LLM needed!"""
    mrr_sum = 0
    for qid, relevant_docs in qrels.items():
        for rank, (doc_id, score) in enumerate(
            sorted(results[qid].items(), key=lambda x: -x[1])[:k], 1
        ):
            if doc_id in relevant_docs:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / len(qrels)
```

### After Adding the LLM Layer (Phase 2)

When you add an LLM to generate answers from retrieved chunks:

| Tool | Use it for | Why |
|---|---|---|
| **Ragas** | One-time deep evaluation of the full pipeline | Industry standard, transparent, framework-agnostic |
| **DeepEval** | If you set up CI/CD testing | pytest integration is excellent |
| **Arize Phoenix** | Visual debugging of embedding clusters | Free, no LLM cost, great for seeing Arabic vs English cluster separation |

### The Production Stack (Phase 3)

```
Ragas          →  Offline evaluation (before deployment)
Arize Phoenix  →  Visual debugging (embedding analysis)
TruLens        →  Production monitoring (after deployment)
```

---

## Summary Decision Table

| Question | Answer |
|---|---|
| "Which one should I learn first?" | **Ragas** — it's the standard everyone references |
| "Which one is simplest?" | **Manual Python with pytrec_eval** for retrieval-only |
| "Which one needs no LLM budget?" | **Arize Phoenix** (visual) or **manual metrics** (math) |
| "Which one is best for my evaluation_30_V1.json?" | **Manual Python** — you already have labeled chunk IDs |
| "Which one for production monitoring?" | **TruLens** or **Arize Phoenix** |
| "Can I use multiple?" | Yes — teams use 2-3 tools for different purposes |
