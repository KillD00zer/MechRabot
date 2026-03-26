# Open-Source Tech Stack: Mechanical RAG System (Reviewed)

> **Decision**: Fully Open Source · **Languages**: Arabic + English · **Domain**: Mechanical Service Manuals

---

## 1. Embedding Models — Updated Comparison

| Model | Parameters | Multilingual (Arabic) | Context Window | VRAM Needed | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`BAAI/bge-m3`** | ~568M | **Exceptional** (100+ langs, trained on 170+ langs) | 8192 tokens | ~2 GB | **Top Pick.** "M3" = Multi-lingual, Multi-functional (dense + sparse + multi-vector), Multi-granularity. Supports hybrid retrieval natively. |
| **`sayed0am/arabic-english-bge-m3`** | ~362M (36% smaller) | **Arabic-Optimized** (pruned for Arabic/English only) | 8192 tokens | ~1.2 GB | **Excellent Alternative.** Pruned BGE-M3 that removes non-Arabic/English tokens. Ranked **#1 open-source on The Arabic RAG Leaderboard**. ONNX quantized version is only 363 MB (~75% smaller) at 98% quality. |
| **`intfloat/e5-mistral-7b-instruct`** | 7B | Very Good | **32,768 tokens** | ~14 GB (fp16) | Uses a full 7B LLM backbone for embeddings. Best raw semantic accuracy, but extremely heavy. Only viable on ≥16 GB VRAM GPUs. |
| **`nomic-ai/nomic-embed-text-v1.5`** | ~137M | English primary (**Weak Arabic**) | 8192 tokens | ~0.5 GB | Best if you translate everything to English first. Supports Matryoshka (truncatable vectors). Very fast. |

> [!IMPORTANT]
> **Key Discovery**: `sayed0am/arabic-english-bge-m3` is a pruned BGE-M3 specifically optimized for Arabic+English. It is 36% smaller, #1 on the Arabic RAG Leaderboard among open-source models, and has an ONNX quantized version at only 363 MB. This is a strong candidate for your project.

---

## 2. The Arabic Question: Translation API vs. Fine-Tuning BGE-M3

This is the most critical architectural decision for your project. Here is a deep comparison:

### Option A: Translation API Layer (Forward & Backward)
**How it works**: User sends Arabic query → Translate to English → Embed & Search English chunks → Retrieve English results → Translate back to Arabic → Present to user.

| Aspect | Detail |
| :--- | :--- |
| **Accuracy** | Depends entirely on translation quality. Technical terms like "مضخة الزيت" (oil pump) or part numbers translate well. But complex procedural sentences can lose critical nuance. |
| **Latency** | **+200-500ms per request** (two translation API calls added to the pipeline). |
| **Cost** | If using a local model like `facebook/nllb-200-distilled-600M` (free, 600M params, supports Arabic): **$0**. If using Google Translate API: ~$20 per 1M characters. |
| **Complexity** | Low to add, but creates a new failure point. Translation errors cascade into bad retrieval. |
| **Pros** | ✅ Your entire knowledge base stays purely English. ✅ Embedding model choice doesn't matter for Arabic. ✅ Simple to implement. |
| **Cons** | ❌ Adds latency. ❌ Translation errors propagate. ❌ Struggles with code-switching (mixed Arabic/English in one query). ❌ Domain-specific jargon may not translate correctly. |

### Option B: Fine-Tune BGE-M3 for Mechanical Arabic Domain
**How it works**: Create Arabic-English query-passage pairs from your manuals → Fine-tune BGE-M3 with `FlagEmbedding` library using LoRA → The model natively maps Arabic queries to English chunks without translation.

| Aspect | Detail |
| :--- | :--- |
| **Accuracy** | **Highest possible.** The model learns *your exact domain vocabulary* in both languages. "أين مضخة الزيت" directly maps to the correct English maintenance chunk. |
| **Latency** | **Zero additional latency.** No translation step needed. |
| **Cost** | Fine-tuning cost: A few hours on a free Kaggle/Colab T4 GPU. Inference cost: Same as base BGE-M3. |
| **Complexity** | Medium. You need to create training data (query-passage pairs). Can use an LLM to synthetically generate Arabic questions from your English chunks. |
| **Pros** | ✅ Zero translation latency. ✅ Handles code-switching naturally. ✅ Learns your exact mechanical vocabulary. ✅ Can be done with LoRA (parameter-efficient, small adapter file). |
| **Cons** | ❌ Requires creating a training dataset (~500-2000 pairs recommended). ❌ Needs a GPU for the fine-tuning step (but free on Kaggle/Colab). |

### Option C: Hybrid Approach (Recommended)
**Phase 1**: Start with the base `BAAI/bge-m3` or `sayed0am/arabic-english-bge-m3` (already Arabic-optimized). Test how well it handles Arabic queries against your English chunks **out of the box**.
**Phase 2**: If accuracy is insufficient, fine-tune with LoRA using synthetic Arabic query data generated from your chunks.
**Phase 3**: Only add a translation layer as a **fallback** for queries where the embedding similarity score is below a confidence threshold.

> [!TIP]
> **Recommendation**: Start with the Hybrid approach. BGE-M3 is already very strong at cross-lingual retrieval. You may find that fine-tuning is unnecessary. Test first, optimize later.

---

## 3. RAG Frameworks & Orchestration

| Framework | Complexity | Key Strength for This Project | Ideal When |
| :--- | :--- | :--- | :--- |
| **LlamaIndex** | Medium | **Node relationships.** Natively handles parent-child chunk hierarchies, image-to-text linking via metadata, and structured table nodes. | You want full programmatic control over how Docling chunks are indexed before any query happens. |
| **RAGFlow** | Medium | **Visual document pipeline.** Built specifically for complex PDF understanding with OCR, table extraction, and chunk management via a web UI. | You want an end-to-end open-source RAG system with a built-in UI, without writing everything from scratch. |
| **Haystack** | Medium | **Clean modular pipelines.** Each component (retriever, reranker, reader) is a pluggable node in a Python graph. | You want maximum code cleanliness and the ability to swap components easily. |

---

## 4. Serving Architecture

| Tool | Role | Best For |
| :--- | :--- | :--- |
| **Ollama** | LLM Serving | Fastest way to get Llama-3, Mistral, Qwen running locally. One command install. |
| **vLLM** | LLM Serving (Production) | Maximum throughput with PagedAttention. Best for concurrent users. |
| **TGI** (HuggingFace) | LLM Serving (Production) | Similar to vLLM. Better native HuggingFace model support. Docker-first deployment. |
| **LiteLLM** | API Router/Gateway | Puts a unified OpenAI-compatible API in front of Ollama/vLLM/TGI. Your frontend code never changes even if you swap the backend model. |

---

## 5. Hosting & Remote Testing — Full Comparison (Including Kaggle & Colab)

| Platform | Free GPU? | GPU Type | Session Limit | Persistent Storage | Can Host API Remotely? | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Colab** | ✅ Yes (Free tier) | T4 (16 GB VRAM) | ~12 hrs max (often shorter), 90 min idle timeout | No (must save to Drive) | **Yes, via ngrok tunnel** | **Best free option for testing.** Run your full RAG stack, expose it with ngrok, test from your phone/laptop. |
| **Kaggle Notebooks** | ✅ Yes (30 hrs/week GPU) | T4 x2 (16 GB each) or P100 | 6 hrs per session | Yes (Kaggle Datasets) | **Limited.** Not designed for API hosting, no easy ngrok integration. | **Best for fine-tuning models.** Use Kaggle to fine-tune BGE-M3 on your mechanical data, then download the model. |
| **RunPod** | ❌ Paid (~$0.40/hr) | RTX 4090 / A100 | Unlimited | Yes (Volumes) | **Yes, full SSH + public URL** | **Best for production-like testing.** Deploy with `docker-compose`, get a public URL, test like a real server. |
| **Vast.ai** | ❌ Paid (~$0.20/hr) | Various (market pricing) | Unlimited | Yes | **Yes, SSH + ports** | Cheapest GPU rental. Community GPUs, less reliable than RunPod. |
| **Modal** | ✅ Free tier ($30/month credits) | A10G / A100 | Serverless (auto-sleep) | Cold storage only | **Yes, automatic public URL** | Deploy Python functions as serverless endpoints. Great for API but tricky for Qdrant. |
| **HuggingFace Spaces** | ✅ Free (CPU) / Paid (GPU) | T4 / A10G | Sleeps after inactivity | Yes | **Yes, automatic public URL** | Host a Gradio/Streamlit demo for free. Paid GPU tier for inference. |

> [!IMPORTANT]
> **Yes, you can absolutely use Colab for remote testing!** The workflow:
> 1. Run your RAG pipeline (FastAPI + Qdrant + Ollama) inside a Colab notebook
> 2. Install `pyngrok` and create a tunnel to expose your FastAPI port
> 3. You get a public URL like `https://xxxx.ngrok-free.app`
> 4. Access it from your phone, laptop, or anywhere
>
> **Limitation**: Sessions die after ~12 hours or 90 min idle. Not for production, but perfect for testing.

> [!WARNING]
> **Kaggle is NOT ideal for hosting an API.** It's designed for batch computation, not serving. Use Kaggle for **fine-tuning your embedding model** (free T4 x2 GPUs, 30 hrs/week), then deploy the fine-tuned model elsewhere (Colab, RunPod, etc.).

---

## 6. Recommended Architecture (Summary)

```
┌─────────────────────────────────────────────────────┐
│                    YOUR LAPTOP / PHONE               │
│              (Access via ngrok public URL)            │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
                       ▼
┌─────────────────────────────────────────────────────┐
│              GOOGLE COLAB (Free T4 GPU)              │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ FastAPI   │  │ Qdrant   │  │ Ollama            │  │
│  │ (API +    │  │ (Vector  │  │ (Llama-3 / Qwen)  │  │
│  │  RAG      │  │  DB)     │  │                   │  │
│  │  Logic)   │  │          │  │                   │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  BGE-M3 or arabic-english-bge-m3 (Embeddings)│    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  ngrok tunnel → public URL                    │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘

Fine-tuning (if needed):
┌─────────────────────────────────────────────────────┐
│              KAGGLE (Free T4 x2 GPUs)                │
│                                                      │
│  Upload chunks JSON → Generate Arabic queries        │
│  → Fine-tune BGE-M3 with LoRA (FlagEmbedding)       │
│  → Download fine-tuned adapter → Use in Colab        │
└─────────────────────────────────────────────────────┘
```

---

## Decision Checklist

- `[ ]` **1. Embedding Model**: Start with `sayed0am/arabic-english-bge-m3` (Arabic-optimized, smaller) or full `BAAI/bge-m3`?
- `[ ]` **2. Arabic Strategy**: Test BGE-M3 out-of-the-box first → Fine-tune if needed → Translation as last resort?
- `[ ]` **3. Framework**: LlamaIndex (code-first) vs RAGFlow (visual pipeline)?
- `[ ]` **4. Hosting for Testing**: Google Colab + ngrok (free) or RunPod ($0.40/hr, more stable)?
- `[ ]` **5. LLM**: Which generative model? (Llama-3 8B, Mistral 7B, Qwen-2.5 7B?)
