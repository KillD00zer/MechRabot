<div align="center">

# ⚙️ MECHRABOT

**Production-Grade Hybrid RAG Engine for Mechanical Maintenance Manuals**

[![Vector DB](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Document AI](https://img.shields.io/badge/Docling-Document_Parser-FFA500.svg)](https://github.com/DS4SD/docling)
[![Embeddings](https://img.shields.io/badge/BGE--M3-Hybrid_Embeddings-4B7BEC.svg)](https://huggingface.co/BAAI/bge-m3)
[![Framework](https://img.shields.io/badge/Haystack-Pipeline_Engine-1A1A2E.svg)](https://haystack.deepset.ai/)
[![Fine-Tuning](https://img.shields.io/badge/FlagEmbedding-Fine--Tuning-6C3483.svg)](https://github.com/FlagOpen/FlagEmbedding)
[![License](https://img.shields.io/badge/License-MIT-27AE60.svg)](#-license)

![Cover Image](Gemini_Generated_Image_7vi6q47vi6q47vi6.jpg)

*Transforming dense, static mechanical documentation into a spatially-aware, context-preserving diagnostic intelligence system — engineered specifically for Arabic/English mechanical domains.*

<br/>

[About](#-about) • [Architecture](#-architecture) • [Data Pipeline](#-data-pipeline) • [Embedding Strategy](#-embedding--fine-tuning-strategy) • [Getting Started](#-getting-started)

</div>

---

## 🔥 About

**MECHRABOT** is a high-precision, open-source Hybrid-RAG engine designed to extract exact, context-rich answers from complex automotive maintenance manuals.

Standard RAG architectures fail catastrophically in mechanical engineering domains:

- A generic semantic search for `"10.5 Nm"` will return semantically similar results like `"10 Nm"` — a catastrophic difference when torquing an engine head.
- Diagram images are blindly dumped to every nearby text chunk causing Vision-LLM hallucination.
- Extracted table rows lose their parent hierarchy: `"Torque: 65 Nm"` becomes meaningless without knowing it applies to the Cylinder Head.
- Arabic mechanical slang (`"سير كاتينة"`) has zero overlap with English training data for standard models.

MechRabot solves every single one of these problems.

---

## ✨ Core Features

> **📐 Spatial Bounding Box Image Linking**
> Eliminates Vision-LLM hallucination. Instead of grouping all images on a page to all text blocks, MechRabot runs Euclidean distance math on `bbox` coordinates from Docling's layout engine. Each chunk's `linked_images` contains **only the diagram physically nearest to it** on the page.

> **🗂️ Hierarchical Spec Table Extraction**
> Eliminates the "Lost Context" problem. Every spec table row (e.g., `Torque: 65 Nm`) is traced upward through the Docling element tree to recover its parent heading chain — stored as `section_path: ["ENGINE", "CYLINDER HEAD", "INSTALLATION"]` — and prepended directly into the embedded content.

> **⛓️ Chunk Linked Lists**
> Eliminates fragmentation. Every text chunk holds deterministic UUID references (`previous_chunk_id`, `next_chunk_id`) to its neighbors. If the retrieved chunk says "tighten to above specification", the Haystack pipeline fetches the adjacent context **without an extra vector search**.

> **📊 Parent-Child Table Architecture**
> Dual-mode table ingestion. Each spec table is stored **both** as precision row-by-row chunks (for surgical spec lookup) **and** as a single full-table chunk (for broad queries), where all row chunks carry a `parent_table_id` pointer.

> **🌍 Arabic-English Cross-Lingual Retrieval**
> Natively maps Arabic mechanical queries — including regional Egyptian slang (`سير كاتينة` → Timing Belt) — to English manual chunks via `BAAI/bge-m3`'s 100-language vector space, optionally supercharged via LoRA fine-tuning.

---

## 🏗️ Architecture

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Document Parsing** | `Docling` + `HybridChunker` | Converts raw PDFs into spatially-rich JSON, preserving `bbox` coordinates for every text block, picture, and table. |
| **Chunk Enrichment** | `build_final_chunks_v2.py` | Injects `section_path`, `linked_images`, `parent_table_id`, and prev/next IDs into each chunk. |
| **Embedding Model** | `BAAI/bge-m3` | Outputs **Dense + Sparse + ColBERT** vectors simultaneously. The only model that handles both semantic meaning AND exact keyword/number matching natively. |
| **Fine-Tuning** | `FlagEmbedding` + LoRA | Unified fine-tuning on Kaggle (Free T4 x2) to bridge Arabic mechanical slang to English manual passages. |
| **Vector Engine** | `Qdrant` | Native hybrid search — fuses Dense (semantic) and Sparse (BM25/lexical) scores in a single query with payload filtering on `section_path`. |
| **Pipeline Engine** | `Haystack` | Modular retrieval graph handling chunk chaining, multi-step reasoning, and dual-LLM routing. |
| **Fast Inference LLM** | `Groq API` | Ultra-low-latency Arabic→English query rewriting and reformulation. |
| **Reasoning LLM** | `Qwen2.5-14B-Instruct` | Deep mechanical reasoning for complex diagnostic queries. Hosted on Kaggle GPU inference. |

---

## 📐 Data Pipeline

```
Raw PDF
   │
   ├──► [Docling DocumentConverter]
   │         └──► output_V3_fullresult.json
   │               ├── texts[]  { text, bbox, page_no }
   │               ├── pictures[] { uri, bbox, page_no }
   │               └── tables[]  { grid[][], bbox, page_no }
   │
   └──► [Docling HybridChunker]
             └──► chunks.json
                   └── { text, contextualized, doc_items[prov → bbox] }

         [build_final_chunks_v2.py]
               │
               ├── Bbox distance math → linked_images (nearest only)
               ├── Heading tree trace → section_path
               ├── Row + Full table chunking → parent_table_id
               └── Sequential hash linking → prev/next chunk IDs
               │
               ▼
        final_chunks_v2.json  ──►  [BGE-M3 Embed]  ──►  [Qdrant Upsert]
```

### Final Chunk Schema (Qdrant Payload)

```json
{
  "chunk_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "content": "ENGINE > CYLINDER HEAD > INSTALLATION\n[Specs] Component: Head Bolt | Torque: 65 N·m",
  "meta": {
    "source_file": "m11_SM.pdf",
    "page_no": 42,
    "chunk_type": "table_spec",
    "section_path": ["ENGINE", "CYLINDER HEAD", "INSTALLATION"],
    "linked_images": ["extracted_artifacts/image_000042_a3f8c1.png"],
    "bbox": { "l": 20.0, "t": 300.0, "r": 400.0, "b": 150.0 },
    "parent_table_id": "6ba7b810-9dad-11d1-80b4-111111111111",
    "previous_chunk_id": "6ba7b810-9dad-11d1-80b4-000000000000",
    "next_chunk_id": "6ba7b810-9dad-11d1-80b4-222222222222"
  }
}
```

---

## 🧠 Embedding & Fine-Tuning Strategy

### Why `BAAI/bge-m3`?

BGE-M3 is the only open-source embedding model that simultaneously outputs **three** vector types from a single inference pass:

| Vector Type | What it captures | Why it matters for mechanical RAG |
| :--- | :--- | :--- |
| **Dense** | Semantic meaning and conceptual proximity | Finds "engine head gasket" when user asks "head sealing component" |
| **Sparse (BM25)** | Exact keyword & number matching | Guarantees `10.5 Nm` ≠ `10 Nm` — critical for safety-critical specs |
| **ColBERT** | Token-level soft matching | Handles part codes, Arabic-English code-switching, abbreviations |

Qdrant natively fuses all three scores in a **Reciprocal Rank Fusion (RRF)** query, no extra orchestration needed.

### Fine-Tuning on Egyptian Mechanical Slang

Using `FlagEmbedding` **Unified Fine-Tuning** with LoRA on Kaggle Free T4 x2 GPUs:

```json
{"query": "سير كاتينة", "positive": "The timing belt synchronizes the crankshaft..."}
{"query": "جلبة مقص", "positive": "Lower control arm bushing absorbs vibrations..."}
{"query": "بلوف مكيف", "positive": "A/C compressor valve leak diagnosis..."}
```

This pushes Arabic slang vectors physically adjacent to their English manual counterparts in the 1024-dimensional vector space — zero translation latency at inference time.

---

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.10
- Docker (for Qdrant)
- Kaggle account (for GPU fine-tuning)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MechRabot.git
cd MechRabot

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Build Chunks

```bash
# Run from the main_work/ directory
cd main_work
python build_final_chunks_v2.py
# → Outputs: final_chunks_v2.json  (2790 Qdrant-ready chunks)
```

### Embed & Ingest to Qdrant

```bash
python scripts/embed_to_qdrant.py \
  --input main_work/final_chunks_v2.json \
  --collection mechrabot_v2
```

---

## 📊 Pipeline Stats (v2)

| Metric | Value |
| :--- | :--- |
| Total Chunks | **2,790** |
| Text Chunks | 2,394 |
| Spec Table Rows | 314 |
| Full Table Chunks | 82 |
| Chunks with `section_path` | **99.8%** |
| Chunks with `linked_images` | **81.1%** (spatially matched, not page-dumped) |
| Chunks with linked-list pointers | **2,393** (full sequence coverage) |
| Unique Qdrant-compliant UUIDs | **2,782 / 2,790** |
| Max chunk length | 3,007 chars (well within BGE-M3's 8,192 token limit) |

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/BetterChunking`
3. Commit changes: `git commit -m 'Improve bbox spatial matching threshold'`
4. Push and open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">
  <i>Built for the mechanics of the future. Every Nm counts. 🛠️⚡</i>
</div>
