# MechRabot — Documentation Index

All project documentation lives in this folder, organized by topic.

---

## 📁 Folder Structure

```
reports/
├── INDEX.md                                      ← you are here
│
├── 01_architecture/
│   ├── architecture_overview.pdf                 ← full system architecture diagram
│   └── data_pipeline_design.md                   ← data pipeline + chunk schema details
│
├── 02_tech_stack/
│   └── tech_stack_evaluation.md                  ← why we chose each tool (Qdrant, BGE-M3, Haystack...)
│
├── 03_embeddings/
│   ├── retrieval_paradigms_guide.md              ← Dense vs Sparse vs ColBERT full guide
│   └── bge_m3_output_explained.md                ← understanding BGE-M3's three output vectors
│
├── 04_storage_transport/
│   └── kaggle_to_qdrant_transport.md             ← how to move embeddings from Kaggle to Qdrant
│
└── 05_evaluation/
    ├── README.md                                  ← metrics theory, dataset structure, learning path
    ├── evaluation_30_V1.json                      ← 30 grounded queries (EN + AR + ar_slang)
    ├── retrieval_evaluation_learning_guide.md     ← 📚 cell-by-cell guide of the original notebook
    └── mathimatical-retreival-evaluation.ipynb   ← 🔬 original notebook (manual math, for study)
```

---

## 📄 Document Summaries

### 01 — Architecture

| File | What it covers |
|:---|:---|
| `architecture_overview.pdf` | System diagram: Docling → Chunking → Embedding → Qdrant → Haystack → LLM |
| `data_pipeline_design.md` | Chunk schema, `final_chunks_v2.json` structure, `section_path`, `linked_images`, UUIDs |

### 02 — Tech Stack

| File | What it covers |
|:---|:---|
| `tech_stack_evaluation.md` | Comparison of all evaluated tools, final decisions and reasoning |

### 03 — Embeddings

| File | What it covers |
|:---|:---|
| `retrieval_paradigms_guide.md` | Dense, Sparse, ColBERT mechanics + hybrid fusion (RRF) + MechRabot query archetypes |
| `bge_m3_output_explained.md` | Why output shape is `(2790, 1024)`, what near-zero values mean, sparse token IDs, ColBERT matrices |

### 04 — Storage & Transport

| File | What it covers |
|:---|:---|
| `kaggle_to_qdrant_transport.md` | Three strategies (JSON, binary files, direct streaming) compared before/during/after |

### 05 — Evaluation

| File | What it covers |
|:---|:---|
| `README.md` | Metrics theory (MRR, NDCG, Recall), dataset structure, thresholds, learning path |
| `evaluation_30_V1.json` | 30 labeled queries — the ground truth qrels file |
| `retrieval_evaluation_learning_guide.md` | Cell-by-cell explanation of the evaluation notebook with worked math examples |
| `mathimatical-retreival-evaluation.ipynb` | Original evaluation notebook — pure math, no libs — use for learning |

---

## 🛠️ Related Notebooks (`main_work/`)

| Notebook / Script | Purpose |
|:---|:---|
| `retrieval-evaluation-ranx.ipynb` | **Production evaluation** — uses `ranx` library, 30+ metrics, cleaner code |
| `encode_and_inject.py` | **Recommended** — encode + inject to Qdrant in one loop, no files needed |
| `save_embeddings.py` | Save all three vector types to binary files (checkpoint for large datasets) |
| `inject_to_qdrant.py` | Load saved binary files and inject into Qdrant (local / cloud / memory) |
| `build_final_chunks_v2.py` | Build `final_chunks_v2.json` from Docling output |

| `build_final_chunks_v2.py` | Build `final_chunks_v2.json` from Docling output |
