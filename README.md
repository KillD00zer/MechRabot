<div align="center">

# ⚙️ MECHRABOT

**A Multi-Advanced Retrieval-Augmented Generation (MRAG) Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Vector DB: Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Document AI](https://img.shields.io/badge/Document_AI-Docling-FFA500.svg)](https://github.com/DS4SD/docling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Cover Image](Gemini_Generated_Image_7vi6q47vi6q47vi6.jpg)

*Transforming static documentation into an interactive, context-aware diagnostic intelligence system.*

<br/>

[Features](#-key-features) • [Installation](#-getting-started) • [Architecture](#-architecture-stack) • [Roadmap](#-roadmap) 

</div>

---

## 🚀 About MechRabot

**MECHRABOT** is a high-precision, purpose-built MRAG engine designed for navigating and extracting insights from complex mechanical maintenance manuals. In high-stakes technical environments, standard RAG fail on dense specs and untethered diagrams. MechRabot solves this by implementing deep spatial understanding, linking images to chunks via bounding boxes, and preserving hierarchical section context.

## ✨ Key Features

> **🧠 Advanced RAG Architecture**  
> Intelligent retrieval pipeline heavily optimized for dense technical specifications, hierarchical tables, and complex mechanical diagrams.

> **⚡ High-Performance Vector Search**  
> Built for uncompromising, lightning-fast semantic querying across extensive manual databases using Qdrant with deterministic UUIDs.

> **📊 Spatial Document Parsing**  
> Preserves context through spatial image-to-chunk linking and hierarchical specification tracking via robust PDF & JSON extraction architectures.

> **🛠️ Extensible NLP Pipeline**  
> Seamlessly integrates with modern LLM frameworks to handle multi-step mechanical reasoning—designed to support both local fine-tuned embeddings (e.g., domain-specific mechanical slang) and premium translation API layers.

---

## 🏗️ Architecture Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Extraction** | `Docling`, `PyMuPDF` | Converts rich PDFs into hierarchical JSON and Markdown, establishing spatial bounds. |
| **Embeddings** | `Sentence-Transformers` | Tailored embedded vectors fine-tuned for mechanical vocabulary and slang. |
| **Vector Engine** | `Qdrant DB` | Handles robust similarities searches and multidimensional metadata filtering. |
| **LLM Orchestration** | Hybrid | Local inference combined with a premium translation/reasoning API for high fidelity final UI output. |

---

## 📦 Getting Started

### Prerequisites
- Python 3.10+
- Qdrant Vector Store instance

### ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/MechRabot.git
cd MechRabot

# 2. Set up your virtual environment
python -m venv venv
source venv/bin/activate  # Windows: `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt
```

### ⚙️ Pipeline Execution

Run the data ingestion pipeline to process technical manuals and build spatially-aware semantic chunks:

```bash
python -m src.pipeline.ingest --input ./manuals --format json
```

Boot up the retrieval inference engine:

```bash
python -m src.api.server --debug
```

---

## 🗺️ Roadmap
- [x] Extract PDF tables and diagrams to hierarchical markdown/JSON.
- [x] Spatial image-to-chunk correlation using bounding boxes.
- [x] Domain-specific fine-tuning of local embedding models.
- [ ] Connect multi-modal UI reasoning over diagram highlights.
- [ ] Real-time diagnostic chat interface.

---

<div align="center">
  <i>Engineered for the mechanics of the future. 🛠️🦾</i>
</div>
