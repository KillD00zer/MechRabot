---
description: how to regenerate the interactive Docling mind map
---
This workflow regenerates the `docling_map.html` interactive D3.js mind map from `docling_arch.json`. It maps out the Docling architecture and filters the interface to explicitly focus on AI pipelines, RAG operations, and Hardware configuration.

1. Ensure both `docling_arch.json` and `generate_map.py` are present in the workspace.
2. Run the `generate_map.py` python script to regenerate the HTML file.

// turbo
```bash
python generate_map.py
```
