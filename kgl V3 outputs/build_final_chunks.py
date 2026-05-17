"""
MECHRABOT — Final Chunk Builder
Uses existing V3 files (already chunked on Kaggle) to produce enriched chunks.
No docling import needed. No version issues. Just plain JSON.

Input:  chunks.json (from HybridChunker on Kaggle)
        output_V3.json (DoclingDocument with image refs)
Output: final_chunks.json (ready for BGE-m3 + Qdrant)

Run:    cd "kgl V3" && python3 build_final_chunks.py
"""

import json
import hashlib
import re
from collections import defaultdict, Counter

# ============================================================
# 1. LOAD FILES
# ============================================================
print("Loading chunks.json...")
raw_chunks = json.load(open("chunks.json"))
print(f"  → {len(raw_chunks)} chunks")

print("Loading output_V3.json for image data...")
doc = json.load(open("output_V3_fullresult.json"))
print(f"  → {len(doc['pictures'])} pictures, {len(doc['pages'])} pages")

# ============================================================
# 2. BUILD PAGE → IMAGES LOOKUP
# ============================================================
page_images = defaultdict(list)
for pic in doc["pictures"]:
    prov = pic.get("prov", [])
    page = prov[0]["page_no"] if prov else None
    uri = pic.get("image", {}).get("uri", "")
    if page and uri:
        page_images[page].append(uri)

print(f"\nPages with images: {len(page_images)}")
print(f"Total image links: {sum(len(v) for v in page_images.values())}")

# ============================================================
# 3. BUILD SPEC TABLE ROW CHUNKS (torque/clearance values)
# ============================================================
spec_pattern = re.compile(r'\b(\d+\.?\d*)\s*(Nm|mm|bar|kPa|MPa)\b')
spec_chunks = []

for table in doc["tables"]:
    if table.get("label") == "document_index":
        continue
    grid = table.get("data", {}).get("grid", [])
    if not grid or len(grid) < 2:
        continue
    flat = " ".join(c.get("text", "") for row in grid for c in row)
    if not spec_pattern.search(flat):
        continue

    table_page = table["prov"][0]["page_no"] if table.get("prov") else None
    headers = [c.get("text", "") for c in grid[0]]

    for row in grid[1:]:
        pairs = []
        for i in range(min(len(headers), len(row))):
            cell_text = row[i].get("text", "").strip()
            if cell_text:
                pairs.append(f"{headers[i]}: {cell_text}")
        if pairs:
            spec_chunks.append({
                "text": "[Specs]\n" + " | ".join(pairs),
                "page": table_page,
            })

print(f"Spec table row chunks: {len(spec_chunks)}")

# ============================================================
# 4. EXTRACT PAGE NUMBER FROM RAW CHUNK doc_items
# ============================================================
page_regex = re.compile(r'page_no=(\d+)')
img_code = re.compile(r'\b[A-Z]{2,6}\d{5,}[A-Z]?\b')

def get_page(chunk):
    """Extract page_no from stringified doc_items."""
    items_str = str(chunk.get("meta", {}).get("doc_items", ""))
    match = page_regex.search(items_str)
    return int(match.group(1)) if match else None

def get_headings(chunk):
    """Extract headings — they're the prefix in 'contextualized' that's not in 'text'."""
    ctx = chunk.get("contextualized", "")
    txt = chunk.get("text", "")
    if ctx and txt and ctx != txt:
        prefix = ctx[:ctx.find(txt)].strip() if txt in ctx else ""
        if prefix:
            return [h.strip() for h in prefix.split("\n") if h.strip()]
    return []

# ============================================================
# 5. BUILD FINAL ENRICHED CHUNKS
# ============================================================
print("\nBuilding final chunks...")
all_chunks = []

# --- Text chunks from HybridChunker ---
for i, c in enumerate(raw_chunks):
    text = c.get("text", "").strip()
    if len(text) < 30:
        continue

    # Clean
    text = img_code.sub("", text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    page = get_page(c)
    headings = get_headings(c)

    all_chunks.append({
        "content": text,
        "meta": {
            "chunk_id": hashlib.md5(f"t_{i}_{text[:80]}".encode()).hexdigest()[:12],
            "source": "m11_SM.pdf",
            "page": page,
            "section": headings,
            "type": "text",
            "image_paths": page_images.get(page, []),
        }
    })

# --- Spec row chunks ---
for i, sc in enumerate(spec_chunks):
    text = sc["text"]
    page = sc["page"]

    all_chunks.append({
        "content": text,
        "meta": {
            "chunk_id": hashlib.md5(f"s_{i}_{text[:80]}".encode()).hexdigest()[:12],
            "source": "m11_SM.pdf",
            "page": page,
            "section": None,
            "type": "table_spec",
            "image_paths": page_images.get(page, []),
        }
    })

# ============================================================
# 6. QUALITY REPORT
# ============================================================
types = Counter(c["meta"]["type"] for c in all_chunks)
ids = [c["meta"]["chunk_id"] for c in all_chunks]
with_imgs = sum(1 for c in all_chunks if c["meta"]["image_paths"])
with_page = sum(1 for c in all_chunks if c["meta"]["page"])
with_headings = sum(1 for c in all_chunks if c["meta"]["section"])
lengths = [len(c["content"]) for c in all_chunks]

print(f"\n{'='*50}")
print(f"  QUALITY REPORT")
print(f"{'='*50}")
print(f"  Total chunks:        {len(all_chunks)}")
print(f"  Types:               {dict(types)}")
print(f"  Unique IDs:          {len(set(ids))} / {len(ids)}")
print(f"  With page number:    {with_page}")
print(f"  With headings:       {with_headings}")
print(f"  With image links:    {with_imgs} ({100*with_imgs/len(all_chunks):.1f}%)")
print(f"  Content length:      min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")
print(f"{'='*50}")

# ============================================================
# 7. SAVE
# ============================================================
with open("final_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(all_chunks)} chunks → final_chunks.json")
