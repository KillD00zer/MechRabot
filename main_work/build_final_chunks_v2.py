"""
MECHRABOT — Final Chunk Builder v2
Implements full target architecture:
  - bbox-based image linking (nearest image, not all-on-page)
  - section_path from output_V3 heading tree
  - Parent-Child table architecture (row-by-row + full-table chunk)
  - previous_chunk_id / next_chunk_id linked list

Input:  chunks.json               (from HybridChunker on Kaggle)
        output_V3_fullresult.json (DoclingDocument with image refs)

Output: final_chunks_v2.json      (ready for BGE-m3 + Qdrant)
"""

import json
import hashlib
import re
import uuid
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
CHUNKS_PATH   = "/mnt/78AA22ECAA22A71A/K_REPO/MechRabot/main_work/chunks.json"
DOC_PATH      = "main_work/output_V3_fullresult.json"
OUTPUT_PATH   = "final_chunks_v2.json"

SPEC_UNITS    = re.compile(r'\b(\d+\.?\d*)\s*(N·m|Nm|mm|bar|kPa|MPa|°|%)\b')
IMG_CODE      = re.compile(r'\b[A-Z]{2,6}\d{5,}[A-Z]?\b')
PAGE_REGEX    = re.compile(r'page_no=(\d+)')
BBOX_REGEX    = re.compile(r'BoundingBox\(l=([\d.]+), t=([\d.]+), r=([\d.]+), b=([\d.]+)')


# ============================================================
# HELPERS
# ============================================================

def sha_id(*parts):
    """Generate a deterministic Qdrant-compliant UUID string."""
    h = hashlib.md5("||".join(str(p) for p in parts).encode()).hexdigest()
    return str(uuid.UUID(h))


def bbox_center(bbox):
    """Return (cx, cy) center of a bbox dict with l/t/r/b keys."""
    return ((bbox['l'] + bbox['r']) / 2, (bbox['t'] + bbox['b']) / 2)


def bbox_distance(a, b):
    """Euclidean distance between two bbox centers."""
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def parse_chunk_bbox(chunk):
    """Extract page_no and bbox from the doc_items string in a raw chunk."""
    items_str = str(chunk.get("meta", {}).get("doc_items", ""))
    page_match = PAGE_REGEX.search(items_str)
    bbox_match  = BBOX_REGEX.search(items_str)
    page = int(page_match.group(1)) if page_match else None
    bbox = None
    if bbox_match:
        l, t, r, b = [float(bbox_match.group(i)) for i in range(1, 5)]
        bbox = {"l": l, "t": t, "r": r, "b": b}
    return page, bbox


def get_headings(chunk):
    """Extract section headings from contextualized prefix."""
    ctx = chunk.get("contextualized", "")
    txt = chunk.get("text", "")
    if ctx and txt and ctx != txt and txt in ctx:
        prefix = ctx[:ctx.find(txt)].strip()
        if prefix:
            return [h.strip() for h in prefix.split("\n") if h.strip()]
    return []


# ============================================================
# 1. LOAD FILES
# ============================================================
print("Loading chunks.json...")
raw_chunks = json.load(open(CHUNKS_PATH))
print(f"  → {len(raw_chunks)} chunks")

print("Loading output_V3_fullresult.json...")
doc = json.load(open(DOC_PATH))
print(f"  → {len(doc.get('pictures', []))} pictures, {len(doc.get('tables', []))} tables")


# ============================================================
# 2. BUILD PAGE → PICTURES LOOKUP (with bbox)
# ============================================================
page_pictures = defaultdict(list)
for pic in doc.get("pictures", []):
    prov = pic.get("prov", [])
    if not prov:
        continue
    page = prov[0].get("page_no")
    bbox = prov[0].get("bbox")
    uri  = pic.get("image", {}).get("uri", "")
    if page and bbox and uri:
        page_pictures[page].append({"uri": uri, "bbox": bbox})

print(f"\nPages with pictures: {len(page_pictures)}")


# ============================================================
# 3. BUILD PAGE → HEADINGS LOOKUP (from texts in output_V3)
#    We need headings to assign to tables that have none
# ============================================================
# Heading labels in Docling
HEADING_LABELS = {"section_header", "title"}

page_headings = defaultdict(list)  # page_no -> [(bbox, text), ...]
for text_item in doc.get("texts", []):
    label = text_item.get("label", "")
    if label not in HEADING_LABELS:
        continue
    prov = text_item.get("prov", [])
    if not prov:
        continue
    page = prov[0].get("page_no")
    bbox = prov[0].get("bbox")
    text = text_item.get("text", "").strip()
    if page and bbox and text:
        page_headings[page].append({"bbox": bbox, "text": text})

# Also track ALL texts per page for fallback heading lookup
page_all_texts = defaultdict(list)
for text_item in doc.get("texts", []):
    prov = text_item.get("prov", [])
    if not prov:
        continue
    page = prov[0].get("page_no")
    bbox = prov[0].get("bbox")
    text = text_item.get("text", "").strip()
    label = text_item.get("label", "")
    if page and bbox and text:
        page_all_texts[page].append({"bbox": bbox, "text": text, "label": label})

print(f"Pages with section headings: {len(page_headings)}")


# ============================================================
# 4. HELPERS: Nearest image, Nearest heading above a bbox
# ============================================================

def nearest_images(page, chunk_bbox, max_distance=600, max_count=2):
    """Return URIs of images physically closest to this chunk's bbox."""
    candidates = page_pictures.get(page, [])
    if not candidates:
        return []
    scored = []
    for pic in candidates:
        d = bbox_distance(chunk_bbox, pic["bbox"])
        scored.append((d, pic["uri"]))
    scored.sort(key=lambda x: x[0])
    # Only return images within a reasonable distance threshold
    return [uri for d, uri in scored if d <= max_distance][:max_count]


def nearest_heading_above(page, table_bbox):
    """Find the closest heading ABOVE the table on the same page."""
    # In BOTTOMLEFT coords, 'b' is the bottom edge. A heading above
    # would have a 'b' value greater than the table's 't' (top of table).
    table_top = table_bbox.get("t", 0)
    
    best_text = []
    # Check the same page for headings above the table
    for heading in page_headings.get(page, []):
        h_bottom = heading["bbox"].get("b", 0)
        if h_bottom >= table_top:  # heading is above table
            best_text.append(heading["text"])
    
    # Fallback: check the page before
    if not best_text and page > 1:
        for heading in page_headings.get(page - 1, []):
            best_text.append(heading["text"])
    
    return best_text[-3:] if best_text else []  # Last 3 headings above


# ============================================================
# 5. PROCESS TEXT CHUNKS
# ============================================================
print("\nBuilding text chunks...")
text_chunks = []

for i, raw in enumerate(raw_chunks):
    text = raw.get("text", "").strip()
    if len(text) < 30:
        continue

    # Clean image codes
    text = IMG_CODE.sub("", text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    if len(text) < 30:
        continue

    page, bbox = parse_chunk_bbox(raw)
    headings   = get_headings(raw)

    # Nearest image via bbox
    if page and bbox:
        linked_imgs = nearest_images(page, bbox)
    elif page:
        # Fallback to all on page if no bbox parsed
        linked_imgs = [p["uri"] for p in page_pictures.get(page, [])][:2]
    else:
        linked_imgs = []

    # Content = section headings prepended so they get embedded
    content_parts = headings + [text]
    content = "\n".join(content_parts)

    chunk_id = sha_id("t", i, page, text[:80])

    text_chunks.append({
        "chunk_id": chunk_id,
        "content":  content,
        "meta": {
            "source_file":       "m11_SM.pdf",
            "page_no":           page,
            "chunk_type":        "text",
            "section_path":      headings,
            "linked_images":     linked_imgs,
            "bbox":              bbox,
            "parent_table_id":   None,
            "previous_chunk_id": None,   # filled in pass 2
            "next_chunk_id":     None,   # filled in pass 2
        }
    })

print(f"  → {len(text_chunks)} text chunks built")


# ============================================================
# 6. PROCESS TABLE CHUNKS (row + full-table)
# ============================================================
print("Building table chunks...")
table_chunks = []

for table in doc.get("tables", []):
    if table.get("label") == "document_index":
        continue
    grid = table.get("data", {}).get("grid", [])
    if not grid or len(grid) < 2:
        continue

    flat = " ".join(c.get("text", "") for row in grid for c in row)
    if not SPEC_UNITS.search(flat):
        continue

    prov       = table.get("prov", [{}])[0]
    table_page = prov.get("page_no")
    table_bbox = prov.get("bbox")
    headers    = [c.get("text", "") for c in grid[0]]

    # Resolve section from headings above table
    section = nearest_heading_above(table_page, table_bbox) if table_page and table_bbox else []

    # Linked images for the table bbox
    if table_page and table_bbox:
        table_imgs = nearest_images(table_page, table_bbox)
    else:
        table_imgs = []

    # --- Full-table chunk ---
    full_rows = []
    for row in grid[1:]:
        cells = [c.get("text", "").strip() for c in row if c.get("text", "").strip()]
        if cells:
            full_rows.append(" | ".join(cells))

    full_table_text = " > ".join(section) + "\n[TABLE]\n" + "\n".join(
        [" | ".join(h.ljust(20) for h in headers if h)] +
        full_rows
    )
    full_table_id = sha_id("table_full", table_page, flat[:80])

    table_chunks.append({
        "chunk_id": full_table_id,
        "content":  full_table_text,
        "meta": {
            "source_file":       "m11_SM.pdf",
            "page_no":           table_page,
            "chunk_type":        "table_full",
            "section_path":      section,
            "linked_images":     table_imgs,
            "bbox":              table_bbox,
            "parent_table_id":   None,
            "previous_chunk_id": None,
            "next_chunk_id":     None,
        }
    })

    # --- Row-by-row spec chunks (child chunks) ---
    for j, row in enumerate(grid[1:]):
        pairs = []
        for k in range(min(len(headers), len(row))):
            cell_text = row[k].get("text", "").strip()
            if cell_text:
                pairs.append(f"{headers[k]}: {cell_text}")
        if pairs:
            row_text = " > ".join(section) + "\n[Specs] " + " | ".join(pairs)
            row_id   = sha_id("table_row", table_page, j, " | ".join(pairs)[:80])

            table_chunks.append({
                "chunk_id": row_id,
                "content":  row_text,
                "meta": {
                    "source_file":       "m11_SM.pdf",
                    "page_no":           table_page,
                    "chunk_type":        "table_spec",
                    "section_path":      section,
                    "linked_images":     table_imgs,
                    "bbox":              table_bbox,
                    "parent_table_id":   full_table_id,
                    "previous_chunk_id": None,
                    "next_chunk_id":     None,
                }
            })

print(f"  → {len(table_chunks)} table chunks built (full + row-by-row)")


# ============================================================
# 7. MERGE & WIRE UP LINKED LIST
# ============================================================
all_chunks = text_chunks + table_chunks

# Only wire the linked list for text chunks (sequential order)
for i in range(len(text_chunks)):
    if i > 0:
        text_chunks[i]["meta"]["previous_chunk_id"] = text_chunks[i-1]["chunk_id"]
    if i < len(text_chunks) - 1:
        text_chunks[i]["meta"]["next_chunk_id"] = text_chunks[i+1]["chunk_id"]


# ============================================================
# 8. QUALITY REPORT
# ============================================================
from collections import Counter

types   = Counter(c["meta"]["chunk_type"] for c in all_chunks)
ids     = [c["chunk_id"] for c in all_chunks]
w_imgs  = sum(1 for c in all_chunks if c["meta"]["linked_images"])
w_sec   = sum(1 for c in all_chunks if c["meta"]["section_path"])
w_prev  = sum(1 for c in all_chunks if c["meta"]["previous_chunk_id"])
lengths = [len(c["content"]) for c in all_chunks]

print(f"\n{'='*55}")
print(f"  QUALITY REPORT v2")
print(f"{'='*55}")
print(f"  Total chunks:          {len(all_chunks)}")
print(f"  Types:                 {dict(types)}")
print(f"  Unique IDs:            {len(set(ids))} / {len(ids)}")
print(f"  With section_path:     {w_sec} ({100*w_sec/len(all_chunks):.1f}%)")
print(f"  With linked_images:    {w_imgs} ({100*w_imgs/len(all_chunks):.1f}%)")
print(f"  With previous_chunk:   {w_prev}")
print(f"  Content length:        min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")
print(f"{'='*55}")

# ============================================================
# 9. SAVE
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(all_chunks)} chunks → {OUTPUT_PATH}")
