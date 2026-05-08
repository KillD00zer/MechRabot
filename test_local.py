"""
MechRabot — Local Pipeline Test
Runs the full pipeline on your machine (no Modal, no cloud GPU needed).
BGE-M3 will run on your GTX 1050 Ti or fall back to CPU.

Usage:
    python test_local.py
    python test_local.py --query "ازاي اغير زيت الموتور؟"
"""

import argparse
from dotenv import load_dotenv

# Load .env before importing anything that reads env vars
load_dotenv()

from app.core.pipeline import build_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What is the torque for cylinder head cover bolts?",
        help="Question to ask MechRabot",
    )
    args = parser.parse_args()

    print(f"\n🔧 Building pipeline...")
    pipe = build_pipeline()

    print(f"🤖 Query: {args.query}\n")
    result = pipe.run({
        "refiner_prompt":   {"query": args.query},
        "generator_prompt": {"query": args.query},
    })

    answer = result["generator_llm"]["replies"][0].text
    docs   = result["retriever"]["documents"]

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print(f"\n📎 Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. [{doc.score:.3f}] {doc.meta.get('source_file')} — p.{doc.meta.get('page_no')}")


if __name__ == "__main__":
    main()
