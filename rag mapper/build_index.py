"""Build (or rebuild) the RAG knowledge base from the domain's ontologies and SOPs.

Usage:  python build_index.py [project] [--force]
"""
import argparse
import logging
import sys

from domain import load_domain, load_project
from knowledge_base import build_knowledge_base, kb_status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project", nargs="?", help="project folder name under projects/ (default: shared Ontologies/ + SOP/)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the index is up to date")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    domain = load_project(args.project) if args.project else load_domain()
    status = kb_status(domain)
    if status["exists"] and not status["stale"] and not args.force:
        print(f"Index up to date ({status['built_at']}): {status['counts']}")
        return 0
    last = [-1]

    def progress(stage: str, frac: float):
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r{pct:3d}%  {stage:<40}", end="", flush=True)

    kb = build_knowledge_base(domain, progress)
    print(f"\nBuilt in {kb.manifest['build_seconds']}s: {kb.counts()}")
    for doc in kb.sop_docs:
        print(f"  SOP {doc.get('file', doc['doc'])}: {doc.get('chunks', 0)} chunks, {doc.get('rules', 0)} rules"
              + (f" — ERROR {doc['error']}" if doc.get("error") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
