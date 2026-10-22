"""Persistent RAG knowledge base: four corpora with hybrid dense (SBERT + FAISS) and
dense (SBERT) retrieval, diversified with MMR.

Corpora
  class     ontology classes (AGRO / ENVO / CHMO and the terms they import)
  property  ontology object / datatype properties
  chunk     SOP / guideline passages
  rule      normative SOP statements (should / must / avoid …)
"""
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import config
from domain import Domain
from ontology_ingest import class_text, ingest_ontologies, property_text
from sop_ingest import ingest_sop, sop_label

logger = logging.getLogger(__name__)

CORPORA = ("class", "property", "chunk", "rule")


def _kb_file(domain: Domain) -> str:
    return os.path.join(domain.index_dir, "kb.pkl")


def _manifest_file(domain: Domain) -> str:
    return os.path.join(domain.index_dir, "manifest.json")


Progress = Optional[Callable[[str, float], None]]
Query = Tuple[str, float, Optional[Dict[str, Any]]]  # (text, weight, filters)


def _source_fingerprint(domain: Domain) -> Dict[str, List[float]]:
    paths = [o.path for o in domain.ontologies] + domain.sop_paths()
    return {os.path.relpath(p, config.ROOT_DIR): [os.path.getsize(p), round(os.path.getmtime(p), 1)]
            for p in paths if os.path.exists(p)}


def kb_status(domain: Domain) -> Dict[str, Any]:
    """Whether a built index exists and still matches the domain sources."""
    if not (os.path.exists(_kb_file(domain)) and os.path.exists(_manifest_file(domain))):
        return {"exists": False, "stale": True, "reason": "not built", "counts": {}}
    with open(_manifest_file(domain), encoding="utf-8") as fh:
        manifest = json.load(fh)
    reason = ""
    if manifest.get("kb_version") != config.KB_VERSION:
        reason = "index format changed"
    elif manifest.get("embed_model") != config.EMBED_MODEL:
        reason = "embedding model changed"
    elif manifest.get("domain") != (domain.kb_id or domain.id):
        reason = "different domain"
    elif manifest.get("sources") != _source_fingerprint(domain):
        reason = "ontology/SOP files changed"
    return {"exists": True, "stale": bool(reason), "reason": reason,
            "counts": manifest.get("counts", {}), "built_at": manifest.get("built_at", "")}


class KnowledgeBase:
    def __init__(self, data: Dict[str, Any], embeddings: Dict[str, np.ndarray]):
        import faiss

        self.docs: Dict[str, List[Dict[str, Any]]] = data["docs"]
        self.labels: Dict[str, str] = data["labels"]
        self.children: Dict[str, List[str]] = data["children"]
        self.sop_docs: List[Dict[str, Any]] = data["sop_docs"]
        self.manifest: Dict[str, Any] = data["manifest"]
        self.emb = embeddings
        self.index = {}
        for corpus in CORPORA:
            vecs = embeddings[corpus]
            idx = faiss.IndexFlatIP(vecs.shape[1] if vecs.size else 384)
            if len(vecs):
                idx.add(vecs)
            self.index[corpus] = idx
        self.by_id = {corpus: {d["id"]: i for i, d in enumerate(self.docs[corpus])} for corpus in CORPORA}
        self.class_by_uri = {d["uri"]: d for d in self.docs["class"]}
        self.prop_by_uri = {d["uri"]: d for d in self.docs["property"]}

    # ── access ────────────────────────────────────────────────────────────
    def get(self, corpus: str, doc_id: str) -> Optional[Dict[str, Any]]:
        i = self.by_id[corpus].get(doc_id)
        return self.docs[corpus][i] if i is not None else None

    def vector(self, corpus: str, doc_id: str) -> Optional[np.ndarray]:
        i = self.by_id[corpus].get(doc_id)
        return self.emb[corpus][i] if i is not None else None

    def label(self, uri: str) -> str:
        return self.labels.get(uri, uri.rsplit("/", 1)[-1])

    def counts(self) -> Dict[str, int]:
        return {c: len(self.docs[c]) for c in CORPORA}

    # ── retrieval ─────────────────────────────────────────────────────────
    @staticmethod
    def _passes(doc: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        files = filters.get("files")
        if files and not set(doc.get("files", [])) & set(files):
            return False
        sources = filters.get("sources")
        if sources and doc.get("source") not in sources:
            return False
        exclude_sources = filters.get("exclude_sources")
        if exclude_sources and doc.get("source") in exclude_sources:
            return False
        docs = filters.get("docs")
        if docs and doc.get("doc") not in docs:
            return False
        return True

    def _rank(self, corpus: str, query: str, qvec: np.ndarray, filters) -> List[int]:
        """Single-query dense (SBERT) ranking of filtered docs."""
        docs = self.docs[corpus]
        n = len(docs)
        if not n or not query.strip():
            return []
        depth = n if filters else min(n, config.CANDIDATE_POOL)
        _, ids = self.index[corpus].search(qvec[None, :], depth)
        dense = [int(i) for i in ids[0] if i >= 0 and self._passes(docs[int(i)], filters)][: config.CANDIDATE_POOL]
        score: Dict[int, float] = defaultdict(float)
        for rank, i in enumerate(dense):
            score[i] += 1.0 / (config.RRF_K + rank + 1)
        if corpus in ("class", "property"):
            q = f" {query.lower()} "
            for i in list(score):
                lab = docs[i]["label"].lower()
                if len(lab) > 3 and f" {lab} " in q:
                    score[i] += 1.0 / (config.RRF_K + 1)  # exact concept mention
        return sorted(score, key=lambda i: -score[i])

    def search_many(self, corpus: str, queries: Sequence[Query], k: int = 10,
                    exclude_ids: Iterable[str] = (), mmr: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """Fuse several (query, weight, filters) rankings with RRF, then diversify with MMR."""
        from embedder import embed

        queries = [q for q in queries if q[0] and q[0].strip()]
        if not queries or not self.docs[corpus]:
            return []
        exclude = set(exclude_ids)
        qvecs = embed([q[0] for q in queries])
        fused: Dict[int, float] = defaultdict(float)
        for (text, weight, filters), qv in zip(queries, qvecs):
            for rank, i in enumerate(self._rank(corpus, text, qv, filters)[: config.CANDIDATE_POOL]):
                fused[i] += weight / (config.RRF_K + rank + 1)
        ranked = [i for i in sorted(fused, key=lambda i: -fused[i]) if self.docs[corpus][i]["id"] not in exclude]
        if not ranked:
            return []
        pool = ranked[: max(k * 4, k)]
        if not mmr or len(pool) <= k:
            chosen = pool[:k]
        else:
            vecs = self.emb[corpus][pool]
            top = fused[pool[0]]
            rel = np.array([fused[i] / top for i in pool])
            chosen_local: List[int] = [0]
            sims = vecs @ vecs[0]
            max_sim = sims.copy()
            while len(chosen_local) < k:
                mmr_score = config.MMR_LAMBDA * rel - (1 - config.MMR_LAMBDA) * max_sim
                mmr_score[chosen_local] = -np.inf
                nxt = int(np.argmax(mmr_score))
                chosen_local.append(nxt)
                max_sim = np.maximum(max_sim, vecs @ vecs[nxt])
            chosen = [pool[j] for j in chosen_local]
        return [(self.docs[corpus][i], fused[i]) for i in chosen]

    def search(self, corpus: str, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None,
               exclude_ids: Iterable[str] = (), mmr: bool = True):
        return self.search_many(corpus, [(query, 1.0, filters)], k=k, exclude_ids=exclude_ids, mmr=mmr)

    def similarity(self, text: str, corpus: str, doc_ids: Iterable[str]) -> float:
        """Max cosine similarity between a text and the given evidence docs (grounding score)."""
        from embedder import embed_one

        vecs = [v for v in (self.vector(corpus, d) for d in doc_ids) if v is not None]
        if not vecs:
            return 0.0
        return float(max(np.stack(vecs) @ embed_one(text)))


# ── build / load ──────────────────────────────────────────────────────────
def build_knowledge_base(domain: Domain, progress: Progress = None) -> KnowledgeBase:
    from embedder import embed

    def report(stage: str, frac: float):
        logger.info("%s (%.0f%%)", stage, frac * 100)
        if progress:
            progress(stage, frac)

    t0 = time.time()
    os.makedirs(domain.index_dir, exist_ok=True)
    onto = ingest_ontologies(
        [(o.key, o.path) for o in domain.ontologies],
        progress=lambda m, i, n: report(m, 0.05 + 0.3 * (i - 1) / n),
    )

    class_docs = []
    for rec in onto["classes"]:
        rec = dict(rec)
        rec["id"] = rec["curie"]
        rec["text"] = class_text(rec)
        class_docs.append(rec)
    prop_docs = []
    for rec in onto["properties"]:
        rec = dict(rec)
        rec["id"] = rec["curie"]
        rec["text"] = property_text(rec)
        prop_docs.append(rec)

    report("Reading SOPs", 0.38)
    chunk_docs, rule_docs, sop_docs = [], [], []
    for path in domain.sop_paths():
        try:
            sop = ingest_sop(path)
        except Exception as exc:  # unreadable document → skip, report in manifest
            logger.error("Failed to ingest %s: %s", path, exc)
            sop_docs.append({"doc": os.path.basename(path), "title": "", "pages": 0, "error": str(exc)})
            continue
        sop_docs.append({"doc": sop["doc"], "title": sop["title"], "pages": sop["pages"],
                         "chunks": len(sop["chunks"]), "rules": len(sop["rules"]),
                         "file": os.path.basename(path)})
        for n, ch in enumerate(sop["chunks"], start=1):
            chunk_docs.append({**ch, "id": f"{sop['doc']}#c{n}", "label": sop_label(ch)})
        for n, r in enumerate(sop["rules"], start=1):
            rule_docs.append({**r, "id": f"{sop['doc']}#r{n}", "label": sop_label(r)})

    docs = {"class": class_docs, "property": prop_docs, "chunk": chunk_docs, "rule": rule_docs}
    embeddings: Dict[str, np.ndarray] = {}
    total = sum(len(v) for v in docs.values()) or 1
    done = 0
    for corpus in CORPORA:
        texts = [d["text"] for d in docs[corpus]]
        base = done

        def on_batch(n, _t, base=base, corpus=corpus):
            report(f"Embedding {corpus} corpus", 0.4 + 0.55 * (base + n) / total)

        embeddings[corpus] = embed(texts, progress=on_batch)
        done += len(texts)
        np.save(os.path.join(domain.index_dir, f"emb_{corpus}.npy"), embeddings[corpus])

    manifest = {
        "kb_version": config.KB_VERSION,
        "embed_model": config.EMBED_MODEL,
        "domain": domain.kb_id or domain.id,
        "sources": _source_fingerprint(domain),
        "counts": {c: len(docs[c]) for c in CORPORA},
        "sop_docs": sop_docs,
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "build_seconds": round(time.time() - t0, 1),
    }
    data = {"docs": docs, "labels": onto["labels"], "children": onto["children"],
            "sop_docs": sop_docs, "manifest": manifest}
    tmp = _kb_file(domain) + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _kb_file(domain))
    with open(_manifest_file(domain), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    report("Ready", 1.0)
    return KnowledgeBase(data, embeddings)


class _Skipped:
    """Placeholder for classes (BM25) that older index files pickled."""


class _Unpickler(pickle.Unpickler):
    """Skips the BM25 objects stored by older index files."""

    def find_class(self, module, name):
        return _Skipped if module == "bm25" else super().find_class(module, name)


def load_knowledge_base(domain: Domain) -> Optional[KnowledgeBase]:
    if not os.path.exists(_kb_file(domain)):
        return None
    with open(_kb_file(domain), "rb") as fh:
        data = _Unpickler(fh).load()
    embeddings = {}
    for corpus in CORPORA:
        path = os.path.join(domain.index_dir, f"emb_{corpus}.npy")
        if not os.path.exists(path):
            return None
        embeddings[corpus] = np.load(path)
        if len(embeddings[corpus]) != len(data["docs"][corpus]):
            return None
    return KnowledgeBase(data, embeddings)
