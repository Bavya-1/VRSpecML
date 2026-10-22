"""Evidence packing used to ground Qwen's output.

EvidencePack numbers retrieved items with short ids (O1, S2, …) that the model must
cite; ids are enumerated in the JSON Schema and resolved back to full references.
"""
import re
from typing import Dict, Iterable, List, Optional, Tuple

from schemas import EvidenceRef

PREFIX_KIND = {"O": "class", "P": "property", "S": "chunk", "R": "rule", "C": "context",
               "A": "axiom", "U": "unit"}

_CATEGORY_LABEL = {"process": "process", "material": "material entity", "quality": "quality", "role": "role",
                   "information": "information entity", "site": "site"}


def _short(text: str, words: int) -> str:
    parts = (text or "").split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def class_ref(doc: dict) -> EvidenceRef:
    return EvidenceRef(kind="class", id=doc["id"], label=doc["label"], uri=doc["uri"], source=doc["source"],
                       text=_short(doc.get("definition", ""), 60))


class EvidencePack:
    def __init__(self):
        self.refs: Dict[str, EvidenceRef] = {}
        self.lines: Dict[str, List[str]] = {p: [] for p in PREFIX_KIND}
        self._seen: Dict[Tuple[str, str], str] = {}

    def add(self, prefix: str, ref: EvidenceRef, line: str) -> str:
        key = (prefix, ref.id)
        if key in self._seen:
            return self._seen[key]
        sid = f"{prefix}{len(self.lines[prefix]) + 1}"
        self._seen[key] = sid
        self.refs[sid] = ref
        self.lines[prefix].append(f"[{sid}] {line}")
        return sid

    def ids(self, *prefixes: str) -> List[str]:
        return [i for i in self.refs if i.rstrip("0123456789") in prefixes]

    def text(self, prefix: str) -> str:
        return "\n".join(self.lines[prefix])

    def resolve(self, ids: Iterable[str]) -> List[EvidenceRef]:
        out, seen = [], set()
        for i in ids or []:
            ref = self.refs.get(i)
            if ref is not None and (ref.kind, ref.id) not in seen:
                seen.add((ref.kind, ref.id))
                out.append(ref)
        return out

    # ── typed adders ────────────────────────────────────────────────────
    def add_class(self, doc: dict) -> str:
        rels = "; ".join(f"{r['property_label']} → {r['filler_label']}" for r in doc.get("restrictions", [])[:4])
        kind = _CATEGORY_LABEL.get(doc.get("category", ""), "")
        line = f"{doc['label']} ({doc['curie']}; {doc['source']}{'; ' + kind if kind else ''})"
        if doc.get("definition"):
            line += f" — {_short(doc['definition'], 28)}"
        if doc.get("parent_labels"):
            line += f" | is-a: {', '.join(doc['parent_labels'][:2])}"
        if rels:
            line += f" | rel: {rels}"
        return self.add("O", class_ref(doc), line)

    def add_property(self, doc: dict) -> str:
        line = f"{doc['label']} ({doc['curie']}, {doc['kind']} property)"
        if doc.get("definition"):
            line += f" — {_short(doc['definition'], 22)}"
        if doc.get("domain"):
            line += f" | domain: {', '.join(doc['domain'][:2])}"
        if doc.get("range"):
            line += f" | range: {', '.join(doc['range'][:2])}"
        if doc.get("usage"):
            c, f = doc["usage"][0]
            line += f" | e.g. {c} {doc['label']} {f}"
        ref = EvidenceRef(kind="property", id=doc["id"], label=doc["label"], uri=doc["uri"], source=doc["source"],
                          text=_short(doc.get("definition", ""), 40))
        return self.add("P", ref, line)

    def add_unit(self, doc: dict) -> str:
        line = f"{doc['label']} ({doc['curie']})"
        if doc.get("definition"):
            line += f" — {_short(doc['definition'], 14)}"
        ref = EvidenceRef(kind="unit", id=doc["id"], label=doc["label"], uri=doc["uri"], source=doc["source"],
                          text=_short(doc.get("definition", ""), 30))
        return self.add("U", ref, line)

    def add_chunk(self, doc: dict, words: int = 95) -> str:
        ref = EvidenceRef(kind="chunk", id=doc["id"], label=doc["label"], source=doc["doc"], text=doc["text"])
        return self.add("S", ref, f"{doc['label']}: \"{_short(doc['text'], words)}\"")

    def add_rule(self, doc: dict, words: int = 70) -> str:
        lead = f" ({_short(doc['lead'], 12)})" if doc.get("lead") else ""
        ref = EvidenceRef(kind="rule", id=doc["id"], label=doc["label"], source=doc["doc"], text=doc["text"])
        return self.add("R", ref, f"{doc['label']} [{doc['modality']}]{lead}: \"{_short(doc['text'], words)}\"")

    def add_context(self, n: int, text: str) -> str:
        ref = EvidenceRef(kind="context", id=f"context#{n}", label="user context", text=text)
        return self.add("C", ref, f"user context: \"{text}\"")

    def add_axiom(self, cls_label: str, cls_uri: str, restr: dict) -> str:
        quant = {"some": "∃", "only": "∀"}.get(restr["kind"].split(" ")[0], restr["kind"])
        dl = f"{cls_label} ⊑ {quant} {restr['property_label']}.({restr['filler_label']})"
        ref = EvidenceRef(kind="axiom", id=f"{cls_uri}|{restr['property']}|{restr['filler_label']}",
                          label=dl, uri=cls_uri, source="OWL axiom", text=dl)
        return self.add("A", ref, dl)
