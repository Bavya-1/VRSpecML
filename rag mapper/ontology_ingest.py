"""OWL ingestion with RDFLib: classes, restrictions, object/datatype properties.

Each domain ontology file is parsed separately so every term remembers which
domain ontology (AGRO / ENVO / CHMO) it came from; terms shared across files
(e.g. ENVO classes imported by AGRO) are merged by URI.
"""
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

import config

logger = logging.getLogger(__name__)

IAO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
SYNONYM_PREDICATES = (
    OBOINOWL.hasExactSynonym, OBOINOWL.hasRelatedSynonym,
    OBOINOWL.hasNarrowSynonym, OBOINOWL.hasBroadSynonym, SKOS.altLabel,
)
_OBO_ID = re.compile(r"/obo/([A-Za-z]+)_([0-9A-Za-z]+)$")
_OBO = "http://purl.obolibrary.org/obo/"
# BFO / PATO / IAO roots → coarse upper category used by O2P role classification (step 3)
UPPER_CATEGORIES = {_OBO + k: v for k, v in {
    "BFO_0000015": "process", "BFO_0000003": "process", "BFO_0000035": "process", "BFO_0000008": "process",
    "BFO_0000182": "process",
    "BFO_0000019": "quality", "BFO_0000016": "quality", "BFO_0000034": "quality", "BFO_0000020": "quality",
    "PATO_0000001": "quality",
    "BFO_0000023": "role", "BFO_0000017": "role",
    "BFO_0000029": "site", "BFO_0000006": "site", "BFO_0000140": "site",
    "BFO_0000040": "material", "BFO_0000030": "material", "BFO_0000024": "material", "BFO_0000027": "material",
    "IAO_0000030": "information", "BFO_0000031": "information",
}.items()}


def _upper_category(uri: str, classes: Dict[str, Dict[str, Any]], memo: Dict[str, Optional[str]]) -> Optional[str]:
    """Walk named superclasses (breadth-first) until a BFO/PATO/IAO root is reached."""
    if uri in memo:
        return memo[uri]
    frontier, seen, found = [uri], {uri}, None
    while frontier and found is None and len(seen) < 400:
        nxt = []
        for u in frontier:
            if u in UPPER_CATEGORIES:
                found = UPPER_CATEGORIES[u]
                break
            if u in memo and memo[u]:
                found = memo[u]
                break
            for p in classes.get(u, {}).get("parents", []):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
    memo[uri] = found
    return found
_HASH_NS = re.compile(r"/obo/([A-Za-z]+)#")


def local_name(uri: str) -> str:
    return re.split(r"[#/]", uri.rstrip("/"))[-1]


def prefix_of(uri: str, fallback: str = "") -> str:
    m = _OBO_ID.search(uri)
    if m:
        return m.group(1).upper()
    m = _HASH_NS.search(uri)
    if m:
        return m.group(1).upper()
    return fallback or "ONTOLOGY"


def curie_of(uri: str, fallback: str = "") -> str:
    m = _OBO_ID.search(uri)
    return f"{m.group(1).upper()}:{m.group(2)}" if m else f"{prefix_of(uri, fallback)}:{local_name(uri)}"


def _humanize(identifier: str) -> str:
    s = identifier.replace("_", " ").replace("-", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.strip()


def _english(values: Iterable) -> List[str]:
    out = []
    for v in values:
        if isinstance(v, Literal) and v.language not in (None, "en", "en-us", "en-gb"):
            continue
        text = " ".join(str(v).split())
        if text and text not in out:
            out.append(text)
    return out


def _first(g: Graph, node, *preds) -> str:
    for p in preds:
        vals = _english(g.objects(node, p))
        if vals:
            return vals[0]
    return ""


class _FileParser:
    """Extracts terms from one ontology graph."""

    def __init__(self, graph: Graph, key: str):
        self.g = graph
        self.key = key
        self.labels: Dict[str, str] = {}
        for s, o in graph.subject_objects(RDFS.label):
            if isinstance(s, URIRef) and (not isinstance(o, Literal) or o.language in (None, "en", "en-us", "en-gb")):
                self.labels.setdefault(str(s), " ".join(str(o).split()))

    def label(self, node) -> str:
        if isinstance(node, URIRef):
            return self.labels.get(str(node)) or _humanize(local_name(str(node)))
        return self.describe(node)

    def _list(self, node) -> List:
        try:
            return list(Collection(self.g, node))
        except Exception:  # malformed RDF list
            return []

    def describe(self, node, depth: int = 0) -> str:
        """Readable Manchester-like text for an anonymous class expression."""
        if isinstance(node, URIRef):
            return self.label(node)
        if node is None or depth > 3:
            return "…"
        g = self.g
        for pred, joiner in ((OWL.unionOf, " or "), (OWL.intersectionOf, " and ")):
            members = g.value(node, pred)
            if members is not None:
                parts = [self.describe(m, depth + 1) for m in self._list(members)]
                return "(" + joiner.join(p for p in parts if p) + ")"
        comp = g.value(node, OWL.complementOf)
        if comp is not None:
            return "not " + self.describe(comp, depth + 1)
        one_of = g.value(node, OWL.oneOf)
        if one_of is not None:
            return "{" + ", ".join(self.label(m) for m in self._list(one_of)) + "}"
        restr = self.restriction(node)
        if restr:
            return f"({restr['property_label']} {restr['kind']} {restr['filler_label']})"
        return "…"

    def restriction(self, node) -> Optional[Dict[str, Any]]:
        g = self.g
        prop = g.value(node, OWL.onProperty)
        if prop is None or not isinstance(prop, URIRef):
            return None
        kind, filler = None, None
        for pred, k in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only"), (OWL.hasValue, "value")):
            val = g.value(node, pred)
            if val is not None:
                kind, filler = k, val
                break
        if kind is None:
            for pred, k in (
                (OWL.minQualifiedCardinality, "min"), (OWL.maxQualifiedCardinality, "max"),
                (OWL.qualifiedCardinality, "exactly"), (OWL.minCardinality, "min"),
                (OWL.maxCardinality, "max"), (OWL.cardinality, "exactly"),
            ):
                val = g.value(node, pred)
                if val is not None:
                    kind = f"{k} {val}"
                    filler = g.value(node, OWL.onClass) or g.value(node, OWL.onDataRange)
                    break
        if kind is None:
            return None
        if isinstance(filler, Literal):
            filler_uri, filler_label = None, str(filler)
        elif isinstance(filler, URIRef):
            filler_uri, filler_label = str(filler), self.label(filler)
        elif filler is None:
            filler_uri, filler_label = None, "Thing"
        else:
            filler_uri, filler_label = None, self.describe(filler, 1)
        return {
            "property": str(prop),
            "property_label": self.label(prop),
            "kind": kind,
            "filler": filler_uri,
            "filler_label": filler_label,
        }

    def _is_deprecated(self, node, label: str) -> bool:
        if any(str(v).lower() == "true" for v in self.g.objects(node, OWL.deprecated)):
            return True
        return label.lower().startswith("obsolete")

    def classes(self) -> Iterable[Dict[str, Any]]:
        g = self.g
        for cls in set(g.subjects(RDF.type, OWL.Class)):
            if not isinstance(cls, URIRef) or cls == OWL.Thing:
                continue
            uri = str(cls)
            label = self.labels.get(uri, "")
            definition = _first(g, cls, IAO_DEFINITION, SKOS.definition, RDFS.comment)
            if not label and not definition:
                continue  # bare imported reference, no content to ground on
            label = label or _humanize(local_name(uri))
            if self._is_deprecated(cls, label):
                continue
            synonyms: List[str] = []
            for p in SYNONYM_PREDICATES:
                for s in _english(g.objects(cls, p)):
                    if s.lower() != label.lower() and s not in synonyms:
                        synonyms.append(s)
            parents, restrictions = [], []
            for sup in g.objects(cls, RDFS.subClassOf):
                if isinstance(sup, URIRef):
                    if sup != OWL.Thing:
                        parents.append(str(sup))
                else:
                    r = self.restriction(sup)
                    if r:
                        restrictions.append(r)
            for eq in g.objects(cls, OWL.equivalentClass):
                if isinstance(eq, URIRef):
                    continue
                members = g.value(eq, OWL.intersectionOf)
                for m in self._list(members) if members is not None else []:
                    if isinstance(m, URIRef):
                        parents.append(str(m))
                    else:
                        r = self.restriction(m)
                        if r:
                            r["kind"] = r["kind"] + " (defining)"
                            restrictions.append(r)
            yield {
                "uri": uri,
                "label": label,
                "definition": definition,
                "synonyms": synonyms[:8],
                "parents": list(dict.fromkeys(parents)),
                "restrictions": restrictions,
            }

    def properties(self) -> Iterable[Dict[str, Any]]:
        g = self.g
        for rdf_type, kind in ((OWL.ObjectProperty, "object"), (OWL.DatatypeProperty, "datatype")):
            for prop in set(g.subjects(RDF.type, rdf_type)):
                if not isinstance(prop, URIRef):
                    continue
                uri = str(prop)
                definition = _first(g, prop, IAO_DEFINITION, SKOS.definition, RDFS.comment)
                if not self.labels.get(uri) and not definition:
                    continue  # unlabeled imported stub
                label = self.labels.get(uri) or _humanize(local_name(uri))
                if self._is_deprecated(prop, label):
                    continue
                inverse = g.value(prop, OWL.inverseOf)
                chars = [c for c, t in (
                    ("functional", OWL.FunctionalProperty), ("transitive", OWL.TransitiveProperty),
                    ("symmetric", OWL.SymmetricProperty),
                ) if (prop, RDF.type, t) in g]
                yield {
                    "uri": uri,
                    "label": label,
                    "kind": kind,
                    "definition": definition,
                    "domain": [self.label(d) for d in g.objects(prop, RDFS.domain)],
                    "range": [self.label(r) for r in g.objects(prop, RDFS.range)],
                    "inverse": self.label(inverse) if inverse is not None else "",
                    "parents": [self.label(p) for p in g.objects(prop, RDFS.subPropertyOf) if isinstance(p, URIRef)],
                    "characteristics": chars,
                }


def _merge_list(a: List, b: List, key=lambda x: x) -> List:
    seen = {key(x) for x in a}
    for x in b:
        k = key(x)
        if k not in seen:
            seen.add(k)
            a.append(x)
    return a


def ingest_ontologies(sources: List[Tuple[str, str]], progress=None) -> Dict[str, Any]:
    """Parse [(key, path), ...] into merged class/property records.

    Returns {"classes", "properties", "labels", "children"}.
    """
    classes: Dict[str, Dict[str, Any]] = {}
    props: Dict[str, Dict[str, Any]] = {}
    labels: Dict[str, str] = {}

    for i, (key, path) in enumerate(sources, start=1):
        if progress:
            progress(f"Parsing {key}", i, len(sources))
        g = Graph()
        fmt = "turtle" if path.lower().endswith(".ttl") else "xml"
        g.parse(path, format=fmt)
        parser = _FileParser(g, key)
        for uri, lab in parser.labels.items():
            labels.setdefault(uri, lab)
        for rec in parser.classes():
            existing = classes.get(rec["uri"])
            if existing is None:
                rec["files"] = [key]
                classes[rec["uri"]] = rec
                continue
            _merge_list(existing["files"], [key])
            existing["definition"] = existing["definition"] or rec["definition"]
            _merge_list(existing["synonyms"], rec["synonyms"])
            _merge_list(existing["parents"], rec["parents"])
            _merge_list(existing["restrictions"], rec["restrictions"],
                        key=lambda r: (r["property"], r["kind"], r["filler"] or r["filler_label"]))
        for rec in parser.properties():
            existing = props.get(rec["uri"])
            if existing is None:
                rec["files"] = [key]
                props[rec["uri"]] = rec
                continue
            _merge_list(existing["files"], [key])
            for field in ("domain", "range", "parents", "characteristics"):
                _merge_list(existing[field], rec[field])
            existing["definition"] = existing["definition"] or rec["definition"]
            existing["inverse"] = existing["inverse"] or rec["inverse"]

    children: Dict[str, List[str]] = defaultdict(list)
    usage: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    memo: Dict[str, Optional[str]] = {}
    for uri, rec in classes.items():
        rec["category"] = _upper_category(uri, classes, memo) or ""
        rec["source"] = prefix_of(uri, rec["files"][0])
        rec["curie"] = curie_of(uri, rec["files"][0])
        rec["parent_labels"] = [labels.get(p, _humanize(local_name(p))) for p in rec["parents"]]
        for p in rec["parents"]:
            children[p].append(uri)
        for r in rec["restrictions"]:
            if len(usage[r["property"]]) < 40:
                usage[r["property"]].append((rec["label"], r["filler_label"]))

    class_list = []
    for uri, rec in classes.items():
        rec["n_children"] = len(children.get(uri, []))
        if rec["source"] in config.EXCLUDED_CLASS_PREFIXES:
            continue
        class_list.append(rec)

    prop_list = []
    for uri, rec in props.items():
        rec["source"] = prefix_of(uri, rec["files"][0])
        rec["curie"] = curie_of(uri, rec["files"][0])
        rec["usage"] = usage.get(uri, [])[:12]
        rec["n_usage"] = len(usage.get(uri, []))
        prop_list.append(rec)

    class_list.sort(key=lambda r: r["uri"])
    prop_list.sort(key=lambda r: r["uri"])
    return {
        "classes": class_list,
        "properties": prop_list,
        "labels": labels,
        "children": {k: sorted(v) for k, v in children.items()},
    }


def class_text(rec: Dict[str, Any]) -> str:
    """Retrieval text for an ontology class."""
    parts = [rec["label"] + "."]
    if rec.get("synonyms"):
        parts.append("Synonyms: " + ", ".join(rec["synonyms"][:5]) + ".")
    if rec.get("definition"):
        parts.append(rec["definition"][:320])
    if rec.get("parent_labels"):
        parts.append("Is a: " + ", ".join(rec["parent_labels"][:3]) + ".")
    rels = [f"{r['property_label']} {r['filler_label']}" for r in rec.get("restrictions", [])[:5]]
    if rels:
        parts.append("Relations: " + "; ".join(rels) + ".")
    return " ".join(parts)


def property_text(rec: Dict[str, Any]) -> str:
    """Retrieval text for an ontology property."""
    parts = [f"{rec['label']} ({rec['kind']} property)."]
    if rec.get("definition"):
        parts.append(rec["definition"][:280])
    if rec.get("domain"):
        parts.append("Domain: " + ", ".join(rec["domain"][:3]) + ".")
    if rec.get("range"):
        parts.append("Range: " + ", ".join(rec["range"][:3]) + ".")
    if rec.get("usage"):
        parts.append("Used as: " + "; ".join(f"{c} {rec['label']} {f}" for c, f in rec["usage"][:4]) + ".")
    return " ".join(parts)
