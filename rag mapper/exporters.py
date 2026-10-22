"""Profile exports: JSON catalog, .ocl file, Markdown catalog, Graphviz profile diagram,
plus requirement → element → invariant traceability and a residue log."""
import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from metamodel import METACLASSES
from schemas import EvidenceRef, Profile


def _ref(r: Optional[EvidenceRef]) -> Optional[Dict[str, Any]]:
    if r is None:
        return None
    out = {"kind": r.kind, "id": r.id, "label": r.label}
    if r.uri:
        out["uri"] = r.uri
    if r.source:
        out["source"] = r.source
    if r.kind in ("chunk", "rule", "context") and r.text:
        out["quote"] = r.text[:400]
    return out


def _valid_invariants(profile: Profile):
    kept = {s.sid for s in profile.kept()}
    return [i for i in profile.invariants if i.context_sid in kept]


def traceability(profile: Profile) -> List[Dict[str, Any]]:
    """Requirement (SOP / context / ontology) → context stereotype → invariant."""
    rows = []
    for inv in _valid_invariants(profile):
        sources = [e.label for e in inv.evidence] or ["(uncited)"]
        rows.append({"requirement": inv.requirement or inv.reading, "sources": sources,
                     "elements": [f"«{inv.context}»"], "invariant": f"{inv.iid} {inv.name}"})
    return rows


def residue_log(profile: Profile) -> List[Dict[str, str]]:
    """Elements without an ontology counterpart."""
    rows = []
    for st in profile.kept():
        if st.mapping == "extension":
            why = st.rationale or "domain extension"
            refs = ", ".join(r.label for r in st.sop_refs + st.ontology_refs) or "context only"
            rows.append({"element": f"«{st.name}»", "kind": "LLM-suggested stereotype", "detail": f"{why} [{refs}]"})
        for t in profile.tagged_values.get(st.sid, []):
            if not t.evidence:
                rows.append({"element": f"«{st.name}».{t.name}", "kind": "uncited tag",
                             "detail": t.description or "no cited evidence"})
    return rows


def _guideline_refs(refs: Sequence[EvidenceRef]) -> List[str]:
    """Ids of cited guideline passages / rules (DOCUMENT#c1, DOCUMENT#r2), without duplicates."""
    return list(dict.fromkeys(r.id for r in refs if r.kind in ("chunk", "rule")))


def to_json(profile: Profile, kb_manifest: Optional[Dict[str, Any]] = None, domain=None) -> str:
    """Generic profile catalog: project metadata, stereotypes with tagged values, enumerations,
    OCL invariants and requirement traceability."""
    stereotypes = []
    for st in profile.kept():
        ontology_class = None
        if st.primary:
            ontology_class = {"id": st.primary.id, "label": st.primary.label, "uri": st.primary.uri}
        stereotypes.append({
            "name": st.name, "extends": st.extends, "mapping": st.mapping, "description": st.description,
            "ontology_class": ontology_class,
            "guideline_refs": _guideline_refs(st.sop_refs),
            "tagged_values": [{
                "name": t.name, "type": t.type, "multiplicity": t.multiplicity, "unit": t.unit,
                "min": t.min, "max": t.max, "description": t.description,
            } for t in profile.tagged_values.get(st.sid, [])],
        })
    invariants = [{
        "id": i.iid, "name": i.name, "context": i.context,
        "ocl": f"context {i.context} inv {i.name}: {i.body}",
        "requirement": i.requirement or i.reading,
        "guideline_refs": _guideline_refs(i.evidence),
    } for i in _valid_invariants(profile)]
    if domain:
        guideline_docs = [os.path.basename(p) for p in domain.sop_paths()]
    else:
        guideline_docs = [d.get("file", d.get("doc", "")) for d in (kb_manifest or {}).get("sop_docs", [])]
    data = {
        "metadata": {
            "project": profile.domain_name, "package": profile.package, "model": profile.model,
            "created": profile.created, "updated": profile.updated,
            "ontologies": [{"key": o.key, "file": o.file} for o in domain.ontologies] if domain else [],
            "guideline_documents": guideline_docs,
        },
        "context": profile.context,
        "stereotypes": stereotypes,
        "enumerations": [{"name": e.name, "literals": e.literals, "description": e.description}
                         for e in profile.enumerations],
        "invariants": invariants,
        "traceability": traceability(profile),
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_ocl(profile: Profile) -> str:
    pkg = profile.package.split("::")[-1]
    lines = [f"-- {profile.package} — OCL invariants (VRSpecML RAG profile generator)",
             f"-- Domain: {profile.domain_name} | Model: {profile.model} | {profile.updated}", "",
             f"package {pkg}", ""]
    for inv in _valid_invariants(profile):
        lines.append(f"-- {inv.iid} [{inv.origin}] {inv.reading or inv.requirement}")
        for e in inv.evidence[:3]:
            lines.append(f"--   source: {e.label}")
        lines += [f"context {inv.context}", f"inv {inv.name}: {inv.body}", ""]
    lines.append("endpackage")
    return "\n".join(lines) + "\n"


def _md(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def to_markdown(profile: Profile) -> str:
    out = [f"# {profile.package} — Profile catalog", "",
           f"*Domain:* {profile.domain_name} · *Model:* {profile.model} · *Updated:* {profile.updated}", ""]
    if profile.context:
        out += ["## Context", "", profile.context, ""]
    if profile.enumerations:
        out += ["## Enumerations", "", "| Enumeration | Literals | Provenance |", "|---|---|---|"]
        for e in profile.enumerations:
            prov = ", ".join(x.label for x in e.evidence) or "—"
            out.append(f"| {e.name} | {_md(', '.join(e.literals))} | {_md(prov)} |")
        out.append("")
    out += ["## Stereotypes", ""]
    for m in METACLASSES:
        group = [s for s in profile.kept() if s.extends == m]
        if not group:
            continue
        out += [f"### {m}", ""]
        for st in group:
            prov = st.primary.label + f" ({st.primary.source}: {st.primary.id})" if st.primary else "extension"
            out += [f"#### «{st.name}» — extends {st.extends}", "", st.description, "",
                    f"*Provenance:* {_md(prov)}" + (f" · *SOP:* {_md('; '.join(r.label for r in st.sop_refs))}" if st.sop_refs else ""), ""]
            tags = profile.tagged_values.get(st.sid, [])
            if tags:
                out += ["| Tagged value | Type | Mult. | Unit / range | Example | Source |", "|---|---|---|---|---|---|"]
                for t in tags:
                    rng = t.unit
                    if t.min is not None or t.max is not None:
                        rng = f"{t.unit} [{'' if t.min is None else f'{t.min:g}'}..{'' if t.max is None else f'{t.max:g}'}]".strip()
                    src = ", ".join(e.label for e in t.evidence[:2]) or t.value_source
                    out.append(f"| {t.name} | {t.type} | {t.multiplicity} | {_md(rng or '—')} | {_md(t.example or '—')} "
                               f"| {_md(src)}{'' if t.verified else ' ⚠'} |")
                out.append("")
    invs = _valid_invariants(profile)
    if invs:
        out += ["## OCL invariants", "", "| ID | Name | Context |", "|---|---|---|"]
        for i in invs:
            out.append(f"| {i.iid} | {i.name} | «{i.context}» |")
        out.append("")
        for i in invs:
            out += [f"**{i.iid} {i.name}** — {i.reading or i.requirement}", "", "```ocl",
                    f"context {i.context}", f"inv {i.name}: {i.body}", "```"]
            if i.evidence:
                out.append("Source: " + "; ".join(e.label for e in i.evidence))
            out.append("")
        trace = traceability(profile)
        out += ["## Traceability", "", "| Requirement | Elements | Invariant |", "|---|---|---|"]
        for r in trace:
            out.append(f"| {_md(r['requirement'])} | {_md(', '.join(r['elements']))} | {r['invariant']} |")
        out.append("")
    residue = residue_log(profile)
    if residue:
        out += ["## Residue log", "", "| Element | Kind | Detail |", "|---|---|---|"]
        for r in residue:
            out.append(f"| {_md(r['element'])} | {r['kind']} | {_md(r['detail'])} |")
        out.append("")
    return "\n".join(out)


_FILL = {"Scene": "#dcfce7", "Article": "#dbeafe", "ActionResponse": "#fef3c7", "Behavior": "#fee2e2", "Timeline": "#ede9fe"}


def to_dot(profile: Profile, metaclasses: Optional[Sequence[str]] = None, max_tags: int = 7,
           show_enums: bool = True) -> str:
    esc = html.escape
    chosen = [s for s in profile.kept() if not metaclasses or s.extends in metaclasses]
    names = {s.name: s for s in chosen}
    lines = ['digraph Profile {', '  rankdir=BT; bgcolor="transparent"; nodesep=0.3; ranksep=0.7;',
             '  node [shape=plain, fontname="Helvetica", fontsize=10];',
             '  edge [fontname="Helvetica", fontsize=8, color="#64748b"];']
    for m in METACLASSES:
        if any(s.extends == m for s in chosen):
            lines.append(f'  "{m}" [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#f1f5f9">'
                         f'<tr><td><font point-size="8">«metaclass»</font></td></tr><tr><td><b>{m}</b></td></tr></table>>];')
    used_enums = set()
    for st in chosen:
        tags = profile.tagged_values.get(st.sid, [])
        rows = []
        for t in tags[:max_tags]:
            used_enums.add(t.type)
            rows.append(f'<tr><td align="left">{esc(t.name)} : {esc(t.type)} [{esc(t.multiplicity)}]</td></tr>')
        if len(tags) > max_tags:
            rows.append(f'<tr><td align="left"><i>+{len(tags) - max_tags} more</i></td></tr>')
        body = "".join(rows) or '<tr><td> </td></tr>'
        lines.append(f'  "{st.sid}" [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff">'
                     f'<tr><td bgcolor="{_FILL[st.extends]}"><font point-size="8">«stereotype»</font><br/><b>{esc(st.name)}</b></td></tr>'
                     f'{body}</table>>];')
        lines.append(f'  "{st.sid}" -> "{st.extends}" [arrowhead=normal, color="#334155"];')
        for t in tags:
            if t.type in names and names[t.type].sid != st.sid:
                lines.append(f'  "{st.sid}" -> "{names[t.type].sid}" [arrowhead=vee, style=dashed, '
                             f'label="{esc(t.name)} [{esc(t.multiplicity)}]", constraint=false];')
    if show_enums:
        for e in profile.enumerations:
            if e.name not in used_enums:
                continue
            lits = "".join(f'<tr><td align="left">{esc(l)}</td></tr>' for l in e.literals[:8])
            lines.append(f'  "enum_{e.name}" [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#fafafa">'
                         f'<tr><td><font point-size="8">«enumeration»</font><br/><b>{esc(e.name)}</b></td></tr>{lits}</table>>];')
    lines.append("}")
    return "\n".join(lines)
