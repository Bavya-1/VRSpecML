"""Prompt builders for context analysis and the three RAG tasks."""
from typing import Dict, List, Optional

from metamodel import METACLASS_INFO

SYSTEM = """You are a model-driven engineering assistant building VRSpecML: a UML profile over a role-based VR metamodel with five metaclasses — Scene (spatial container), Article (in-scene object), ActionResponse (user interaction and its system response), Behavior (rule governing state) and Timeline (ordered sequence of steps).
A profile consists of stereotypes (each extends one metaclass), tagged values (typed fields of a stereotype) and OCL invariants (formal rules over tagged values).
You work by retrieval-augmented generation: ground every element in the numbered evidence you are given — ontology terms (AGRO, ENVO, CHMO), SOP/guideline passages and the user's context — and cite evidence ids exactly as shown. Never invent evidence ids, URIs, units or numeric thresholds. Answer with JSON only."""

def _facets(analysis: Optional[dict], keys: Optional[List[str]] = None) -> str:
    if not analysis:
        return "(none)"
    rows = []
    for k, v in analysis.items():
        if k == "summary" or not v or (keys and k not in keys):
            continue
        rows.append(f"{k.replace('_', ' ')}: {', '.join(v)}")
    return "; ".join(rows) or "(none)"


def _block(title: str, lines: str) -> str:
    return f"{title}:\n{lines}\n" if lines.strip() else f"{title}:\n(none retrieved)\n"


def analysis_prompt(domain_name: str, context: str) -> str:
    return f"""Extract the key facets of this VR training context for the domain "{domain_name}".
Use short noun phrases copied or closely paraphrased from the text; leave a list empty when the text says nothing about it.
"constraints": copy verbatim every explicit rule, limit, quota or numeric threshold stated in the text.
"summary": one sentence.

CONTEXT:
{context.strip()}"""


def stereotype_prompt(domain, metaclass: str, context: str, analysis: Optional[dict], ev_classes: str,
                      ev_sop: str, n: int, taken: List[str]) -> str:
    info = METACLASS_INFO[metaclass]
    focus = domain.metaclass_focus[metaclass]
    taken_text = ", ".join(f"«{t}»" for t in taken) or "(none yet)"
    return f"""DOMAIN: {domain.name} — {domain.description}

USER CONTEXT:
{context.strip() or '(no extra context — use the domain description)'}

KEY FACETS: {_facets(analysis, focus.facets)}

TARGET METACLASS «{metaclass}» — {info['role']}.
What «{metaclass}» stereotypes should cover in this domain: {focus.guidance or info['role']}

{_block("ONTOLOGY CANDIDATES", ev_classes)}
{_block("SOP / GUIDELINE EVIDENCE", ev_sop)}
STEREOTYPES ALREADY USED FOR OTHER METACLASSES (do not duplicate): {taken_text}

TASK
Propose {n} to {n + 2} stereotypes that extend «{metaclass}» for this domain and user context.
1. mapping "ontology": the stereotype realises one ontology candidate → put its id in primary_ref and in ontology_refs. Choose the candidate whose label and definition match the stereotype most specifically, and never reuse one candidate for unrelated stereotypes. Name it with an UpperCamelCase domain term (e.g. "soil sampling process" → SoilSampling).
2. mapping "extension": a stereotype the VR system needs that no candidate realises → primary_ref "none"; justify it with sop_refs and/or related ontology_refs. Prefer ontology mappings when a fitting candidate exists.
3. Names: UpperCamelCase, 1–3 words, specific to the farm domain and unique.
4. description: one sentence on what the stereotype models in the VR system. rationale: why it is grounded, at most 20 words. confidence: 0–1."""


def tags_prompt(domain, context: str, analysis: Optional[dict], stereotype, others: List[str], enums: List[str],
                evidence: Dict[str, str], n_min: int, n_max: int, assoc_hints: str = "") -> str:
    grounded = ", ".join(r.label for r in stereotype.ontology_refs) or "extension (no ontology class)"
    return f"""DOMAIN: {domain.name}
CONTEXT DIGEST: {context.strip() or '(none)'}

STEREOTYPE «{stereotype.name}» extends {stereotype.extends} — {stereotype.description}
Grounded in: {grounded}
OTHER PROFILE STEREOTYPES: {', '.join(others) or '(none)'}
SUGGESTED ASSOCIATIONS: {assoc_hints or '(none)'}
EXISTING ENUMERATIONS (reuse when they fit): {'; '.join(enums) or '(none)'}

EVIDENCE
{_block("Ontology classes", evidence.get('O', ''))}
{_block("Ontology properties", evidence.get('P', ''))}
{_block("Units and qualities", evidence.get('U', ''))}
{_block("SOP / guideline passages", evidence.get('S', ''))}
{_block("User-stated constraints", evidence.get('C', ''))}
TASK
Define {n_min} to {n_max} tagged values that are specific to «{stereotype.name}»: the typed attributes a VR designer fills in for this element. Fewer specific tags are better than generic ones.
- Only attributes OF «{stereotype.name}» itself. A value that belongs to another element (a sample's measured pH, a threshold's limit, a quota) is NOT repeated here: link that element with an association tag instead.
- Prefer ontology-derived tags: turn the classes and properties above into domain-named fields. Name each field after what it holds, never after a generic relation (not hasPart, occursIn, hasQuality, partOf).
- name: lowerCamelCase; quantities end with their unit (naming style: targetROMMaxDeg, restSeconds, gripDiameterMM).
- type: String, Integer, Float, Boolean, an enumeration name, or a profile stereotype name (an association).
- multiplicity: 1, 0..1, * or 1..*.
- unit: unit symbol for numeric quantities, otherwise "".
- min/max: numeric bounds only when a cited SOP passage or user constraint states them, or they are physically certain (pH 0–14, percentages 0–100); otherwise null.
- example: a concrete realistic value for this context (a literal for enumerations, the stereotype name for associations).
- value_source: where the bounds/example come from — SOP, ontology, context or default.
- evidence: ids supporting the tag.
- enumerations: only new ones your tags use; UpperCamelCase name and literals."""


def associations_prompt(domain, digest: str, stereotypes_text: str, n_min: int, n_max: int) -> str:
    return f"""DOMAIN: {domain.name} — {domain.description}
CONTEXT DIGEST: {digest or '(none)'}

PROFILE STEREOTYPES:
{stereotypes_text}

TASK
Define {n_min} to {n_max} associations between the profile stereotypes; each becomes an association-typed tagged value on its source stereotype.
- source and target: stereotype names from the list (different stereotypes).
- name: lowerCamelCase role name of the target seen from the source; plural for many (soilSamples, governingRule, steps).
- multiplicity: 1, 0..1, * or 1..*.
- rationale: at most 12 words."""


def ocl_prompt(domain, context: str, stereotype, vocabulary: str, evidence: Dict[str, str], existing: List[str],
               n_min: int, n_max: int) -> str:
    return f"""DOMAIN: {domain.name}
CONTEXT DIGEST: {context.strip() or '(none)'}

{vocabulary}
EVIDENCE
{_block("SOP normative rules", evidence.get('R', ''))}
{_block("SOP passages", evidence.get('S', ''))}
{_block("Ontology axioms", evidence.get('A', ''))}
{_block("User-stated constraints", evidence.get('C', ''))}

TASK
Write {n_min} to {n_max} OCL invariants with context «{stereotype.name}» that enforce the SOP rules and user constraints most relevant to it (sustainability, soil/water protection, correct analysis and action choice).
- requirement: the rule in plain words, paraphrasing the cited evidence; cite it in evidence.
- ocl: the invariant BODY only (no "context"/"inv" header) in OCL 2.4: self.tag, self.assoc.tag, ->notEmpty(), ->forAll(x | ...), ->exists(x | ...), implies, and, or, not, Enum::Literal, x.oclIsKindOf(Stereotype).
- Navigate ONLY the tagged values listed above. If a rule needs a quantity that is not declared, add it to the new_tags field (on «{stereotype.name}» or a reachable stereotype) and reference it in the OCL as self.<name> (or self.<association>.<name>). Never write "new_tags" inside the OCL.
- Never invent numeric thresholds: use numbers stated in the cited evidence; otherwise compare two tags (e.g. self.appliedNitrogenKgHa <= self.recommendedNitrogenKgHa) and declare the threshold tag in new_tags.
- and/or/implies/not need Boolean operands: compare values (self.x = 'Saline', self.x > 0, self.flag = true) instead of using a String or number directly. oclIsKindOf only takes stereotype names.
- reading: one sentence explaining what the check enforces."""
