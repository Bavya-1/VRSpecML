"""Profile data model (Pydantic) and JSON Schemas for schema-constrained Qwen output.

Evidence ids offered to the model are enumerated inside the JSON Schema, so the
grammar-constrained decoder can only cite evidence that was actually retrieved.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from metamodel import METACLASSES, MULTIPLICITIES

Metaclass = Literal["Scene", "Article", "ActionResponse", "Behavior", "Timeline"]
ValueSource = Literal["SOP", "ontology", "context", "default"]
EvidenceKind = Literal["class", "property", "chunk", "rule", "context", "axiom", "seed", "unit", "partition"]


# ── Profile model ────────────────────────────────────────────────────────────
class EvidenceRef(BaseModel):
    kind: EvidenceKind
    id: str
    label: str
    uri: Optional[str] = None
    source: Optional[str] = None
    text: str = ""


class Stereotype(BaseModel):
    sid: str
    name: str
    extends: Metaclass
    mapping: Literal["ontology", "extension"] = "extension"
    description: str = ""
    rationale: str = ""
    confidence: float = 0.5
    primary: Optional[EvidenceRef] = None
    ontology_refs: List[EvidenceRef] = Field(default_factory=list)
    sop_refs: List[EvidenceRef] = Field(default_factory=list)
    grounding: float = 0.0
    keep: bool = True
    notes: List[str] = Field(default_factory=list)


class TaggedValue(BaseModel):
    name: str
    type: str
    multiplicity: str = "1"
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    example: str = ""
    description: str = ""
    value_source: ValueSource = "default"
    rule: str = "extension"
    evidence: List[EvidenceRef] = Field(default_factory=list)
    verified: bool = False
    origin: Literal["task2", "task3"] = "task2"
    notes: List[str] = Field(default_factory=list)


class Enumeration(BaseModel):
    name: str
    literals: List[str]
    description: str = ""
    evidence: List[EvidenceRef] = Field(default_factory=list)


class Invariant(BaseModel):
    iid: str
    name: str
    context: str
    context_sid: str
    body: str
    reading: str = ""
    requirement: str = ""
    evidence: List[EvidenceRef] = Field(default_factory=list)
    origin: str = "llm"
    grounding: float = 0.0


class ContextAnalysis(BaseModel):
    summary: str = ""
    farm_setting: List[str] = Field(default_factory=list)
    crops: List[str] = Field(default_factory=list)
    soil_issues: List[str] = Field(default_factory=list)
    water_issues: List[str] = Field(default_factory=list)
    chemical_tests: List[str] = Field(default_factory=list)
    agricultural_actions: List[str] = Field(default_factory=list)
    environmental_impacts: List[str] = Field(default_factory=list)
    equipment: List[str] = Field(default_factory=list)
    climate_location: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


FACETS = [f for f in ContextAnalysis.model_fields if f != "summary"]


class Profile(BaseModel):
    domain_id: str
    domain_name: str
    package: str
    context: str = ""
    analysis: Optional[ContextAnalysis] = None
    stereotypes: List[Stereotype] = Field(default_factory=list)
    tagged_values: Dict[str, List[TaggedValue]] = Field(default_factory=dict)   # sid → tags
    enumerations: List[Enumeration] = Field(default_factory=list)
    invariants: List[Invariant] = Field(default_factory=list)
    task_log: List[Dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    created: str = ""
    updated: str = ""

    # helpers
    def kept(self) -> List[Stereotype]:
        return [s for s in self.stereotypes if s.keep]

    def by_sid(self, sid: str) -> Optional[Stereotype]:
        return next((s for s in self.stereotypes if s.sid == sid), None)

    def by_name(self, name: str) -> Optional[Stereotype]:
        return next((s for s in self.stereotypes if s.name == name), None)

    def enum(self, name: str) -> Optional[Enumeration]:
        return next((e for e in self.enumerations if e.name == name), None)


# ── JSON Schemas for Qwen output ─────────────────────────────────────────────
def _ids(ids: List[str], max_items: int = 8) -> Dict[str, Any]:
    if not ids:
        return {"type": "array", "maxItems": 0, "items": {"type": "string"}}
    return {"type": "array", "maxItems": max_items, "items": {"type": "string", "enum": list(ids)}}


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _strs(max_items: int = 8) -> Dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": {"type": "string"}}


def analysis_schema() -> Dict[str, Any]:
    props: Dict[str, Any] = {"summary": {"type": "string"}}
    for f in FACETS:
        props[f] = _strs(8)
    return _obj(props)


def stereotype_schema(o_ids: List[str], s_ids: List[str], n_min: int, n_max: int) -> Dict[str, Any]:
    item = _obj({
        "name": {"type": "string"},
        "mapping": {"type": "string", "enum": ["ontology", "extension"]},
        "primary_ref": {"type": "string", "enum": (list(o_ids) + ["none"]) if o_ids else ["none"]},
        "ontology_refs": _ids(o_ids, 4),
        "sop_refs": _ids(s_ids, 3),
        "description": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "number"},
    })
    return _obj({"stereotypes": {"type": "array", "minItems": n_min, "maxItems": n_max, "items": item}})


def tags_schema(evidence_ids: List[str], n_min: int = 4, n_max: int = 10) -> Dict[str, Any]:
    tag = _obj({
        "name": {"type": "string"},
        "type": {"type": "string"},
        "multiplicity": {"type": "string", "enum": list(MULTIPLICITIES)},
        "unit": {"type": "string"},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
        "example": {"type": "string"},
        "value_source": {"type": "string", "enum": ["SOP", "ontology", "context", "default"]},
        "evidence": _ids(evidence_ids, 4),
        "description": {"type": "string"},
    })
    enum = _obj({
        "name": {"type": "string"},
        "literals": {"type": "array", "minItems": 2, "maxItems": 10, "items": {"type": "string"}},
        "evidence": _ids(evidence_ids, 3),
        "description": {"type": "string"},
    })
    return _obj({
        "tagged_values": {"type": "array", "minItems": n_min, "maxItems": n_max, "items": tag},
        "enumerations": {"type": "array", "maxItems": 4, "items": enum},
    })


def associations_schema(stereo_names: List[str], evidence_ids: List[str], n_min: int, n_max: int) -> Dict[str, Any]:
    item = _obj({
        "source": {"type": "string", "enum": list(stereo_names)},
        "name": {"type": "string"},
        "target": {"type": "string", "enum": list(stereo_names)},
        "multiplicity": {"type": "string", "enum": list(MULTIPLICITIES)},
        "evidence": _ids(evidence_ids, 2),
        "rationale": {"type": "string"},
    })
    return _obj({"associations": {"type": "array", "minItems": n_min, "maxItems": n_max, "items": item}})


def _new_tag(stereo_names: List[str]) -> Dict[str, Any]:
    return _obj({
        "stereotype": {"type": "string", "enum": list(stereo_names)},
        "name": {"type": "string"},
        "type": {"type": "string"},
        "multiplicity": {"type": "string", "enum": list(MULTIPLICITIES)},
        "unit": {"type": "string"},
    })


def ocl_schema(evidence_ids: List[str], stereo_names: List[str], n_min: int = 2, n_max: int = 5) -> Dict[str, Any]:
    inv = _obj({
        "name": {"type": "string"},
        "requirement": {"type": "string"},
        "evidence": _ids(evidence_ids, 3),
        "ocl": {"type": "string"},
        "reading": {"type": "string"},
        "new_tags": {"type": "array", "maxItems": 3, "items": _new_tag(stereo_names)},
    })
    return _obj({"invariants": {"type": "array", "minItems": n_min, "maxItems": n_max, "items": inv}})


assert set(METACLASSES) == set(Metaclass.__args__)  # keep Literal and tuple in sync
