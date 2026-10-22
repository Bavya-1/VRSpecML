"""Role-based VR metamodel (Scene, Article, ActionResponse, Behavior, Timeline) and naming helpers."""
import re
from typing import Dict, List

METACLASSES = ("Scene", "Article", "ActionResponse", "Behavior", "Timeline")

METACLASS_INFO: Dict[str, Dict[str, str]] = {
    "Scene": {"role": "the spatial container of the VR system"},
    "Article": {"role": "an in-scene object"},
    "ActionResponse": {"role": "a user interaction and its observable system response"},
    "Behavior": {"role": "a rule governing state"},
    "Timeline": {"role": "an ordered sequence of steps"},
}

PRIMITIVE_TYPES = ("String", "Integer", "Float", "Boolean")
MULTIPLICITIES = ("1", "0..1", "*", "1..*")
SEVERITIES = ("Error", "Warning")

XSD_TO_UML = {
    "string": "String", "normalizedString": "String", "anyURI": "String",
    "integer": "Integer", "int": "Integer", "long": "Integer", "short": "Integer",
    "nonNegativeInteger": "Integer", "positiveInteger": "Integer",
    "double": "Float", "float": "Float", "decimal": "Float",
    "boolean": "Boolean", "dateTime": "String", "date": "String",
}

_WORD = re.compile(r"[A-Za-z0-9]+")
_OCL_RESERVED = {
    "and", "or", "xor", "not", "implies", "if", "then", "else", "endif", "let", "in", "self",
    "true", "false", "null", "invalid", "context", "inv", "pre", "post", "def", "package", "endpackage",
}


def _words(text: str) -> List[str]:
    # Split camelCase / PascalCase boundaries before tokenising.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return _WORD.findall(text)


def to_upper_camel(text: str) -> str:
    words = _words(text.replace("«", "").replace("»", ""))
    name = "".join(w[:1].upper() + w[1:] for w in words)
    if not name:
        return "Unnamed"
    if name[0].isdigit():
        name = "S" + name
    return name


def to_lower_camel(text: str) -> str:
    upper = to_upper_camel(text)
    if upper == "Unnamed":
        return "value"
    # Keep leading acronyms readable: "PHValue" → "phValue", "EC" → "ec"
    match = re.match(r"^([A-Z]+)(?=[A-Z][a-z]|$)", upper)
    if match:
        head = match.group(1)
        name = head.lower() + upper[len(head):]
    else:
        name = upper[0].lower() + upper[1:]
    if name in _OCL_RESERVED:
        name += "Value"
    return name


def to_enum_literal(text: str) -> str:
    literal = to_upper_camel(text)
    return literal if literal != "Unnamed" else "Unknown"


def split_camel(name: str) -> str:
    return " ".join(_words(name)).lower()


def normalize_multiplicity(value: str) -> str:
    v = (value or "1").strip().replace(" ", "").replace("[", "").replace("]", "")
    aliases = {"0..*": "*", "n": "*", "many": "*", "1..n": "1..*", "0..n": "*", "1..1": "1", "optional": "0..1"}
    v = aliases.get(v, v)
    return v if v in MULTIPLICITIES else "1"


def is_many(multiplicity: str) -> bool:
    return normalize_multiplicity(multiplicity) in ("*", "1..*")
