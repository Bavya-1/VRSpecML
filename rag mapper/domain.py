"""Domain definition loader (ontologies, SOPs, retrieval focus) and per-project storage.

A project is a folder under config.PROJECTS_DIR holding the ontologies and guideline
documents the user uploaded, the index built from them and the saved session. Its
retrieval focus is taken from the domain template (config.DOMAIN_FILE).
"""
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import config
from metamodel import METACLASSES


@dataclass
class OntologySource:
    key: str
    file: str
    focus: str = ""
    dir: str = ""  # folder holding `file`; empty → the shared Ontologies/ folder

    @property
    def path(self) -> str:
        return os.path.join(self.dir or config.ONTOLOGY_DIR, self.file)


@dataclass
class MetaclassFocus:
    ontologies: List[str]
    facets: List[str]
    query: str
    guidance: str = ""


@dataclass
class Domain:
    id: str
    name: str
    package: str
    description: str
    ontologies: List[OntologySource]
    sop_files: List[str]
    metaclass_focus: Dict[str, MetaclassFocus]
    contrast_terms: List[str] = field(default_factory=list)
    project_dir: str = ""  # empty → the legacy shared Ontologies/ SOP/ index/ outputs/ folders
    shared_kb: bool = False  # no files uploaded → ground on the shared built-in knowledge base
    kb_id: str = ""  # id recorded in the index manifest (the domain id of whoever built the index)

    @property
    def sop_dir(self) -> str:
        own = self.project_dir and not self.shared_kb
        return os.path.join(self.project_dir, "sop") if own else config.SOP_DIR

    @property
    def index_dir(self) -> str:
        own = self.project_dir and not self.shared_kb
        return os.path.join(self.project_dir, "index") if own else config.INDEX_DIR

    @property
    def ontology_keys(self) -> List[str]:
        return [o.key for o in self.ontologies]

    def sop_paths(self) -> List[str]:
        """Guideline documents: the listed files, or every supported file in the SOP folder."""
        if not os.path.isdir(self.sop_dir):
            return []
        names = self.sop_files or sorted(
            f for f in os.listdir(self.sop_dir) if f.lower().endswith(config.SOP_EXTENSIONS)
        )
        return [os.path.join(self.sop_dir, n) for n in names if os.path.isfile(os.path.join(self.sop_dir, n))]


def load_domain(path: str = config.DOMAIN_FILE) -> Domain:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    focus = {m: MetaclassFocus(**raw["metaclass_focus"][m]) for m in METACLASSES}
    return Domain(
        id=raw["id"],
        name=raw["name"],
        package=raw["package"],
        description=raw["description"],
        ontologies=[OntologySource(**o) for o in raw["ontologies"]],
        sop_files=raw.get("sop_files", []),
        metaclass_focus=focus,
        contrast_terms=[t.lower() for t in raw.get("contrast_terms", [])],
    )


# ── projects ─────────────────────────────────────────────────────────────────
def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name)) or "file"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


def _project_file(slug: str) -> str:
    return os.path.join(config.PROJECTS_DIR, slug, "project.json")


def load_project(slug: str) -> Domain:
    """The domain template specialised with one project's name, ontologies and guidelines."""
    with open(_project_file(slug), encoding="utf-8") as fh:
        meta = json.load(fh)
    template = load_domain()
    project_dir = os.path.dirname(_project_file(slug))
    camel = "".join(w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", meta["name"]) if w) or "Project"
    if not meta["ontologies"] and not meta["sop_files"]:
        # nothing uploaded: the template's own ontologies, guidelines and prebuilt index are the knowledge base
        return replace(template, id=meta["slug"], name=meta["name"], package=f"VRSpecML::{camel}",
                       description=f"VR training project: {meta['name']}", project_dir=project_dir,
                       shared_kb=True, kb_id=template.id)
    ontologies = [OntologySource(o["key"], o["file"], o.get("focus", ""), os.path.join(project_dir, "ontologies"))
                  for o in meta["ontologies"]]
    keys = [o.key for o in ontologies]
    focus = {m: replace(f, ontologies=list(keys)) for m, f in template.metaclass_focus.items()}
    return Domain(
        id=meta["slug"], name=meta["name"], package=f"VRSpecML::{camel}",
        description=f"VR training project: {meta['name']}",
        ontologies=ontologies, sop_files=list(meta["sop_files"]), metaclass_focus=focus,
        contrast_terms=template.contrast_terms, project_dir=project_dir,
    )


def create_project(name: str, ontology_files: List[Tuple[str, bytes]], sop_files: List[Tuple[str, bytes]]) -> Domain:
    """Store the uploaded files in a new project folder and return its domain."""
    base = slugify(name)
    slug, n = base, 2
    while os.path.exists(os.path.join(config.PROJECTS_DIR, slug)):
        slug, n = f"{base}-{n}", n + 1
    project_dir = os.path.join(config.PROJECTS_DIR, slug)
    onto_dir, sop_dir = os.path.join(project_dir, "ontologies"), os.path.join(project_dir, "sop")
    os.makedirs(onto_dir)
    os.makedirs(sop_dir)
    try:
        ontologies, used = [], set()
        for fname, data in ontology_files:
            safe = safe_filename(fname)
            key = re.sub(r"[^A-Za-z0-9]+", "", os.path.splitext(safe)[0]).upper() or "ONTOLOGY"
            base_key, i = key, 2
            while key in used:
                key, i = f"{base_key}{i}", i + 1
            used.add(key)
            with open(os.path.join(onto_dir, safe), "wb") as fh:
                fh.write(data)
            ontologies.append({"key": key, "file": safe, "focus": ""})
        sop_names = []
        for fname, data in sop_files:
            safe = safe_filename(fname)
            with open(os.path.join(sop_dir, safe), "wb") as fh:
                fh.write(data)
            sop_names.append(safe)
        meta = {"slug": slug, "name": name.strip(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ontologies": ontologies, "sop_files": sop_names}
        with open(_project_file(slug), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise
    return load_project(slug)


def delete_project(slug: str) -> None:
    shutil.rmtree(os.path.join(config.PROJECTS_DIR, slug), ignore_errors=True)
