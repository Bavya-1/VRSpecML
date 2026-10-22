# VRSpecML RAG · Sustainable Smart Farm

A RAG pipeline that turns AGRO/ENVO/CHMO ontologies + FAO guideline text + a free-text
VR training scenario into a VRSpecML UML profile:

1. **Stereotypes** — ontology classes → stereotypes on the five VR metaclasses (Scene,
   Article, ActionResponse, Behavior, Timeline), grounded in AGRO/ENVO/CHMO + SOPs.
2. **Tagged values** — guidelines + ontology → typed fields (units, bounds, examples,
   associations, enumerations) for each stereotype.
3. **OCL invariants** — SOP rules + ontology axioms → OCL 2.4 constraints, type-checked
   against the profile and auto-repaired when invalid.

Retrieval is hybrid (SBERT + BM25, RRF-fused, MMR-diversified) over four corpora built
from the ontologies and SOP PDFs. Generation uses **Qwen2.5-7B-Instruct** running fully
locally via `llama-server` (llama.cpp, GPU) — no API key, no cloud.

## Setup (one-time)

```bash
conda activate vrspecml     # env with streamlit, rdflib, sentence-transformers, faiss, pypdf, llama-cpp-python
python build_index.py       # parses Ontologies/{agro,envo,chmo-full}.owl + SOP/*.pdf → index/
```

Rebuild whenever ontology or SOP files change (the sidebar also flags a stale index).

## Run

```bash
streamlit run app.py
```

Start Qwen from the sidebar ("Start Qwen" button — launches `llama-server` on the GPU
with the GGUF at `~/llm-models/qwen2.5-7b-instruct/`), or manually:

```bash
python server.py start --wait
```

Then: paste/describe a VR scenario (or pick a sample) → **① Generate** stereotypes →
**② Generate** tagged values → **③ Generate** OCL invariants → **Export** (JSON / OCL /
Markdown / Graphviz diagram).

## Layout

- `domains/sustainable_smart_farm.json` — the one configured domain (ontologies, VR
  flow, per-metaclass retrieval focus, sample scenarios).
- `ontology_ingest.py`, `sop_ingest.py` — parse OWL classes/properties/restrictions and
  chunk/extract normative rules from the SOP PDFs.
- `knowledge_base.py`, `bm25.py`, `embedder.py` — the hybrid retrieval index.
- `grounding.py` — evidence packing + deterministic O2P derivations (tag seeds from OWL
  restrictions, enumerations from value partitions).
- `pipeline.py` — the three tasks, ontology re-alignment, association linking, and the
  OCL repair loop.
- `ocl.py` — an OCL 2.4 parser/type-checker validating every generated invariant against
  the actual profile (no unknown tags, enum literals, or type errors).
- `schemas.py` — JSON Schemas that constrain Qwen's output (evidence ids are enumerated,
  so it can't cite evidence that wasn't retrieved).
- `exporters.py` — JSON catalog, `.ocl` file, Markdown catalog, Graphviz profile diagram.
