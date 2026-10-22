# rag mapper — Developer Architecture Report

A RAG pipeline that turns the AGRO/ENVO/CHMO ontologies, FAO SOP text, and a free-text VR
training scenario into a VRSpecML UML profile (stereotypes, tagged values, OCL invariants)
for a Sustainable Smart Farm VR training app. Generation runs entirely locally against a
Qwen2.5-7B model served by `llama-server`. ~5,100 lines across 17 Python modules.

## 1. At a glance

```
┌─────────────────────┐   ┌──────────────────┐
│ Ontologies/*.owl     │   │ SOP/*.pdf         │
│ (AGRO, ENVO, CHMO)   │   │ (FAO guidelines)  │
└──────────┬───────────┘   └─────────┬─────────┘
           │ ontology_ingest.py      │ sop_ingest.py
           ▼                         ▼
   classes, properties      chunks, normative rules
           │                         │
           └───────────┬─────────────┘
                        ▼
              knowledge_base.py (build_index.py)
      4 corpora: class / property / chunk / rule
      SBERT embeddings (embedder.py) + BM25 (bm25.py)
      → index/kb.pkl + index/emb_*.npy + manifest.json
                        │
                        ▼  hybrid search (RRF fusion + MMR diversify)
┌───────────────────────────────────────────────────────────────┐
│                      pipeline.py (Pipeline)                    │
│                                                                 │
│  Task 1  Stereotypes   grounding.py builds an EvidencePack     │
│  (ontology+SOP → UML   per metaclass → prompts.py renders a    │
│   stereotypes per      prompt → llm_client.py calls Qwen with  │
│   metaclass, R6/R7     a JSON Schema (schemas.py) → response   │
│   tag seeds)           parsed back into Stereotype objects,    │
│                        then re-aligned against ontology classes │
│                        by cosine similarity (_align)            │
│                                                                 │
│  Task 2  Tagged values  same evidence→prompt→LLM→parse loop,   │
│  (+ enumerations,       one call per stereotype in parallel,   │
│   R7 associations)      then a single global associations call │
│                                                                 │
│  Task 3  OCL invariants  evidence→prompt→LLM→parse, each       │
│  (+ derived R9/R10       invariant is type-checked by ocl.py;  │
│   invariants)            invalid ones go through up to 3        │
│                          repair round-trips with the LLM        │
└───────────────────────────────────────────────────────────────┘
                        │
                        ▼
                  schemas.py: Profile
       (stereotypes, tagged_values, enumerations, invariants)
                        │
                        ▼
              exporters.py → JSON / .ocl / Markdown / Graphviz
                        │
                        ▼
                  app.py (Streamlit UI, 3 tabs + Export)
```

Everything downstream of the knowledge base is driven by `app.py`, which is a thin
Streamlit shell around `Pipeline` — the UI has almost no logic of its own beyond
rendering `Profile` objects and wiring buttons to `run_task1/2/3`.

## 2. Module map

| File | Lines | Responsibility |
|---|---|---|
| `config.py` | 64 | All paths, embedding/retrieval/LLM constants in one place. Nothing else imports hardcoded paths. |
| `domain.py` | 93 | Loads `domains/sustainable_smart_farm.json` into dataclasses (`Domain`, `FlowStep`, `MetaclassFocus`). The domain config is data, not code — a new domain is a new JSON file. |
| `metamodel.py` | 153 | Static knowledge about the VRSpecML metamodel: the 5 metaclasses, the O2P rules R1–R12, BFO upper-category tables, and name-casing helpers (`to_upper_camel`, `to_lower_camel`, etc.) used everywhere. |
| `ontology_ingest.py` | 395 | RDFLib-based OWL parser. Extracts classes/properties/restrictions from AGRO/ENVO/CHMO, merges terms shared across files by URI, and classifies each class into a coarse BFO category (process/quality/role/site/material/information) by walking superclasses. |
| `sop_ingest.py` | 228 | PDF/text ingestion for the FAO SOP: extracts pages (prefers `pdftotext -layout`, falls back to `pypdf`), reconstructs section-aware paragraph/bullet blocks, chunks them (~170 words), and separately extracts "normative rules" — sentences matching modality cues (prohibition/obligation/recommendation regexes). |
| `embedder.py` | 50 | Thin wrapper around a cached `SentenceTransformer` (all-MiniLM-L6-v2), CPU-only, L2-normalized output. |
| `bm25.py` | 56 | A from-scratch Okapi BM25 inverted index (no external dependency) with a tiny stemmer and stopword list. |
| `knowledge_base.py` | 289 | The retrieval core. Builds/loads 4 FAISS-indexed corpora (class/property/chunk/rule), exposes `search`/`search_many` which fuse one or more weighted (query, filters) pairs via Reciprocal Rank Fusion, then diversify the top results with MMR. Also tracks a manifest so the UI can detect a stale index. |
| `grounding.py` | 270 | `EvidencePack`: numbers every retrieved item with a short id (`O1`, `S2`, `R3`...) that the LLM must cite — ids are enumerated directly into the JSON Schema so the model is structurally prevented from inventing a citation. Also implements the deterministic O2P derivations: turning an OWL restriction into a `TagSeed` (R6/R7/R8), and finding "value partitions" (sibling subclasses) to seed enumerations. |
| `schemas.py` | 259 | Two things bundled together: (1) the Pydantic data model for a `Profile` (Stereotype, TaggedValue, Enumeration, Invariant, EvidenceRef...), and (2) JSON Schema builders for each LLM call, parameterized by the evidence ids and stereotype names actually available in that call so the grammar-constrained decoder can't hallucinate references. |
| `llm_client.py` | 196 | `QwenClient`: talks to either an OpenAI-compatible `/v1` endpoint (llama.cpp's `llama-server`, the default) or a native Ollama `/api/chat` endpoint, auto-detected from the host URL. Every call passes `response_format: json_schema` (or Ollama's `format`) for grammar-constrained decoding. Includes a truncated-JSON repair routine that walks back to the last complete `}` and closes open brackets, so a response cut off by `max_tokens` still yields the complete items. |
| `prompts.py` | 192 | Pure string-building functions for the system prompt and the 5 task prompts (analysis, stereotype, tags, associations, OCL, repair). No logic beyond formatting — all the actual grounding/evidence selection happens in `pipeline.py`/`grounding.py` before these are called. |
| `pipeline.py` | 939 | The orchestrator. See §4. |
| `ocl.py` | 782 | A hand-rolled OCL 2.4 tokenizer, recursive-descent parser, and type checker. See §6. |
| `exporters.py` | 244 | Renders a `Profile` to JSON (with full provenance), `.ocl` (plain OCL source), Markdown (a readable catalog with tables), and Graphviz DOT (a class-diagram-like picture of stereotypes/tags/enums). Also computes `traceability()` (requirement → profile elements → invariant) and `residue_log()` (O2P step 7: things with no ontology grounding, unverified bounds, invalid invariants). |
| `server.py` | 115 | Starts/stops/probes the local `llama-server` process (PID file in `run/`, GPU offload `-ngl 99`). Used by both the CLI and the Streamlit sidebar's "Start Qwen" button. |
| `build_index.py` | 40 | CLI entry point: `python build_index.py [--force]`, wraps `knowledge_base.build_knowledge_base`. |
| `app.py` | 569 | Streamlit UI: sidebar (Qwen status/settings, KB status/rebuild, session save/restore), a context textarea, and 4 tabs (①Stereotypes ②Tagged values ③OCL ④Export) that just render `Profile` state and call into `Pipeline`. |
| `domains/sustainable_smart_farm.json` | 202 | The one configured domain: which ontology files to use, SOP files, the VR training flow steps, per-metaclass retrieval focus/guidance, sample scenario texts, and "contrast terms" (e.g. soil vs. water) used to penalize cross-domain ontology mismatches. |

## 3. Retrieval & grounding

The knowledge base (`knowledge_base.py`) holds four separate corpora — **class**, **property**
(from the OWL ontologies), **chunk**, **rule** (from the SOP PDF) — each with its own FAISS flat
index (cosine via inner product on normalized vectors) and its own BM25 index.

A single `search_many(corpus, queries, k)` call takes a *list* of `(text, weight, filters)` tuples,
ranks each one independently with a hybrid dense+lexical scheme (`_rank`), fuses all the rankings
with **Reciprocal Rank Fusion** (`1/(RRF_K + rank + 1)`, weighted), then diversifies the top results
with **MMR** (`MMR_LAMBDA = 0.72` relevance/diversity trade-off) so the evidence handed to the LLM
isn't N near-duplicates of the same passage. `filters` can restrict by source ontology file,
ontology prefix (e.g. exclude `UO` unit-ontology noise from class search), or SOP document.

This is how every pipeline step gathers its evidence: e.g. `_task1_evidence` builds ~15-20 weighted
queries (the metaclass's configured focus query, per-ontology-file variants, per-flow-step
variants, the context analysis summary, and individual extracted facets) and merges them into one
ranked class list and one ranked chunk list.

`grounding.py`'s `EvidencePack` is the glue between retrieval and prompting: every retrieved item
gets a short id like `O3` (ontology class #3) or `S2` (SOP chunk #2) via `pack.add_class(doc)` etc.
Those ids are then (a) formatted into the prompt text the LLM reads, and (b) passed to the relevant
`schemas.py` builder (e.g. `stereotype_schema(o_ids, s_ids, ...)`) which enumerates them as a JSON
Schema `enum` — so the grammar-constrained decoder (llama.cpp/Ollama JSON-schema mode) is
*structurally* prevented from citing an id that wasn't actually retrieved. `pack.resolve(ids)` turns
the ids the model chose back into full `EvidenceRef` objects for storage/display.

`EvidencePack` also carries the **deterministic** (non-LLM) O2P derivations:
- `tag_seeds_for`: walks a class's OWL restrictions (and its parents' restrictions, as fallback) and
  turns each into a `TagSeed` candidate tagged value — R6 (quality/measurement), R7 (association, if
  the restriction's filler class maps to another stereotype), or R8 (enumeration, if the filler class
  has ≥2 named subclasses).
- `partitions_for`: finds classes with 2-40 direct subclasses as enumeration-literal candidates,
  ranked by cosine similarity to a context vector when one is given.

## 4. The three-task pipeline

`Pipeline` (pipeline.py) runs the three VRSpecML generation tasks. All three share a pattern: build
an `EvidencePack` → render a prompt (`prompts.py`) → call `client.chat_json(system, prompt, schema)`
→ normalize/validate the raw JSON into typed objects → merge into `Profile`.

**Task 1 — stereotypes** (`run_task1`): for each of the 5 metaclasses (Scene, Article,
ActionResponse, Behavior, Timeline), in parallel (`ThreadPoolExecutor`, size = `LLM_PARALLEL`), ask
Qwen for `n` to `n+2` candidate stereotypes. After collecting all candidates:
1. `_align`: for every stereotype, re-embed its name+description and compare (cosine, with role/
   contrast penalties) against both its cited ontology class and the best-retrieved alternative. This
   can **re-align** a weak choice to a better class, **demote** a weak ontology mapping to an
   "extension", or **promote** an unmapped extension to an ontology mapping if the similarity is high
   enough (thresholds in `config.py`: `ALIGN_ACCEPT`/`ALIGN_MARGIN`/`ALIGN_REJECT`/`ALIGN_PROMOTE`).
2. De-duplicate by name and by (metaclass, primary ontology class).
3. Force exactly one "core" stereotype per metaclass (the rest become "auxiliary").
4. Derive R6/R7/R8 tag seeds and a grounding score (max cosine similarity to cited evidence) per
   stereotype — these seeds aren't tagged values yet, just candidates surfaced to Task 2.

**Task 2 — tagged values** (`run_task2`): one LLM call per *kept* stereotype (parallel), each seeded
with its ontology refs, tag seeds, relevant properties/units, value partitions (for enumerations),
and user-constraint text routed to its 3 most-relevant stereotypes (`_route_constraints`, by cosine
similarity). The raw LLM output is heavily normalized (`_normalize_tag`): type aliasing/casing,
multiplicity coercion, bounds sanity (swap if min>max, check numeric examples fall in range),
enumeration literal matching, and a `verified` flag — `True` only if any numeric bound is literally
found in the cited evidence texts, or is "physically certain" (pH 0-14, percentages 0-100, a
non-negativity bound). After all stereotypes are processed, one more LLM call (`_associations`)
proposes R7 association-typed tags linking stereotypes together (Scene→Articles, ActionResponse→
Behavior, etc.).

**Task 3 — OCL invariants** (`run_task3`): per kept stereotype, first computes **deterministic**
invariants (`_derived_invariants`) from already-verified tag bounds (R9: `self.x >= a and self.x <=
b`) and mandatory associations (R10: `self.assoc->notEmpty()`) — these need no LLM call at all. Then
an LLM call proposes additional invariants grounded in SOP rules/ontology axioms. Every invariant,
LLM-written or derived, is passed through `ocl.validate_invariant` (§6); if invalid, it goes through
up to `OCL_REPAIR_ATTEMPTS` (3) repair round-trips where the type checker's exact error list is fed
back to the LLM (`repair_prompt`) asking for a fix.

A subtlety worth knowing: an invariant can ask for a tagged value that doesn't exist yet
(`new_tags` in the schema) — the pipeline accepts it only if the repaired OCL actually ends up
navigating it (`res.navigated`), then injects it as a real `TaggedValue` on the target stereotype.

## 5. Profile data model

Everything the pipeline produces lives in one Pydantic object tree (`schemas.py`):

- **`Profile`** — top-level container: `stereotypes: List[Stereotype]`, `tagged_values: Dict[sid,
  List[TaggedValue]]`, `enumerations: List[Enumeration]`, `invariants: List[Invariant]`, plus
  `analysis` (the extracted context facets) and a `task_log` for UI diagnostics.
- **`Stereotype`** — `sid` (`ST01`...), `name`, `extends` (one of the 5 metaclasses), `role`
  (core/auxiliary), `mapping` (ontology/extension), `primary`/`ontology_refs`/`sop_refs`
  (`EvidenceRef`s), `tag_seeds` (candidate R6/R7/R8 tags from OWL), `grounding` (float score), `keep`
  (soft-delete flag the UI's data editor toggles), `notes` (human-readable audit trail, e.g.
  "ontology class re-aligned: ...").
- **`TaggedValue`** — `name`/`type`/`multiplicity`/`unit`/`min`/`max`/`example`, `value_source`
  (SOP/ontology/context/default), `rule` (R6/R7/R8/extension), `verified` (bool), `evidence`.
- **`Invariant`** — `iid` (`INV-01`...), `context`/`context_sid`, `body` (the OCL expression),
  `severity` (Error/Warning), `pattern` (R9/R10/R11/R12/SOP), `origin` (llm/derived), `status`
  (valid/invalid), `errors`/`warnings` from the type checker, `repairs` (count), `grounding`,
  `unsupported_constants` (numeric literals in the body not found in any cited evidence).
- **`EvidenceRef`** — the uniform citation object: `kind` (class/property/chunk/rule/context/axiom/
  seed/unit/partition), `id`, `label`, `uri`, `source`, `text` (a short quote/snippet).

`pipeline.build_env(profile)` projects a `Profile` into the much smaller `ocl.TypeEnv` (stereotype →
tag name → type/multiplicity, plus enum → literals) that the OCL checker actually needs.
`pipeline.revalidate(profile)` re-runs every invariant through the checker — called after any edit
(rename, tag change, invariant edit) to keep `status`/`errors` consistent with the current profile.

## 6. The OCL checker (`ocl.py`)

This is the most self-contained and arguably most interesting module: a complete hand-written OCL
2.4 subset — tokenizer, recursive-descent parser, and a structural type checker — with no external
OCL library dependency.

- **Tokenizer/parser**: standard precedence-climbing expression grammar (implies → or/xor → and →
  equality → relational → additive → multiplicative → unary → postfix → primary), supporting
  navigation (`.`), implicit collect / arrow operations (`->forAll`, `->exists`, `->select`, ...),
  `let`, `if-then-else-endif`, `Enum::Literal`, collection literals (`Sequence{1..5}`,
  `Set{...}`), and `oclIsKindOf`/`oclAsType`/etc.
- **`normalize()`**: cleans up what an LLM actually tends to write — strips a `context X inv Y:`
  header if the model included one anyway, converts smart quotes/arrows/≤≥≠, rewrites `==`/`!=`/`&&`/
  `||` to proper OCL, converts double-quoted strings to single-quoted.
- **`Checker`**: walks the AST against a `TypeEnv` (built fresh from the live `Profile` on every
  check) and assigns a structural type `T` to every subexpression — primitives, enum, stereotype,
  metaclass, or collection-of. It checks: every navigated tag name actually exists on that
  stereotype (and suggests which stereotype *does* declare it, if the name exists elsewhere); every
  enum literal belongs to the enum it's qualified with; boolean operators get boolean operands;
  `oclIsKindOf` only takes stereotype names; collection operations are well-typed recursively through
  iterators. It also **records** which (stereotype, tag) pairs were actually navigated
  (`checker.navigated`) — this is what lets the pipeline decide whether a proposed `new_tag` was
  really used, and what powers `exporters.traceability()`.
- Errors are deduplicated and phrased to be directly re-fed to the LLM as a repair prompt (see
  `prompts._FIX_HINTS`, which maps substrings of checker error messages to a one-line fix hint).

## 7. Things to know before changing code

- **Config is the single source of truth for thresholds.** Every "magic number" that affects
  generation quality (alignment thresholds, RRF/MMR constants, chunk size, repair attempts,
  temperature/seed) lives in `config.py`, not scattered through `pipeline.py`. Change behavior there
  first.
- **The knowledge base is a pickle + raw `.npy` embeddings, versioned by `config.KB_VERSION`.**
  Bump `KB_VERSION` in `config.py` whenever you change the shape of a doc record or chunking logic —
  otherwise a stale `index/kb.pkl` will silently be reused (`kb_status` only rebuilds on a genuine
  mismatch: KB_VERSION, embed model, domain id, or file size/mtime fingerprint change).
- **Every LLM call is schema-constrained AND still re-validated in Python.** Don't assume a
  schema-conformant response is semantically correct — `pipeline.py`'s normalization functions
  (`_normalize_tag`, `_normalize_type`, `_stereotype_from_raw`, etc.) are doing real work (type
  coercion, bounds sanity, duplicate detection) that the schema can't express. If you add a new LLM
  call, budget for a normalization pass, not just a schema.
- **Evidence ids are the *only* thing preventing hallucinated citations** — they're enumerated per
  call from whatever was actually retrieved for *that* stereotype/metaclass. If you add a new
  evidence type, it needs a `PREFIX_KIND` entry in `grounding.py`, an `add_*` method, and to be
  threaded through the relevant `schemas.py` builder's id list.
- **`Pipeline._run_parallel`** swallows per-item exceptions (logs + collects into an `errors` dict)
  so one stereotype's LLM failure doesn't sink the whole task — but `run_task1` does raise if *every*
  job failed. Keep that asymmetry in mind if you're debugging a "silently missing stereotype" vs. a
  hard crash.
- **`revalidate()` must be called after any profile mutation** that could affect OCL validity
  (renaming a stereotype, deleting a tag, editing an invariant body by hand in the UI) — it's not
  automatic. `app.py` calls it explicitly after every such edit path.
- **The Streamlit UI holds almost no state logic** — `st.session_state.profile` is the single
  source of truth, persisted to `outputs/session_latest.json` after every successful run
  (`save_session`). If something looks wrong in the UI, the bug is almost always in `pipeline.py` or
  `schemas.py`, not `app.py`.
- **Domain is pluggable, metamodel is not.** `domains/sustainable_smart_farm.json` is the only
  domain-specific config file (ontology sources, SOP files, VR flow, per-metaclass retrieval
  focus/guidance, sample contexts, contrast terms). The five metaclasses, the O2P rules, and the
  casing/naming helpers in `metamodel.py` are hardcoded to the VRSpecML paper's metamodel — adding a
  6th metaclass would touch `metamodel.py`, `schemas.py`'s `Metaclass` Literal, and every prompt in
  `prompts.py`.
