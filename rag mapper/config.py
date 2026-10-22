"""Central configuration for the VRSpecML RAG profile generator."""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ONTOLOGY_DIR = os.path.join(ROOT_DIR, "Ontologies")
SOP_DIR = os.path.join(ROOT_DIR, "SOP")
DOMAIN_DIR = os.path.join(BASE_DIR, "domains")
DOMAIN_FILE = os.path.join(DOMAIN_DIR, "sustainable_smart_farm.json")
INDEX_DIR = os.path.join(BASE_DIR, "index")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RUN_DIR = os.path.join(BASE_DIR, "run")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SESSION_FILE = os.path.join(OUTPUT_DIR, "session_latest.json")

PROJECTS_DIR = os.path.join(BASE_DIR, "projects")  # one folder per project: ontologies/, sop/, index/, outputs/

SOP_EXTENSIONS = (".pdf", ".txt", ".md")
ONTOLOGY_EXTENSIONS = (".owl", ".rdf", ".xrdf", ".ttl")  # .ttl is Turtle, the rest RDF/XML

# ─── Embeddings (same SBERT encoder as ../mapper) ────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 256
KB_VERSION = 5  # bump when ingestion/chunking logic changes to force a rebuild

# ─── Retrieval ───────────────────────────────────────────────────────────────
RRF_K = 60                 # Reciprocal Rank Fusion constant
CANDIDATE_POOL = 400       # dense depth before filtering and fusion
MMR_LAMBDA = 0.72          # relevance vs. diversity trade-off
CHUNK_WORDS = 170          # target SOP chunk size
CHUNK_OVERLAP_SENTENCES = 1
EXCLUDED_CLASS_PREFIXES = {"BFO", "COB", "RO"}  # upper-level scaffolding, not stereotype material

# ─── Qwen 2.5 7B (local) ─────────────────────────────────────────────────────
# llama.cpp's llama-server exposes an OpenAI-compatible API under /v1; a host
# without /v1 is treated as a native Ollama server (same scheme as ../llm mapper).
DEFAULT_LLM_HOST = os.environ.get("VRSPEC_LLM_HOST", "http://127.0.0.1:8080/v1")
DEFAULT_LLM_MODEL = os.environ.get("VRSPEC_LLM_MODEL", "qwen2.5-7b-instruct")
OLLAMA_QWEN_MODEL = "qwen2.5:7b"
QWEN_GGUF = os.environ.get(
    "VRSPEC_QWEN_GGUF",
    os.path.expanduser("~/llm-models/qwen2.5-7b-instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
)
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_CTX = 32768   # split across slots
LLAMA_SERVER_PARALLEL = 2  # concurrent generation slots
LLM_PARALLEL = 2           # client-side worker threads (match the server slots)
LLM_TEMPERATURE = 0.15
LLM_SEED = 42
LLM_TIMEOUT = 600

# ─── Generation defaults ─────────────────────────────────────────────────────
STEREOTYPES_PER_METACLASS = 5
CLASS_CANDIDATES_PER_PROMPT = 28
SOP_CHUNKS_PER_PROMPT = 6
RULES_PER_PROMPT = 8
OCL_REPAIR_ATTEMPTS = 3

# ─── Ontology alignment of stereotypes (cosine similarity, SBERT) ────────────
ALIGN_ACCEPT = 0.58    # a better class must reach this similarity to replace the LLM's choice
ALIGN_MARGIN = 0.08    # … and beat the LLM's choice by this margin
ALIGN_REJECT = 0.40    # below this the LLM's class is a weak match (→ extension unless a better one exists)
ALIGN_PROMOTE = 0.62   # an extension is mapped to a class this similar
ALIGN_ROLE_PENALTY = 0.12  # class kind (BFO category) incompatible with the metaclass
ALIGN_CONTRAST_PENALTY = 0.10  # class names a contrasting material (soil vs water) absent from the stereotype
SEED_MIN_SIM = 0.50    # related (non-primary) classes contribute tag seeds above this similarity
