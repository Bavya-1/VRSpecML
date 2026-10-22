"""SOP / guideline ingestion: text extraction, section-aware chunking and
normative-rule extraction (the evidence units used for OCL constraints).
"""
import logging
import os
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Tuple

import config

logging.getLogger("pypdf").setLevel(logging.ERROR)

_BULLET_CHARS = "•●▪◦■•●"
_BULLET_RE = re.compile(rf"^\s*(?:[{_BULLET_CHARS}]|[-–*]\s|\(?[a-z]\)\s|\d+\)\s)\s*")
_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][A-Za-z0-9 ,;:()/'’&\-–]{1,90}[A-Za-z)])\s*$")
_NAMED_HEADINGS = {"foreword", "contents", "introduction", "annex", "annexes", "references", "glossary", "summary",
                   "scope", "disclaimer and copyright", "disclaimer"}
_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(“\"])")

MODALITY_CUES: List[Tuple[str, re.Pattern]] = [
    ("prohibition", re.compile(r"\b(should|must|shall)\s+(not|never)\b|\bbe\s+avoided\b|\bavoid(ed|ing)?\b|\bprohibit|\bnot\s+exceed|\bmust\s+not\b|\bdo\s+not\b|\bnever\b", re.I)),
    ("obligation", re.compile(r"\b(shall|must|required|requires?|is\s+imperative|need(s)?\s+to|has\s+to|have\s+to|essential)\b", re.I)),
    ("recommendation", re.compile(r"\b(should|recommend(ed)?|encourag(e|ed)|could|where\s+appropriate|preferably|best\s+managed|optimi[sz]ed?)\b", re.I)),
    ("recommendation", re.compile(r"^(minimi[sz]e|maximi[sz]e|increase|decrease|reduce|protect|prevent|maintain|improve|enhance|ensure|promote|apply|use|make|implement(ing)?|monitor|limit|adopt|avoid|foster|establish|restore)\b", re.I)),
]


_PAGE_NUMBER = re.compile(r"^\s*(\d{1,3}|[ivxlc]{1,6})\s{2,}\S|\S\s{2,}(\d{1,3}|[ivxlc]{1,6})\s*$", re.I)
_SKIP_SECTIONS = {"contents", "table of contents", "disclaimer and copyright", "disclaimer"}


def _clean_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace("­", "").replace("\t", " ")
    for ch in _BULLET_CHARS:
        line = line.replace(ch, "•")
    return re.sub(r"[  ]{2,}", " ", line).strip()


def _norm_key(line: str) -> str:
    return re.sub(r"[\divxlc\s]+$|^[\divxlc\s]+|\s+", "", line.lower())


def extract_pages(path: str) -> List[str]:
    """Raw text per page. PDFs use poppler's pdftotext (layout mode) when installed,
    otherwise pypdf; text/markdown files are split on form feeds."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if shutil.which("pdftotext"):
            try:
                out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", path, "-"],
                                     capture_output=True, text=True, timeout=180)
                if out.returncode == 0 and out.stdout.strip():
                    pages = out.stdout.split("\f")
                    return pages[:-1] if pages and not pages[-1].strip() else pages
            except (OSError, subprocess.SubprocessError):
                pass
        from pypdf import PdfReader

        return [page.extract_text() or "" for page in PdfReader(path).pages]
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().split("\f")


def _strip_running_lines(pages: List[List[str]]) -> List[List[str]]:
    """Drop running headers/footers (repeated lines carrying a page number), bare page
    numbers and photo credits. Raw lines keep their indentation."""
    counts = Counter()
    for lines in pages:
        counts.update({_norm_key(l) for l in lines if _PAGE_NUMBER.search(l)})
    out = []
    for lines in pages:
        kept = []
        for raw in lines:
            text = raw.strip()
            if not text:
                kept.append("")
            elif re.fullmatch(r"[ivxlcdm\d]+", text.lower()) or text.startswith("©"):
                continue
            elif _PAGE_NUMBER.search(raw) and counts[_norm_key(raw)] >= 2:
                continue
            else:
                kept.append(raw.rstrip())
        out.append(kept)
    return out


def _blocks(pages: List[List[str]]) -> List[Dict[str, Any]]:
    """Merge lines into paragraph / bullet blocks carrying section and page."""
    blocks: List[Dict[str, Any]] = []
    section_id, section_title = "", ""
    current: Dict[str, Any] = {}
    after_blank = False

    def flush():
        nonlocal current
        if current and current["text"].strip() and section_title.lower() not in _SKIP_SECTIONS:
            current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
            current.pop("indent", None)
            blocks.append(current)
        current = {}

    for page_no, lines in enumerate(pages, start=1):
        for raw in lines:
            indent = len(raw) - len(raw.lstrip())
            line = _clean_line(raw)
            if not line:
                if current and not current.get("bullet"):
                    flush()
                after_blank = True
                continue
            heading = _HEADING_RE.match(line)
            is_named = line.lower().rstrip(":") in _NAMED_HEADINGS
            if (heading and len(line.split()) <= 14) or is_named:
                flush()
                if heading:
                    section_id, section_title = heading.group(1), heading.group(2).strip()
                else:
                    section_id, section_title = "", line.rstrip(":")
                after_blank = False
                continue
            bullet = bool(_BULLET_RE.match(line))
            if current.get("bullet") and not bullet and after_blank and indent <= current["indent"]:
                flush()  # prose resumes at the bullet's margin → the bullet has ended
            after_blank = False
            if bullet:
                flush()
                current = {"text": _BULLET_RE.sub("", line), "bullet": True, "page": page_no, "indent": indent,
                           "section_id": section_id, "section": section_title}
                continue
            if not current:
                current = {"text": line, "bullet": False, "page": page_no, "indent": indent,
                           "section_id": section_id, "section": section_title}
                continue
            # Continuation line; de-hyphenate "pre-\ncondition" style breaks.
            if current["text"].endswith("-") and line[:1].islower():
                current["text"] += line
            else:
                current["text"] += " " + line
    flush()
    return blocks


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def modality_of(text: str) -> str:
    for label, pattern in MODALITY_CUES:
        if pattern.search(text):
            return label
    return ""


def ingest_sop(path: str) -> Dict[str, Any]:
    """Parse one SOP/guideline document into chunks and normative rules."""
    doc = os.path.splitext(os.path.basename(path))[0]
    raw_pages = extract_pages(path)
    pages = _strip_running_lines([p.splitlines() for p in raw_pages])
    blocks = _blocks(pages)
    opening = []
    for line in (pages[0] if pages else []):
        if not line.strip():
            if opening:
                break
            continue
        opening.append(_clean_line(line))
    title = " ".join(opening)[:120] or doc

    # ── chunks: consecutive blocks of the same section, ~CHUNK_WORDS words ──
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    meta: Dict[str, Any] = {}

    def emit():
        if buf:
            chunks.append({**meta, "text": " ".join(buf)})

    for b in blocks:
        units = _sentences(b["text"]) if len(b["text"].split()) > 1.5 * config.CHUNK_WORDS else [b["text"]]
        for unit in units:
            unit = ("• " if b["bullet"] else "") + unit
            same_section = meta and meta["section"] == b["section"]
            size = sum(len(x.split()) for x in buf)
            if buf and (not same_section or size + len(unit.split()) > config.CHUNK_WORDS):
                emit()
                overlap = buf[-config.CHUNK_OVERLAP_SENTENCES:] if same_section and config.CHUNK_OVERLAP_SENTENCES else []
                tail = _sentences(overlap[0])[-1:] if overlap else []
                buf = tail[:]
                meta = {}
            if not meta:
                meta = {"doc": doc, "page": b["page"], "section_id": b["section_id"], "section": b["section"]}
            buf.append(unit)
    emit()

    # ── normative rules: every bullet, plus deontic sentences in prose ──
    rules: List[Dict[str, Any]] = []
    lead = ""
    for b in blocks:
        if b["bullet"]:
            candidates = [b["text"]]
        else:
            sents = _sentences(b["text"])
            lead = sents[-1] if sents and sents[-1].endswith(":") else ""
            candidates = sents
        for sent in candidates:
            words = len(sent.split())
            if words < 6:
                continue
            modality = modality_of(sent)
            if not modality:
                continue
            rules.append({
                "doc": doc, "page": b["page"], "section_id": b["section_id"], "section": b["section"],
                "text": sent if words <= 140 else " ".join(sent.split()[:140]) + " …",
                "modality": modality, "lead": lead if b["bullet"] else "",
            })

    return {"doc": doc, "title": title, "path": path, "pages": len(raw_pages), "chunks": chunks, "rules": rules}


def sop_label(rec: Dict[str, Any]) -> str:
    sec = f"§{rec['section_id']} " if rec.get("section_id") else ""
    name = rec.get("section") or ""
    return f"{rec['doc']} {sec}{name} (p.{rec['page']})".replace("  ", " ").strip()
