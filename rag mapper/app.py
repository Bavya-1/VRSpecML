"""VRSpecML · RAG profile generator (Streamlit UI): project form → profile workspace."""
import json
import os
import time
from typing import List, Optional

import pandas as pd
import streamlit as st

import config
import server
from domain import Domain, create_project, delete_project, load_project
from exporters import residue_log, to_dot, to_json, to_markdown, to_ocl, traceability
from knowledge_base import build_knowledge_base, kb_status, load_knowledge_base
from llm_client import LLMError, QwenClient, check_connection, resolve_model
from metamodel import METACLASSES
from pipeline import (Pipeline, add_stereotype, cleanup_enumerations, new_profile, remove_stereotype,
                      rename_stereotype)
from schemas import EvidenceRef, Profile

st.set_page_config(page_title="VRSpecML", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem; max-width: 1280px;}
    div[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:.45rem .8rem;}
    div[data-testid="stMetricValue"] {font-size:1.35rem;}
    .muted {color:#64748b; font-size:.82rem;}
    </style>""",
    unsafe_allow_html=True,
)

LOCAL_HOSTS = {server.LOCAL_HOST, f"http://localhost:{config.LLAMA_SERVER_PORT}/v1"}
MC_ICON = {"Scene": "🗺️", "Article": "📦", "ActionResponse": "⚡", "Behavior": "🛡️", "Timeline": "⏱️"}


# ── resources ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_kb(index_dir: str, built_at: str, _domain: Domain):
    from embedder import embed_one

    kb = load_knowledge_base(_domain)
    if kb is not None:
        embed_one("warm up")  # load the encoder once, before the first task
    return kb


@st.cache_data(ttl=8, show_spinner=False)
def llm_status(host: str):
    return check_connection(host, timeout=2)


def state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def open_project(slug: Optional[str]):
    """Switch page: a project slug opens its workspace, None returns to the project form."""
    st.session_state.project = slug
    st.session_state.profile = None
    st.session_state.ctx = ""
    st.session_state.editor_version = state("editor_version", 0) + 1
    st.rerun()


# ── small renderers ──────────────────────────────────────────────────────────
def ref_links(refs: List[EvidenceRef]) -> str:
    parts = []
    for r in refs:
        if r.uri and r.kind in ("class", "property", "unit", "partition"):
            parts.append(f"[{r.label}]({r.uri}) `{r.source or ''}`")
        else:
            parts.append(r.label)
    return " · ".join(parts)


def evidence_popover(refs: List[EvidenceRef], key: str, label: str = "Evidence"):
    if not refs:
        return
    with st.popover(f"{label} ({len(refs)})", key=key):
        for r in refs:
            if r.kind in ("chunk", "rule", "context"):
                st.markdown(f"**{r.label}**")
                st.caption(r.text[:900])
            elif r.uri:
                st.markdown(f"**[{r.label}]({r.uri})** · `{r.id}`")
                if r.text:
                    st.caption(r.text[:400])
            else:
                st.markdown(f"**{r.label}**")
                if r.text:
                    st.caption(r.text[:400])


def run_with_status(label: str, fn):
    """Run a long task with a live progress bar; returns True on success."""
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")
        t0 = time.time()

        def progress(msg: str, frac: float):
            bar.progress(min(1.0, max(0.0, frac)), text=msg)

        try:
            fn(progress)
        except LLMError as exc:
            status.update(label=f"Qwen error: {exc}", state="error")
            return False
        except Exception as exc:  # surfaced to the user rather than a blank page
            status.update(label=f"Failed: {exc}", state="error")
            st.exception(exc)
            return False
        status.update(label=f"{label.rstrip('…')} — done in {time.time() - t0:.0f}s", state="complete", expanded=False)
    return True


# ── sidebar ──────────────────────────────────────────────────────────────────
def sidebar(domain: Optional[Domain]):
    with st.sidebar:
        st.markdown(f"### 🌱 {domain.name if domain else 'VRSpecML'}")

        for key, default in (("llm_model", config.DEFAULT_LLM_MODEL), ("temperature", config.LLM_TEMPERATURE),
                             ("parallel", config.LLM_PARALLEL), ("per_metaclass", config.STEREOTYPES_PER_METACLASS)):
            state(key, default)
        host = state("llm_host", config.DEFAULT_LLM_HOST)
        info = llm_status(host)

        # Qwen status and model properties share one dropdown
        with st.popover(f"Model properties {'🟢' if info['ok'] else '🔴'}", icon=":material/tune:", width="stretch"):
            st.markdown("**Qwen**")
            if info["ok"]:
                st.markdown(f"🟢 {', '.join(info['models']) or 'online'}")
            else:
                st.markdown(f"🔴 {info['error']}")
                if host.rstrip("/") in LOCAL_HOSTS:
                    if st.button("Start Qwen", icon=":material/play_arrow:", width="stretch",
                                 help=f"Launch llama-server with {os.path.basename(config.QWEN_GGUF)} on the GPU"):
                        with st.spinner("Loading Qwen on the GPU…"):
                            res = server.start(wait=True)
                        llm_status.clear()
                        if res.get("ok"):
                            st.rerun()
                        st.error(res.get("message", ""))
            st.divider()
            st.text_input("Host", key="llm_host",
                          help="…/v1 = llama-server (OpenAI API); otherwise Ollama (e.g. http://localhost:11434)")
            st.text_input("Model", key="llm_model", help=f"Qwen 2.5 7B only. Ollama tag: {config.OLLAMA_QWEN_MODEL}")
            st.slider("Temperature", 0.0, 1.0, key="temperature", step=0.05)
            st.slider("Parallel requests", 1, 4, key="parallel", help="Match the server's slots (-np)")
            st.number_input("Per metaclass", 3, 8, key="per_metaclass", help="Stereotypes generated for each metaclass")
    return info


# ── Home: project form ───────────────────────────────────────────────────────
def home():
    sidebar(None)
    st.markdown("## 🌱 VRSpecML · profile generator")
    st.caption("Create a project from your ontologies and guideline documents, then describe VR training contexts "
               "and generate stereotypes, tagged values and OCL invariants.")

    with st.form("new_project"):
        name = st.text_input("Context name", placeholder="e.g. Sustainable smart farm")
        st.caption("Ontologies and guidelines are optional. With none, the built-in farm knowledge base "
                   "(AGRO · ENVO · CHMO ontologies, FAO guidelines) is used. Files you upload are indexed for this project.")
        onts = st.file_uploader("Ontologies (optional)", type=[e.lstrip(".") for e in config.ONTOLOGY_EXTENSIONS],
                                accept_multiple_files=True,
                                help=".owl / .rdf / .xrdf are read as RDF/XML, .ttl as Turtle")
        sops = st.file_uploader("Guideline documents (optional)", type=[e.lstrip(".") for e in config.SOP_EXTENSIONS],
                                accept_multiple_files=True)
        submitted = st.form_submit_button("Create project", type="primary", icon=":material/rocket_launch:")

    if submitted:
        if not name.strip():
            st.error("Enter a context name.")
        else:
            domain = create_project(name, [(f.name, f.getvalue()) for f in onts or []], [(f.name, f.getvalue()) for f in sops or []])
            status = kb_status(domain)
            if status["exists"] and not status["stale"]:  # e.g. the shared built-in knowledge base: nothing to build
                open_project(domain.id)
            elif run_with_status("Building knowledge base…", lambda p: build_knowledge_base(domain, p)):
                open_project(domain.id)
            else:
                delete_project(domain.id)  # keep unusable projects out of the projects folder


# ── Stereotypes + tagged values ──────────────────────────────────────────────
def tab_stereotypes(profile: Optional[Profile], domain: Domain, can_gen: bool, can_run: bool, run, run_tags):
    c1, c2, c3 = st.columns([4, 1.4, 1], vertical_alignment="bottom")
    has = bool(profile and profile.stereotypes)
    with c1:
        if has:
            st.caption("Re-generating replaces all stereotypes, tagged values and invariants.")
    with c2:
        with st.popover("Add stereotype", icon=":material/add:", width="stretch", disabled=not (has and can_run)):
            name = st.text_input("Name", key="add_name", placeholder="e.g. SoilSampler")
            mc = st.selectbox("Metaclass", list(METACLASSES), key="add_mc")
            desc = st.text_area("Description (optional)", key="add_desc", height=80)
            if st.button("Add & generate tagged values", key="add_go", type="primary", icon=":material/auto_awesome:"):
                run_tags("add", name=name, metaclass=mc, description=desc)
    if c3.button("Generate", key="go1", type="primary", icon=":material/auto_awesome:", disabled=not can_gen, width="stretch"):
        run(1)
    if not has:
        st.info("Write a context above, then Generate: stereotypes on the five VR metaclasses, each with its tagged values, "
                "grounded in your ontologies and guideline documents (when you provided any).", icon=":material/account_tree:")
        return

    state("mc_pick", "Scene")
    pick = st.segmented_control("Metaclass", list(METACLASSES), key="mc_pick",
                                format_func=lambda m: f"{MC_ICON[m]} {m}", label_visibility="collapsed")
    for s in [x for x in profile.stereotypes if x.extends == (pick or "Scene")]:
        with st.container(border=True):
            h1, h2 = st.columns([6, 1], vertical_alignment="center")
            badge = f":{'blue' if s.mapping == 'ontology' else 'orange'}-badge[{'ontology' if s.mapping == 'ontology' else 'LLM-suggested'}]"
            h1.markdown(f"**«{s.name}»** extends *{s.extends}* &nbsp; {badge}")
            if h2.button("Remove", key=f"rm_{s.sid}", icon=":material/delete:", width="stretch",
                         help="Delete this stereotype with its tagged values and any invariants that use it"):
                remove_stereotype(profile, s.sid)
                st.session_state.editor_version = state("editor_version", 0) + 1
                st.rerun()
            if s.description:
                st.markdown(s.description)
            if s.ontology_refs:
                st.markdown("🔗 " + ref_links(s.ontology_refs))
            evidence_popover(s.sop_refs, key=f"sop_{s.sid}", label="SOP")
            for n in s.notes:
                st.caption(f"⚠ {n}")
            tags = profile.tagged_values.get(s.sid, [])
            if tags:
                st.caption(f"Tagged values · {len(tags)}")
                st.dataframe(pd.DataFrame([{
                    "tag": t.name, "type": t.type, "mult": t.multiplicity, "unit": t.unit,
                    "range": "" if t.min is None and t.max is None else
                    f"{'' if t.min is None else f'{t.min:g}'} … {'' if t.max is None else f'{t.max:g}'}",
                } for t in tags]), hide_index=True, width="stretch")
            elif st.button("Generate tagged values", key=f"gen_{s.sid}", icon=":material/auto_awesome:", disabled=not can_run):
                run_tags("one", sid=s.sid)


# ── OCL invariants ────────────────────────────────────────────────────────────────────
def tab_ocl(profile: Optional[Profile], domain: Domain, can_run: bool, run):
    kept = profile.kept() if profile else []
    has_tags = bool(profile and any(profile.tagged_values.get(s.sid) for s in kept))
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    invs = [i for i in (profile.invariants if profile else []) if profile.by_sid(i.context_sid) and profile.by_sid(i.context_sid).keep]
    with c1:
        if kept and invs:
            covered = len({i.context_sid for i in invs})
            st.progress(covered / max(1, len(kept)), text=f"{covered}/{len(kept)} stereotypes")
    force = c2.toggle("All", key="force3", help="Regenerate stereotypes that already have invariants")
    if c3.button("Generate", key="go3", type="primary", icon=":material/auto_awesome:",
                 disabled=not (can_run and has_tags), width="stretch", help=None if has_tags else "Generate ① first"):
        run(3, force=force)
    if not invs:
        st.info("Guideline rules + ontology axioms → OCL invariants written by Qwen.", icon=":material/rule:")
        return

    st.metric("Invariants", len(invs))
    names = ["All"] + [s.name for s in kept if any(i.context_sid == s.sid for i in invs)]
    ctx_pick = st.selectbox("Context", names, key="inv_ctx", label_visibility="collapsed")
    for inv in invs:
        if ctx_pick not in (None, "All") and inv.context != ctx_pick:
            continue
        with st.container(border=True):
            st.markdown(f"**{inv.iid} · {inv.name}**")
            st.code(f"context {inv.context}\ninv {inv.name}: {inv.body}", language=None, wrap_lines=True)
            if inv.reading or inv.requirement:
                st.caption(inv.reading or inv.requirement)
            cols = st.columns([1, 1, 4])
            with cols[0]:
                evidence_popover(inv.evidence, key=f"iev_{inv.iid}")
            with cols[1]:
                with st.popover("Edit", icon=":material/edit:", key=f"edit_{inv.iid}"):
                    body = st.text_area("OCL", inv.body, key=f"body_{inv.iid}", height=120, label_visibility="collapsed")
                    if st.button("Save", key=f"chk_{inv.iid}", type="primary"):
                        inv.body = body.strip()
                        st.rerun()
                    if st.button("Delete", key=f"del_{inv.iid}", icon=":material/delete:"):
                        profile.invariants = [i for i in profile.invariants if i.iid != inv.iid]
                        cleanup_enumerations(profile)
                        st.rerun()


# ── Export ───────────────────────────────────────────────────────────────────
def tab_export(profile: Optional[Profile], kb, domain: Domain):
    if not profile or not profile.stereotypes:
        st.info("Nothing to export yet.", icon=":material/download:")
        return
    stamp = time.strftime("%Y%m%d_%H%M")
    base = f"vrspecml_{domain.id}_{stamp}"
    c = st.columns(4)
    c[0].download_button("JSON", to_json(profile, kb.manifest if kb else None, domain), f"{base}.json",
                         "application/json", icon=":material/data_object:", width="stretch")
    c[1].download_button("OCL", to_ocl(profile), f"{base}.ocl", "text/plain", icon=":material/rule:", width="stretch")
    c[2].download_button("Markdown", to_markdown(profile), f"{base}.md", "text/markdown", icon=":material/description:",
                         width="stretch")
    c[3].download_button("Diagram", to_dot(profile), f"{base}.dot", "text/vnd.graphviz", icon=":material/schema:",
                         width="stretch")


# ── References ───────────────────────────────────────────────────────────────
def tab_references(profile: Optional[Profile], domain: Domain):
    if not profile or not profile.stereotypes:
        st.info("Nothing to show yet.", icon=":material/menu_book:")
        return
    view = st.segmented_control("View", ["Stereotypes", "Traceability", "Residue", "Log"], default="Stereotypes",
                                key="ref_view", label_visibility="collapsed")
    if view == "Stereotypes":
        stereotype_table(profile, domain)
    elif view == "Traceability":
        rows = traceability(profile)
        if rows:
            st.dataframe(pd.DataFrame([{"requirement": r["requirement"], "source": "; ".join(r["sources"]),
                                        "elements": ", ".join(r["elements"]), "invariant": r["invariant"]} for r in rows]), hide_index=True, width="stretch")
        else:
            st.caption("Generate ② to trace requirements to invariants.")
    elif view == "Residue":
        rows = residue_log(profile)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("Empty.")
    else:
        if profile.task_log:
            st.dataframe(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
                                       for r in profile.task_log]), hide_index=True, width="stretch")
        else:
            st.caption("Empty.")


def stereotype_table(profile: Profile, domain: Domain):
    """Editable overview of all stereotypes: rename and change metaclass."""
    sts = profile.stereotypes
    version = state("editor_version", 0)
    df = pd.DataFrame([{
        "sid": s.sid, "name": s.name, "extends": s.extends,
        "mapping": "ontology" if s.mapping == "ontology" else "LLM-suggested",
        "ontology": s.primary.label if s.primary else "—", "grounding": s.grounding,
    } for s in sts])
    edited = st.data_editor(
        df, key=f"st_editor_{version}", hide_index=True, width="stretch", disabled=["sid", "mapping", "ontology", "grounding"],
        column_config={
            "sid": st.column_config.TextColumn("", width="small"),
            "name": st.column_config.TextColumn("Stereotype"),
            "extends": st.column_config.SelectboxColumn("Metaclass", options=list(METACLASSES)),
            "ontology": st.column_config.TextColumn("Ontology class"),
            "grounding": st.column_config.ProgressColumn("Grounding", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    changed = False
    for row in edited.to_dict("records"):
        s = profile.by_sid(row["sid"])
        if s is None:
            continue
        if row["extends"] != s.extends and row["extends"] in METACLASSES:
            s.extends, changed = row["extends"], True
        if str(row["name"]).strip() and str(row["name"]).strip() != s.name:
            rename_stereotype(profile, s.sid, str(row["name"]))
            changed = True
    if changed:
        st.session_state.editor_version = version + 1
        st.rerun()


# ── Workspace ────────────────────────────────────────────────────────────────
def workspace(slug: str):
    try:
        domain = load_project(slug)
    except (OSError, ValueError, KeyError):
        st.error(f"Project “{slug}” could not be loaded.")
        if st.button("Back to projects"):
            open_project(None)
        return
    profile: Optional[Profile] = state("profile", None)
    info = sidebar(domain)

    status = kb_status(domain)
    if not status["exists"] or status["stale"]:  # e.g. the project's files were changed on disk
        if not run_with_status("Building knowledge base…", lambda p: build_knowledge_base(domain, p)):
            return
        st.rerun()
    kb = get_kb(domain.index_dir, status.get("built_at", ""), domain)

    h1, h2 = st.columns([1, 6], vertical_alignment="center")
    if h1.button("Projects", icon=":material/arrow_back:", help="Back to the project form"):
        open_project(None)
    h2.markdown(f"## 🌱 {domain.name}")

    state("ctx", profile.context if profile else "")
    context = st.text_area("Context", key="ctx", height=130, label_visibility="collapsed",
                           placeholder="Describe the VR training context you want a profile for: setting, actors, "
                                       "objects, tasks, hazards, measurements, limits…")
    if profile and profile.stereotypes and context.strip() != profile.context.strip():
        st.caption("⚠ Context changed since ① — regenerate stereotypes to apply it.")

    can_run = bool(info["ok"] and kb is not None)
    if not info["ok"]:
        st.warning("Qwen is offline — start it from Model properties in the sidebar.", icon=":material/power_off:")

    def make_pipe() -> Pipeline:
        host = st.session_state.llm_host
        model = resolve_model(host, st.session_state.get("llm_model", config.DEFAULT_LLM_MODEL))
        client = QwenClient(host=host, model=model, temperature=st.session_state.get("temperature", config.LLM_TEMPERATURE))
        return Pipeline(kb, domain, client, parallel=st.session_state.get("parallel", config.LLM_PARALLEL))

    def run(task: int, force: bool = False):
        pipe = make_pipe()
        prof = st.session_state.profile
        if task == 1:  # stereotypes, then the tagged values of every stereotype
            prof = new_profile(domain, context)
            per_metaclass = st.session_state.get("per_metaclass", config.STEREOTYPES_PER_METACLASS)

            def stereotypes_and_tags(p):
                pipe.run_task1(prof, per_metaclass, lambda m, f: p(m, 0.5 * f))
                st.session_state.profile = prof
                pipe.run_task2(prof, progress=lambda m, f: p(m, 0.5 + 0.5 * f))

            ok = run_with_status("Generating stereotypes and tagged values…", stereotypes_and_tags)
        else:
            ok = run_with_status("Generating OCL invariants…", lambda p: pipe.run_task3(prof, force=force, progress=p))
        if ok:
            st.session_state.profile = prof
            st.session_state.editor_version = state("editor_version", 0) + 1
            st.rerun()

    def run_tags(kind: str, sid: str = "", name: str = "", metaclass: str = "Scene", description: str = ""):
        """Tagged values for one stereotype: a newly added one (kind='add') or an existing one without any."""
        prof = st.session_state.profile
        if kind == "add":
            try:
                sid = add_stereotype(prof, name, metaclass, description).sid
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.mc_pick = metaclass
        label = prof.by_sid(sid).name
        pipe = make_pipe()
        ok = run_with_status(f"Generating tagged values for «{label}»…", lambda p: pipe.run_task2(prof, sids=[sid], progress=p))
        if ok:
            for key in ("add_name", "add_desc"):
                st.session_state.pop(key, None)
            st.session_state.editor_version = state("editor_version", 0) + 1
            st.rerun()

    t1, t2, t3, t4 = st.tabs(["① Stereotypes & tagged values", "② OCL", "Export", "References"])
    with t1:
        tab_stereotypes(profile, domain, can_run and bool(context.strip()), can_run, run, run_tags)
    with t2:
        tab_ocl(profile, domain, can_run, run)
    with t3:
        tab_export(profile, kb, domain)
    with t4:
        tab_references(profile, domain)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    slug = state("project", None)
    if slug:
        workspace(slug)
    else:
        home()


main()
