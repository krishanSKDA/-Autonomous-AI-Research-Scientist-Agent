import streamlit as st
import time
import threading
from src.graph.pipeline import graph

# ── page config ──
st.set_page_config(
    page_title="Research Agent",
    page_icon="🧠",
    layout="wide"
)

# ── custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
    }
    .stage-done  { color: #22c55e; font-weight: 600; }
    .stage-run   { color: #f59e0b; font-weight: 600; }
    .stage-wait  { color: #475569; }
    .paper-score {
        background: #1e3a5f;
        color: #60a5fa;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── header ──
st.title("🧠 Autonomous Research Agent")
st.caption("Multi-agent AI system · LangGraph + Groq + ChromaDB + ArXiv")
st.divider()

# ── sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    max_papers = st.slider("Max papers to fetch", 3, 10, 5)
    show_draft = st.toggle("Show draft report", value=False)
    st.divider()
    st.markdown("**Pipeline agents**")
    st.markdown("1. 🗺️ Planner")
    st.markdown("2. 🔍 Search")
    st.markdown("3. 📖 Reader")
    st.markdown("4. 🔬 Critic")
    st.markdown("5. 🧠 Synthesizer")
    st.markdown("6. ✍️ Writer")
    st.divider()
    st.markdown("**Free APIs used**")
    st.markdown("• Groq — LLM inference")
    st.markdown("• ArXiv — paper search")
    st.markdown("• ChromaDB — vector store")

# ── sample topics ──
st.markdown("**💡 Try a sample topic:**")
samples = [
    "AI-based counter-drone systems for airspace security",
    "Edge AI inference optimization for embedded systems",
    "Autonomous UAV swarm coordination using reinforcement learning",
    "Large language models for automated threat intelligence",
    "IoT-based patient monitoring using federated learning"
]
cols = st.columns(3)
for i, sample in enumerate(samples[:3]):
    if cols[i].button(sample[:40] + "...", use_container_width=True):
        st.session_state["query_input"] = sample

# ── query input ──
query = st.text_input(
    "Enter your research query",
    placeholder="e.g. AI-based drone threat detection using computer vision",
    key="query_input"
)

run = st.button("🚀 Start Research", type="primary", use_container_width=True)

# ── pipeline stages ──
STAGES = [
    ("🗺️", "Planner",      "Decomposing query into sub-questions"),
    ("🔍", "Search",       "Searching ArXiv for papers"),
    ("📖", "Reader",       "Reading PDFs and storing in ChromaDB"),
    ("🔬", "Critic",       "Scoring relevance and flagging gaps"),
    ("🧠", "Synthesizer",  "Merging findings into knowledge graph"),
    ("✍️", "Writer",       "Drafting and self-reflecting on report"),
]

def run_pipeline(query: str):
    """Run the full LangGraph pipeline."""
    return graph.invoke({
        "query":            query,
        "sub_questions":    [],
        "search_keywords":  [],
        "paper_urls":       [],
        "paper_metadata":   [],
        "extracted_chunks": [],
        "rag_answers":      [],
        "relevance_scores": [],
        "flagged_gaps":     [],
        "synthesis":        "",
        "knowledge_graph":  {},
        "draft_report":     "",
        "final_report":     "",
        "iteration":        0,
        "messages":         []
    })

# ── main run ──
if run and query.strip():
    st.divider()

    # progress section
    st.subheader("⚡ Pipeline Progress")
    stage_placeholders = []
    progress_cols = st.columns(2)

    for i, (icon, name, desc) in enumerate(STAGES):
        col = progress_cols[i % 2]
        ph = col.empty()
        ph.markdown(f"⬜ **{icon} {name}** — {desc}")
        stage_placeholders.append(ph)

    progress_bar = st.progress(0, text="Starting pipeline...")
    status_text  = st.empty()

    # run pipeline with stage simulation
    result_container = {}
    error_container  = {}

    def pipeline_thread():
        try:
            result_container["result"] = run_pipeline(query)
        except Exception as e:
            error_container["error"] = str(e)

    thread = threading.Thread(target=pipeline_thread)
    thread.start()

    # animate stages while pipeline runs
    stage_idx = 0
    while thread.is_alive():
        elapsed = 0
        while thread.is_alive() and elapsed < 18:
            time.sleep(1)
            elapsed += 1

        if stage_idx < len(STAGES):
            # mark previous as done
            if stage_idx > 0:
                icon, name, desc = STAGES[stage_idx - 1]
                stage_placeholders[stage_idx - 1].markdown(
                    f"✅ **{icon} {name}** — {desc}"
                )
            # mark current as running
            icon, name, desc = STAGES[stage_idx]
            stage_placeholders[stage_idx].markdown(
                f"🔄 **{icon} {name}** — {desc}"
            )
            progress_bar.progress(
                (stage_idx + 1) / len(STAGES),
                text=f"Running {name} agent..."
            )
            status_text.info(f"🔄 {name} agent is working...")
            stage_idx += 1

    thread.join()

    # mark all done
    for i, (icon, name, desc) in enumerate(STAGES):
        stage_placeholders[i].markdown(f"✅ **{icon} {name}** — {desc}")
    progress_bar.progress(1.0, text="Pipeline complete!")

    # ── error ──
    if "error" in error_container:
        st.error(f"❌ Pipeline failed: {error_container['error']}")
        st.stop()

    result = result_container["result"]
    status_text.success("✅ Research complete!")

    st.divider()

    # ── results ──
    st.subheader("📊 Results")

    # metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers Found",     len(result["paper_metadata"]))
    m2.metric("Sub-questions",    len(result["sub_questions"]))
    m3.metric("Research Gaps",    len(result["flagged_gaps"]))
    m4.metric("Reflection Passes", result["iteration"])

    st.divider()

    # tabs for results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Papers",
        "❓ Q&A",
        "⚠️ Gaps",
        "🕸️ Knowledge Graph",
        "📝 Final Report"
    ])

    # ── tab 1: papers ──
    with tab1:
        st.markdown("### Papers Found")
        for i, paper in enumerate(result["paper_metadata"][:max_papers]):
            score = result["relevance_scores"][i] if i < len(result["relevance_scores"]) else 0
            with st.expander(f"{'🟢' if score >= 0.7 else '🟡' if score >= 0.5 else '🔴'} {paper['title'][:80]}"):
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")
                col_a.markdown(f"**Published:** {paper['published']}")
                col_a.markdown(f"**Abstract:** {paper['abstract'][:400]}...")
                col_b.metric("Relevance", f"{score*100:.0f}%")
                col_b.markdown(f"[📥 PDF]({paper['url']})")

    # ── tab 2: Q&A ──
    with tab2:
        st.markdown("### Sub-questions & RAG Answers")
        for i, (q, a) in enumerate(zip(
            result["sub_questions"],
            result["rag_answers"]
        )):
            st.markdown(f"**Q{i+1}: {q}**")
            st.info(a)
            st.divider()

    # ── tab 3: gaps ──
    with tab3:
        st.markdown("### Research Gaps Identified")
        for gap in result["flagged_gaps"]:
            st.warning(f"⚠️ {gap}")

    # ── tab 4: knowledge graph ──
    with tab4:
        st.markdown("### Knowledge Graph")
        kg = result["knowledge_graph"]
        if kg:
            kc1, kc2 = st.columns(2)
            with kc1:
                st.markdown("**🎯 Core Concept**")
                st.success(kg.get("core_concept", "N/A"))
                st.markdown("**🔬 Methods**")
                for m in kg.get("methods", []):
                    st.markdown(f"- {m}")
            with kc2:
                st.markdown("**🏷️ Key Themes**")
                for t in kg.get("key_themes", []):
                    st.markdown(f"- {t}")
                st.markdown("**📊 Datasets**")
                for d in kg.get("datasets", []):
                    st.markdown(f"- {d}")

            st.markdown("**🔗 Relationships**")
            for rel in kg.get("relationships", []):
                st.markdown(
                    f"`{rel.get('from')}` → **{rel.get('relation')}** → `{rel.get('to')}`"
                )

    # ── tab 5: final report ──
    with tab5:
        st.markdown("### 📝 Final Research Report")
        if show_draft:
            st.markdown("#### Draft")
            st.text_area("Draft report", result["draft_report"], height=300)
            st.markdown("#### Final (after reflection)")

        st.markdown(result["final_report"])
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=f"# Research Report\n\n**Query:** {result['query']}\n\n{result['final_report']}",
            file_name="research_report.md",
            mime="text/markdown"
        )

elif run and not query.strip():
    st.warning("⚠️ Please enter a research query first.")