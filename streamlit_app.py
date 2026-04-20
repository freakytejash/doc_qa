import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4A90D9;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 13px;
    }
    .relevance-badge {
        background: #e8f4fd;
        color: #1a6fa8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .answer-box {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #c8e0f7;
        font-size: 15px;
        line-height: 1.7;
    }
    .meta-tag {
        color: #666;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=30)  # change 3 → 30
        return r.json() if r.ok else None
    except requests.exceptions.ConnectionError:
        return None


def get_documents() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/documents/", timeout=5)
        return r.json().get("documents", []) if r.ok else []
    except Exception:
        return []


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API_BASE}/documents/ingest",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=120,
    )
    return r.json()


def query(question: str, doc_id: str | None, top_k: int) -> dict:
    payload = {"question": question, "top_k": top_k}
    if doc_id:
        payload["doc_id"] = doc_id
    try:
        r = requests.post(
            f"{API_BASE}/query/",
            json=payload,
            timeout=60,  # change from 60 to 120
        )
        if not r.text:
            return {"detail": "Server returned empty response. Try again."}, 500
        return r.json(), r.status_code
    except requests.exceptions.ReadTimeout:
        return {"detail": "Request timed out. The model is still loading, please try again."}, 500
    except requests.exceptions.JSONDecodeError:
        return {"detail": f"Invalid response from server: {r.text[:200]}"}, 500


def delete_document(doc_id: str) -> bool:
    r = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
    return r.ok


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Document Q&A")
st.caption("Upload a PDF and ask questions — answers are grounded in your document.")

# ── Server status ─────────────────────────────────────────────────────────────
health = check_health()
if not health:
    st.error(
        "⚠️ Cannot reach the API server. "
        "Make sure it's running: `python main.py`",
        icon="🔴",
    )
    st.stop()

with st.expander("🟢 Server connected", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("LLM", f"{health['llm_provider']} / {health['llm_model'].split('-')[0]}")
    col2.metric("Embeddings", health["embedding_provider"])
    col3.metric("Model", health["embedding_model"])

st.divider()

# ── Layout: sidebar + main ────────────────────────────────────────────────────
sidebar, main = st.columns([1, 2], gap="large")

# ══ SIDEBAR ══════════════════════════════════════════════════════════════════
with sidebar:
    st.subheader("📂 Documents")

    # Upload
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("📥 Ingest Document", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting {uploaded.name}…"):
                result = ingest_pdf(uploaded.read(), uploaded.name)

            if "doc_id" in result:
                if result.get("already_existed"):
                    st.info(f"Already indexed: **{uploaded.name}**")
                else:
                    st.success(
                        f"✅ Indexed **{result['total_chunks']}** chunks "
                        f"from **{result['total_pages']}** pages."
                    )
                st.session_state["last_doc_id"] = result["doc_id"]
            else:
                st.error(f"Ingestion failed: {result.get('detail', 'Unknown error')}")

    st.markdown("---")

    # Document list
    docs = get_documents()
    if not docs:
        st.caption("No documents indexed yet.")
    else:
        st.caption(f"{len(docs)} document(s) in index")
        for doc in docs:
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**{doc['filename']}**  \n`{doc['doc_id'][:8]}…`")
            if col_b.button("🗑", key=f"del_{doc['doc_id']}", help="Delete"):
                if delete_document(doc["doc_id"]):
                    st.success("Deleted.")
                    st.rerun()

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    doc_choices = {"All documents": None} | {
        f"{d['filename']} ({d['doc_id'][:8]}…)": d["doc_id"] for d in docs
    }
    selected_label = st.selectbox("Search in", options=list(doc_choices.keys()))
    selected_doc_id = doc_choices[selected_label]

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

# ══ MAIN ════════════════════════════════════════════════════════════════════
with main:
    st.subheader("💬 Ask a Question")

    # Quick question suggestions
    st.caption("Try one of these:")
    suggestions = [
        "What are the primal cuts of beef?",
        "What is the gestation period for beef cattle?",
        "Describe cow-calf operations.",
        "What breeds of dairy cattle are in the US?",
        "How are horses measured?",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion[:28] + "…", key=f"sug_{i}", use_container_width=True):
            st.session_state["question_input"] = suggestion

    # Question input
    question = st.text_area(
        "Your question",
        value=st.session_state.get("question_input", ""),
        placeholder="e.g. What are the primal cuts of beef?",
        height=80,
        label_visibility="collapsed",
    )

    ask_clicked = st.button("Ask ↗", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not docs:
            st.warning("Please ingest a document first.")
        else:
            with st.spinner("Retrieving and generating answer…"):
                response, status_code = query(question.strip(), selected_doc_id, top_k)

            if status_code != 200:
                st.error(f"Error: {response.get('detail', 'Unknown error')}")
            else:
                # ── Answer ────────────────────────────────────────────────
                st.markdown("#### Answer")
                st.markdown(
                    f"<div class='answer-box'>{response['answer']}</div>",
                    unsafe_allow_html=True,
                )

                # ── Metadata ──────────────────────────────────────────────
                st.markdown(
                    f"<p class='meta-tag'>LLM: {response['llm_provider']} &nbsp;|&nbsp; "
                    f"Embeddings: {response['embedding_provider']} &nbsp;|&nbsp; "
                    f"Sources used: {len(response['sources'])}</p>",
                    unsafe_allow_html=True,
                )

                # ── Sources ───────────────────────────────────────────────
                with st.expander(f"📚 View {len(response['sources'])} source chunks", expanded=False):
                    for i, src in enumerate(response["sources"], 1):
                        score_pct = int(src["relevance_score"] * 100)
                        st.markdown(
                            f"""<div class='source-card'>
                            <b>Chunk {i}</b> &nbsp;
                            <span class='meta-tag'>{src['filename']} — Page {src['page']}</span>
                            &nbsp; <span class='relevance-badge'>{score_pct}% match</span>
                            <br><br>{src['text']}
                            </div>""",
                            unsafe_allow_html=True,
                        )

                # Save to history
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].insert(0, {
                    "question": question,
                    "answer": response["answer"],
                })

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state.get("history"):
        st.markdown("---")
        st.subheader("🕑 Previous Questions")
        for item in st.session_state["history"][:5]:
            with st.expander(f"Q: {item['question'][:80]}…"):
                st.write(item["answer"])
