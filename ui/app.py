"""Streamlit front end. Start the API first, then:
    poetry run streamlit run ui/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("CROSSVERIFY_API", "http://127.0.0.1:8000")
BADGES = {"aligned": "🟢 aligned", "low_alignment": "🟠 low image-caption alignment", "unknown": "⚪ not calibrated"}

st.set_page_config(page_title="CrossVerify", layout="wide")
st.title("CrossVerify")
st.caption("Multi-modal news retrieval with provenance. A low alignment score is not proof of misinformation.")

for key, default in (("results", []), ("relevant", set()), ("irrelevant", set()), ("last_query", "")):
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Search settings")
    mode = st.selectbox("Ranking", ["rrf", "clip", "bm25"], help="rrf = CLIP + BM25 fused")
    top_k = st.slider("Results", 3, 20, 8)
    expand = st.checkbox("Query expansion (WordNet)")
    rerank = st.checkbox("Cross-encoder re-ranking")

tab_text, tab_image, tab_check = st.tabs(["Text search", "Image search", "Check image + caption"])


def call(path: str, **kwargs):
    try:
        response = requests.post(f"{API}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def render_reuse(matches: list[dict]):
    if matches:
        st.warning(f"Image reuse: {len(matches)} other article(s) use the same/near-identical image")
        for m in matches:
            st.markdown(f"- #{m['article_id']} **{m['headline']}** ({m['source']}, {m['date']}) — distance {m['distance']}")


def render_results(results: list[dict], allow_feedback: bool):
    for r in results:
        left, right = st.columns([1, 3])
        with left:
            st.image(f"{API}{r['image_url_local']}", use_container_width=True)
        with right:
            st.markdown(f"### [{r['headline']}]({r['url']})")
            st.caption(f"{r['source']} · {r.get('date') or 'date unknown'} · score {r['score']:.4f}")
            st.write(f"**Caption:** {r.get('caption') or '—'}")
            cons = r.get("consistency")
            if cons:
                st.markdown(f"{BADGES.get(cons['status'], cons['status'])} (cosine {cons['similarity']:.3f})")
            render_reuse(r.get("reuse_matches", []))
            with st.expander("Provenance & why this ranked"):
                st.markdown(
                    f"- **Publisher:** {r['source']}\n- **Article URL:** {r['url']}\n"
                    f"- **Image URL:** {r.get('image_url')}\n- **Fetched:** {r.get('fetched_at')}\n"
                    f"- **Caption source:** {r.get('caption_source')}\n- **pHash:** `{r.get('phash')}`"
                )
                st.json({"ranking": r.get("explanation"), "rerank_score": r.get("rerank_score"),
                         "query_variants": r.get("variants")})
            if allow_feedback:
                aid = r["article_id"]
                c1, c2, _ = st.columns([1, 1, 6])
                if c1.button("👍", key=f"up{aid}"):
                    st.session_state.relevant.add(aid); st.session_state.irrelevant.discard(aid)
                if c2.button("👎", key=f"down{aid}"):
                    st.session_state.irrelevant.add(aid); st.session_state.relevant.discard(aid)
                if aid in st.session_state.relevant:
                    st.caption("Marked relevant")
                if aid in st.session_state.irrelevant:
                    st.caption("Marked not relevant")
        st.divider()


def run_search(query: str, feedback: dict | None = None):
    payload = {"query": query, "mode": mode, "top_k": top_k, "expand": expand, "rerank": rerank, "feedback": feedback}
    data = call("/search", json=payload)
    if data:
        st.session_state.results = data["results"]
        st.session_state.last_query = query


with tab_text:
    query = st.text_input("Describe the news event or image", placeholder="e.g. protesters carrying signs in the rain")
    if st.button("Search", type="primary") and query:
        st.session_state.relevant, st.session_state.irrelevant = set(), set()
        run_search(query)
    if st.session_state.results and st.session_state.last_query:
        if st.session_state.relevant or st.session_state.irrelevant:
            if st.button("Refine results with my feedback (Rocchio)"):
                run_search(st.session_state.last_query, {
                    "relevant": sorted(st.session_state.relevant),
                    "irrelevant": sorted(st.session_state.irrelevant)})
        render_results(st.session_state.results, allow_feedback=True)

with tab_image:
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="img_search")
    if upload and st.button("Find similar images"):
        data = call("/search/image", files={"file": upload.getvalue()}, data={"top_k": top_k})
        if data:
            render_results(data["results"], allow_feedback=False)

with tab_check:
    upload2 = st.file_uploader("Image to check", type=["jpg", "jpeg", "png", "webp"], key="img_check")
    caption = st.text_area("Caption that accompanies it")
    if upload2 and caption and st.button("Check consistency"):
        data = call("/consistency", files={"file": upload2.getvalue()}, data={"caption": caption})
        if data:
            c = data["consistency"]
            st.markdown(f"**{BADGES.get(c['status'], c['status'])}** — cosine {c['similarity']:.3f} (threshold {c['threshold']})")
            st.info(c["note"])
            render_reuse(data["reuse_matches"])