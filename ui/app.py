"""Streamlit front end. Start the API first, then:
    poetry run streamlit run ui/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

API = os.getenv("CROSSVERIFY_API", "http://127.0.0.1:8000")
BADGES = {"aligned": "🟢 aligned", "low_alignment": "🟠 low image-caption alignment", "unknown": "⚪ not calibrated"}

st.set_page_config(page_title="CrossVerify", layout="wide")
st.title("CrossVerify")
st.caption(
    "Multi-modal news retrieval with provenance. Every result is cross-encoder "
    "re-ranked. A low alignment score is not proof of misinformation."
)

for key, default in (("results", []), ("relevant", set()), ("irrelevant", set()), ("last_query", ""), ("last_mode", "rrf")):
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Search settings")
    mode = st.selectbox("Retrieval method", ["rrf", "clip", "bm25"], help="rrf = CLIP + BM25 fused (recommended)")
    top_k = st.slider("Results", 3, 20, 8)
    expand = st.checkbox("Query expansion (WordNet)")
    st.caption("Cross-encoder re-ranking is always applied before results are shown.")

tab_text, tab_image, tab_check, tab_timelines, tab_graph = st.tabs(
    ["Text search", "Image search", "Check image + caption", "Story timelines", "Reuse graph"]
)


def call(path: str, **kwargs):
    try:
        response = requests.post(f"{API}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def call_get(path: str, **params):
    try:
        response = requests.get(f"{API}{path}", params=params, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def render_reuse(matches: list[dict]):
    if matches:
        st.warning(f"⚠️ Image reuse: {len(matches)} other article(s) use the same/near-identical image")
        for m in matches:
            st.markdown(f"- #{m['article_id']} **{m['headline']}** ({m['source']}, {m['date']}) — distance {m['distance']}")


def render_provenance(r: dict):
    """Provenance is the primary evidence for a result, so it is always visible, not tucked away."""

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**Publisher:** {r['source']}  \n"
            f"**Article URL:** [{r['url']}]({r['url']})  \n"
            f"**Fetched:** {r.get('fetched_at') or 'unknown'}"
        )
    with right:
        image_url = r.get("image_url")
        image_link = f"[{image_url}]({image_url})" if image_url else "unknown"
        st.markdown(
            f"**Image URL:** {image_link}  \n"
            f"**Caption source:** {r.get('caption_source') or 'unknown'}  \n"
            f"**Image hash (pHash):** `{r.get('phash') or 'unknown'}`"
        )


def render_results(results: list[dict], allow_feedback: bool):
    if not results:
        st.info("No results.")
        return
    for r in results:
        with st.container(border=True):
            img_col, main_col = st.columns([1, 3])
            with img_col:
                st.image(f"{API}{r['image_url_local']}", use_container_width=True)
                cons = r.get("consistency")
                if cons:
                    st.markdown(f"{BADGES.get(cons['status'], cons['status'])}  \ncosine {cons['similarity']:.3f}")
            with main_col:
                st.markdown(f"#### [{r['headline']}]({r['url']})")
                st.caption(f"{r['source']} · {r.get('date') or 'date unknown'} · relevance score {r['score']:.4f}")
                st.write(f"**Caption:** {r.get('caption') or '—'}")

                st.markdown("**Provenance**")
                render_provenance(r)
                render_reuse(r.get("reuse_matches", []))

                with st.expander("Ranking details"):
                    st.json({"ranking": r.get("explanation"), "rerank_score": r.get("rerank_score"),
                             "query_variants": r.get("variants")})

                if allow_feedback:
                    aid = r["article_id"]
                    c1, c2, _ = st.columns([1, 1, 6])
                    if c1.button("👍 Relevant", key=f"up{aid}"):
                        st.session_state.relevant.add(aid)
                        st.session_state.irrelevant.discard(aid)
                        st.rerun()
                    if c2.button("👎 Not relevant", key=f"down{aid}"):
                        st.session_state.irrelevant.add(aid)
                        st.session_state.relevant.discard(aid)
                        st.rerun()
                    if aid in st.session_state.relevant:
                        st.caption("✅ Marked relevant")
                    elif aid in st.session_state.irrelevant:
                        st.caption("🚫 Marked not relevant")


def run_search(query: str, feedback: dict | None = None):
    payload = {"query": query, "mode": mode, "top_k": top_k, "expand": expand, "feedback": feedback}
    data = call("/search", json=payload)
    if data:
        st.session_state.results = data["results"]
        st.session_state.last_query = query
        st.session_state.last_mode = mode


with tab_text:
    query = st.text_input("Describe the news event or image", placeholder="e.g. protesters carrying signs in the rain")
    if st.button("Search", type="primary") and query:
        st.session_state.relevant, st.session_state.irrelevant = set(), set()
        run_search(query)

    if st.session_state.results and st.session_state.last_query:
        n_rel, n_irr = len(st.session_state.relevant), len(st.session_state.irrelevant)
        if n_rel or n_irr:
            if st.session_state.last_mode == "bm25":
                st.caption(
                    "Relevance feedback refines the CLIP similarity signal, so it has no effect "
                    "in BM25-only mode — switch to rrf or clip to use it."
                )
            else:
                if st.button(f"🔁 Refine results with feedback ({n_rel} relevant, {n_irr} not relevant)", type="primary"):
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

with tab_timelines:
    st.caption(
        "Articles clustered by headline similarity, gated so only stories reported "
        "within a few days of each other can merge into one timeline."
    )
    if st.button("Build timelines"):
        data = call_get("/events")
        st.session_state.events = data["events"] if data else []

    events = st.session_state.get("events")
    if events is None:
        st.info("Click \"Build timelines\" to cluster the corpus into story arcs.")
    elif not events:
        st.info("No multi-article story clusters found — try a larger or longer-running corpus.")
    else:
        for event in events:
            label = f"{event['start_date']} → {event['end_date']} · {event['size']} articles · {event['headline']}"
            with st.expander(label):
                for a in event["articles"]:
                    img_col, main_col = st.columns([1, 4])
                    with img_col:
                        st.image(f"{API}{a['image_url_local']}", use_container_width=True)
                    with main_col:
                        st.markdown(f"**{a['date']}** · [{a['headline']}]({a['url']}) — {a['source']}")
                        st.caption(a.get("caption") or "—")
                    st.divider()

with tab_graph:
    st.caption(
        "Articles connected by a shared or near-identical image (pHash match). "
        "Needs a corpus collected over time — a single day's pull rarely has reuse yet."
    )
    if st.button("Build reuse graph"):
        st.session_state.reuse_graph = call_get("/reuse-graph")

    graph = st.session_state.get("reuse_graph")
    if graph is None:
        st.info("Click \"Build reuse graph\" to look for image reuse across the corpus.")
    elif not graph["edges"]:
        st.info("No reused images detected in the current corpus.")
    else:
        sources = sorted({n["source"] for n in graph["nodes"]})
        palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        color_for = {s: palette[i % len(palette)] for i, s in enumerate(sources)}

        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")
        for n in graph["nodes"]:
            net.add_node(
                n["id"],
                label=n["headline"][:40],
                title=f"{n['headline']}\n{n['source']} · {n['date']}",
                color=color_for[n["source"]],
            )
        for e in graph["edges"]:
            net.add_edge(
                e["source"], e["target"],
                title=f"pHash distance {e['distance']}, headline overlap {e['headline_overlap']}",
                value=max(1, 10 - e["distance"]),
            )
        net.repulsion(node_distance=160)
        components.html(net.generate_html(notebook=False), height=620, scrolling=True)
        st.caption(f"{len(graph['nodes'])} article(s), {len(graph['edges'])} reuse edge(s)")
