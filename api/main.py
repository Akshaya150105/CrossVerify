"""FastAPI service for CrossVerify.

Run from the project root:
    poetry run uvicorn api.main:app --reload

Every search is cross-encoder re-ranked; this is not a caller-configurable option.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from analysis.event_clustering import load_or_build_events
from analysis.reuse_graph import load_or_build_graph
from common import resolve_image_path
from consistency.reuse import DEFAULT_MAX_DISTANCE
from retrieval.pipeline import DEFAULT_DB, DEFAULT_INDEX, MODES, SearchPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipe"] = SearchPipeline(
        os.getenv("CROSSVERIFY_DB", DEFAULT_DB),
        os.getenv("CROSSVERIFY_INDEX", DEFAULT_INDEX),
    )
    yield
    state.clear()


app = FastAPI(title="CrossVerify", version="0.1.0", lifespan=lifespan)


class Feedback(BaseModel):
    relevant: list[int] = Field(default_factory=list)
    irrelevant: list[int] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    mode: str = "rrf"
    top_k: int = Field(10, ge=1, le=50)
    expand: bool = False
    feedback: Feedback | None = None


def public(doc: dict) -> dict:
    """Trim the full article body to a snippet for API responses."""

    out = {k: v for k, v in doc.items() if k != "body"}
    body = doc.get("body") or ""
    out["snippet"] = body[:300] + ("..." if len(body) > 300 else "")
    out["image_url_local"] = f"/image/{doc['article_id']}"
    return out


def read_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read()))
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    results = state["pipe"].search(
        req.query,
        mode=req.mode,
        top_k=req.top_k,
        expand=req.expand,
        feedback=req.feedback.model_dump() if req.feedback else None,
    )
    return {"query": req.query, "results": [public(r) for r in results]}


@app.post("/search/image")
def search_image(file: UploadFile = File(...), top_k: int = Form(10)):
    results = state["pipe"].search_by_image(read_image(file), top_k=top_k)
    return {"results": [public(r) for r in results]}


@app.post("/consistency")
def consistency(file: UploadFile = File(...), caption: str = Form(...)):
    """Score an uploaded image against a caption, plus any pHash reuse in the archive."""

    import imagehash

    from consistency.reuse import matches_for

    pipe = state["pipe"]
    image = read_image(file)
    verdict = pipe.checker.check_image(image, caption)
    import sqlite3

    with sqlite3.connect(pipe.db_path) as conn:
        reuse = matches_for(conn, str(imagehash.phash(image)))
    return {"consistency": verdict, "reuse_matches": reuse}


@app.get("/events")
def events(min_size: int = 2, distance_threshold: float = 0.15, max_days_gap: int = 10):
    """Articles clustered into story timelines (cached by analysis/event_clustering.py
    for a large corpus; query params are only honored when there is no cache file,
    since a precomputed cache was already built with its own parameters)."""

    pipe = state["pipe"]
    cache_path = Path(pipe.db_path).parent / "event_clusters.json"
    clusters = load_or_build_events(
        pipe.db_path,
        pipe.clip.encode_text,
        cache_path=cache_path,
        min_size=min_size,
        distance_threshold=distance_threshold,
        max_days_gap=max_days_gap,
    )
    for event in clusters:
        for a in event["articles"]:
            a["image_url_local"] = f"/image/{a['article_id']}"
    return {"events": clusters}


@app.get("/reuse-graph")
def reuse_graph(max_distance: int = DEFAULT_MAX_DISTANCE):
    """Articles connected by shared/near-duplicate images (cached by
    analysis/reuse_graph.py for a large corpus, since it's O(n^2))."""

    pipe = state["pipe"]
    cache_path = Path(pipe.db_path).parent / "reuse_graph.json"
    return load_or_build_graph(pipe.db_path, cache_path=cache_path, max_distance=max_distance)


@app.get("/article/{article_id}")
def article(article_id: int):
    rec = state["pipe"].fetch_records([article_id]).get(article_id)
    if rec is None:
        raise HTTPException(404, "Unknown article")
    return public(rec)


@app.get("/image/{article_id}")
def image(article_id: int):
    rec = state["pipe"].fetch_records([article_id]).get(article_id)
    if rec is None or not rec.get("filepath"):
        raise HTTPException(404, "No image for this article")
    path = resolve_image_path(rec["filepath"])
    if not path.is_file():
        raise HTTPException(404, "Image file missing on disk")
    return FileResponse(path)