"""FastAPI service for CrossVerify.

Run from the project root:
    poetry run uvicorn api.main:app --reload
Set CROSSVERIFY_RERANK=1 to enable the cross-encoder by default.
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

from common import resolve_image_path
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
    rerank: bool = os.getenv("CROSSVERIFY_RERANK") == "1"
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
        rerank=req.rerank,
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