"""Cross-encoder re-ranking of the fused top-N (guide section 7)."""

from __future__ import annotations

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_chars: int = 1500):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.max_chars = max_chars

    @staticmethod
    def document_text(doc: dict) -> str:
        """Headline + caption + start of body: what the encoder scores against."""

        parts = [doc.get("headline") or "", doc.get("caption") or "", doc.get("body") or ""]
        return " . ".join(p for p in parts if p)

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Return docs sorted by cross-encoder score, adding `rerank_score`."""

        if not docs:
            return []
        pairs = [(query, self.document_text(d)[: self.max_chars]) for d in docs]
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)