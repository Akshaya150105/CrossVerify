import re
import sqlite3
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 retrieval over the article headline and body stored in the
    live CrossVerify SQLite database.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)

        self.article_ids = []
        self.documents = []
        self.tokenized_documents = []

        self.bm25 = None

    # --------------------------------------------------
    # Text preprocessing
    # --------------------------------------------------

    @staticmethod
    def clean_text(text):
        if not text:
            return ""

        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text
        )

        # Remove non-alphanumeric characters
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    @classmethod
    def tokenize(cls, text):
        text = cls.clean_text(text)

        return [
            token
            for token in text.split()
            if len(token) > 1
        ]

    # --------------------------------------------------
    # Load articles from SQLite
    # --------------------------------------------------

    def load_documents(self):
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        rows = cursor.execute("""
            SELECT id, headline, body
            FROM articles
            ORDER BY id
        """).fetchall()

        conn.close()

        print(f"Articles in database: {len(rows):,}")

        for article_id, headline, body in rows:
            text = " ".join(part for part in (headline, body) if part)

            if not text.strip():
                continue

            self.article_ids.append(article_id)
            self.documents.append(text)
            self.tokenized_documents.append(
                self.tokenize(text)
            )

        print(
            f"Documents loaded: {len(self.documents):,}"
        )

    # --------------------------------------------------
    # Build BM25
    # --------------------------------------------------

    def build_index(self):

        if not self.tokenized_documents:
            raise RuntimeError(
                "No documents available for indexing."
            )

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        print("BM25 index ready.")

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(self, query, top_k=10):

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "article_id": self.article_ids[index],
                "score": float(scores[index]),
                "document": self.documents[index]
            })

        return results
