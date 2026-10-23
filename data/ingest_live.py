#!/usr/bin/env python3
"""
CrossVerify - live news ingestion.

Pipeline per article:
  feed entry -> fetch page -> extract headline/body/date -> find main image + caption
  -> download + resize image -> compute pHash -> write to SQLite.

Run:
  python indexing/ingest_live.py --feeds indexing/feeds.txt \
      --db data/processed/crossverify.db --img-dir data/processed/images --max-per-feed 50

Be polite: it checks robots.txt, rate-limits per request, and identifies itself.
Edit USER_AGENT below with your contact email before running.
"""
import argparse
import hashlib
import io
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import feedparser
import imagehash
import requests
import trafilatura
from bs4 import BeautifulSoup
from PIL import Image

USER_AGENT = "CrossVerifyResearchBot/0.1 (academic project; contact: your-email@example.com)"
MIN_SIDE = 180      # drop tiny images (icons, trackers)
MAX_DIM = 512       # resize longest side to this
TIMEOUT = 20

log = logging.getLogger("ingest")

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT, body TEXT, date TEXT, source TEXT,
    url TEXT UNIQUE, fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS images(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER REFERENCES articles(id),
    filepath TEXT, caption TEXT, date_taken_estimate TEXT,
    phash TEXT, caption_source TEXT, image_url TEXT,
    width INTEGER, height INTEGER
);
CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
"""


# ---------------------------------------------------------------- politeness
class Fetcher:
    def __init__(self, delay: float):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.delay = delay
        self._robots = {}
        self._last = 0.0

    def _wait(self):
        gap = time.time() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.time()

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            try:
                self._wait()
                r = self.s.get(base + "/robots.txt", timeout=TIMEOUT)
                if r.status_code == 404:
                    rp.parse([])                      # no robots.txt -> allowed
                elif r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:
                    rp = None                         # unclear -> be conservative
            except requests.RequestException:
                rp = None
            self._robots[base] = rp
        rp = self._robots[base]
        return bool(rp and rp.can_fetch(USER_AGENT, url))

    def get(self, url: str):
        if not self.allowed(url):
            log.info("robots.txt disallows: %s", url)
            return None
        try:
            self._wait()
            r = self.s.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("fetch failed %s (%s)", url, e)
            return None


# ---------------------------------------------------------------- extraction
def _src_of(img) -> str:
    for attr in ("src", "data-src", "data-original"):
        if img.get(attr):
            return img[attr]
    srcset = img.get("srcset", "")
    return srcset.split(",")[0].strip().split(" ")[0] if srcset else ""


def find_image_and_caption(html: str, page_url: str):
    """Return (image_url, caption, caption_source). caption_source in
    figcaption | alt | og_description | none."""
    soup = BeautifulSoup(html, "lxml")

    og = soup.find("meta", property="og:image")
    og_url = urljoin(page_url, og["content"]) if og and og.get("content") else None
    og_stem = Path(urlparse(og_url).path).stem if og_url else ""

    best_fig = None   # figure whose image matches og:image
    first_fig = None  # fallback: first figure with a figcaption
    for fig in soup.find_all("figure"):
        img, cap = fig.find("img"), fig.find("figcaption")
        if not img:
            continue
        src = urljoin(page_url, _src_of(img)) if _src_of(img) else None
        if not src:
            continue
        if cap and cap.get_text(strip=True) and first_fig is None:
            first_fig = (src, cap.get_text(" ", strip=True))
        if og_stem and og_stem in (src + " " + img.get("srcset", "")):
            best_fig = (src, cap.get_text(" ", strip=True) if cap else "", img.get("alt", ""))
            break

    if best_fig:
        src, cap, alt = best_fig
        image_url = og_url or src
        if cap:
            return image_url, cap, "figcaption"
        if len(alt) > 10:
            return image_url, alt, "alt"
    elif first_fig:
        # og:image didn't match any <figure> (common when sites serve different sizes).
        # Keep the figure's own image + caption so the pair stays consistent, but tag it.
        return first_fig[0], first_fig[1], ("figcaption_unmatched" if og_url else "figcaption")

    image_url = og_url or (first_fig[0] if first_fig else None)
    desc = soup.find("meta", property="og:description")
    if desc and desc.get("content"):
        return image_url, desc["content"].strip(), "og_description"
    return image_url, None, "none"


def _norm(text: str) -> str:
    return " ".join(text.split())


def get_headline(html: str, fallback: str = "") -> str:
    """og:title first (plain attribute, no markup), then <h1> with spaces between
    child elements, then the fallback. Avoids words glued together by markup."""
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", property="og:title")
    if og and og.get("content", "").strip():
        return _norm(og["content"])
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _norm(h1.get_text(" ", strip=True))
    return _norm(fallback)


def extract_article(html: str, url: str):
    body = trafilatura.extract(html, include_comments=False, include_tables=False,
                               favor_precision=True)
    meta = trafilatura.extract_metadata(html)
    if not body or len(body) < 200:
        return None
    return {
        "headline": get_headline(html, meta.title if meta and meta.title else ""),
        "body": body,
        "date": meta.date if meta and meta.date else None,
        "source": urlparse(url).netloc.replace("www.", ""),
    }


# ---------------------------------------------------------------- images
def save_image(content: bytes, url: str, img_dir: Path):
    try:
        im = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return None
    if min(im.size) < MIN_SIDE:
        return None
    ph = str(imagehash.phash(im))             # hash the image before resizing artefacts
    im.thumbnail((MAX_DIM, MAX_DIM))
    name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"
    path = img_dir / name
    im.save(path, "JPEG", quality=90)
    return str(path), ph, im.size[0], im.size[1]


# ---------------------------------------------------------------- main loop
def ingest(feeds, db_path, img_dir, max_per_feed, delay):
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    fetcher = Fetcher(delay)
    stats = dict(seen=0, skipped_existing=0, no_article=0, no_image=0, added=0)

    for feed_url in feeds:
        parsed = feedparser.parse(feed_url, agent=USER_AGENT)
        log.info("feed %s: %d entries", feed_url, len(parsed.entries))
        for entry in parsed.entries[:max_per_feed]:
            url = entry.get("link")
            if not url:
                continue
            stats["seen"] += 1
            if conn.execute("SELECT 1 FROM articles WHERE url=?", (url,)).fetchone():
                stats["skipped_existing"] += 1
                continue

            page = fetcher.get(url)
            if page is None:
                continue
            art = extract_article(page.text, url)
            if art is None:
                stats["no_article"] += 1
                continue
            if not art["date"] and entry.get("published_parsed"):
                art["date"] = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

            image_url, caption, cap_src = find_image_and_caption(page.text, url)
            saved = None
            if image_url:
                r = fetcher.get(image_url)
                if r is not None:
                    saved = save_image(r.content, image_url, img_dir)
            if saved is None:
                stats["no_image"] += 1
                continue

            now = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                "INSERT INTO articles(headline, body, date, source, url, fetched_at) VALUES(?,?,?,?,?,?)",
                (art["headline"], art["body"], art["date"], art["source"], url, now))
            path, ph, w, h = saved
            conn.execute(
                "INSERT INTO images(article_id, filepath, caption, date_taken_estimate, phash,"
                " caption_source, image_url, width, height) VALUES(?,?,?,?,?,?,?,?,?)",
                (cur.lastrowid, path, caption, art["date"], ph, cap_src, image_url, w, h))
            conn.commit()
            stats["added"] += 1
            log.info("added [%s] %s", cap_src, art["headline"][:70])

    reused = conn.execute(
        "SELECT COUNT(*) FROM (SELECT phash FROM images GROUP BY phash HAVING COUNT(DISTINCT article_id) > 1)"
    ).fetchone()[0]
    log.info("done: %s | pHash values shared by >1 article: %d", stats, reused)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", required=True, help="text file, one feed URL per line (# for comments)")
    ap.add_argument("--db", default="data/processed/crossverify.db")
    ap.add_argument("--img-dir", default="data/processed/images")
    ap.add_argument("--max-per-feed", type=int, default=50)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feeds = [l.strip() for l in Path(args.feeds).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    ingest(feeds, args.db, args.img_dir, args.max_per_feed, args.delay)