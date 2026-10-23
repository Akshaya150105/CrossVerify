import sqlite3
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "processed" / "visualnews_30k.csv"
DB_PATH = BASE_DIR / "data" / "processed" / "crossverify.db"


# --------------------------------------------------
# Validate input
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )


# --------------------------------------------------
# Load 30K dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Records loaded: {len(df):,}")
print("Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# Validate required columns
# --------------------------------------------------

required_columns = {
    "id",
    "caption",
    "topic",
    "source",
    "image_path",
    "article_path",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# --------------------------------------------------
# Create database directory
# --------------------------------------------------

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Connect to SQLite
# --------------------------------------------------

print(f"\nCreating database:")
print(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


# --------------------------------------------------
# Drop old tables
# --------------------------------------------------

cursor.execute("DROP TABLE IF EXISTS images")
cursor.execute("DROP TABLE IF EXISTS articles")


# --------------------------------------------------
# Articles table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    source TEXT,
    topic TEXT,
    article_path TEXT
)
""")


# --------------------------------------------------
# Images table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    filepath TEXT,
    caption TEXT,
    FOREIGN KEY (article_id)
        REFERENCES articles(id)
)
""")


# --------------------------------------------------
# Insert articles
# --------------------------------------------------

articles = df[
    [
        "id",
        "source",
        "topic",
        "article_path"
    ]
].drop_duplicates(subset=["id"])


cursor.executemany(
    """
    INSERT INTO articles (
        id,
        source,
        topic,
        article_path
    )
    VALUES (?, ?, ?, ?)
    """,
    articles.itertuples(index=False, name=None)
)


# --------------------------------------------------
# Insert images
# --------------------------------------------------

images = df[
    [
        "id",
        "image_path",
        "caption"
    ]
].copy()

images.rename(
    columns={
        "id": "article_id",
        "image_path": "filepath"
    },
    inplace=True
)

cursor.executemany(
    """
    INSERT INTO images (
        article_id,
        filepath,
        caption
    )
    VALUES (?, ?, ?)
    """,
    images.itertuples(index=False, name=None)
)


# --------------------------------------------------
# Indexes
# --------------------------------------------------

cursor.execute("""
CREATE INDEX idx_articles_source
ON articles(source)
""")

cursor.execute("""
CREATE INDEX idx_articles_topic
ON articles(topic)
""")

cursor.execute("""
CREATE INDEX idx_images_article
ON images(article_id)
""")


# --------------------------------------------------
# Commit
# --------------------------------------------------

conn.commit()


# --------------------------------------------------
# Verification
# --------------------------------------------------

article_count = cursor.execute(
    "SELECT COUNT(*) FROM articles"
).fetchone()[0]

image_count = cursor.execute(
    "SELECT COUNT(*) FROM images"
).fetchone()[0]

source_count = cursor.execute(
    "SELECT COUNT(DISTINCT source) FROM articles"
).fetchone()[0]

topic_count = cursor.execute(
    "SELECT COUNT(DISTINCT topic) FROM articles"
).fetchone()[0]


print("\n" + "=" * 50)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 50)

print(f"Articles : {article_count:,}")
print(f"Images   : {image_count:,}")
print(f"Sources  : {source_count}")
print(f"Topics   : {topic_count}")

print(f"\nDatabase:")
print(DB_PATH)


# --------------------------------------------------
# Sample records
# --------------------------------------------------

print("\nSample article:")

row = cursor.execute("""
SELECT *
FROM articles
LIMIT 1
""").fetchone()

print(row)


print("\nSample image:")

row = cursor.execute("""
SELECT *
FROM images
LIMIT 1
""").fetchone()

print(row)


conn.close()

print("\nDone.")