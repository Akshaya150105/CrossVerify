from clip_index import CLIPIndexer


indexer = CLIPIndexer(
    db_path="../data/processed/crossverify.db"
)


texts = [
    "a politician giving a speech",
    "a football match",
    "a courtroom",
    "a natural disaster"
]


embeddings = indexer.encode_text(texts)


print("\nCLIP TEST")
print("=" * 50)

print(
    "Number of texts:",
    len(texts)
)

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "First embedding:",
    embeddings[0][:10]
)

print("\nCLIP is working.")