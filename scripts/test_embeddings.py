from app.rag.embeddings import EmbeddingModel


model = EmbeddingModel()


texts = [
    "I will talk to you of art.",
    "The artist is creating a new painting.",
    "The car is driving down the road.",
]


embeddings = model.embed_texts(texts)


print(f"Number of texts: {len(texts)}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")

print("\nFirst embedding:")
print(embeddings[0][:10])


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert len(embeddings) == len(texts)

assert all(
    len(embedding) == 384
    for embedding in embeddings
), "Unexpected embedding dimension."

assert all(
    isinstance(value, float)
    for embedding in embeddings
    for value in embedding
), "Embedding contains non-float values."

query_embedding = model.embed_query(
    "What does the artist say about art?"
)

assert len(query_embedding) == 384

print("\n" + "=" * 70)
print("ALL EMBEDDING TESTS PASSED")
print("=" * 70)