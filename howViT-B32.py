from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
# [4, 4]