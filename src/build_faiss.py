from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# load documents
documents = open("data/documents.txt", "r", encoding="utf-8").read().split("\n")

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# create embeddings
embeddings = model.encode(documents)

# convert to float32
embeddings = np.array(embeddings).astype("float32")

# create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# add embeddings
index.add(embeddings)

# save index
faiss.write_index(index, "data/faiss.index")

print("FAISS index created!")