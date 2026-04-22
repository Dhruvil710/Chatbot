import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("car_knowledge.txt") as f:
    docs = [line.strip() for line in f if line.strip()]

embeddings = model.encode(docs)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

def search_knowledge(query, k=2):
    q = model.encode([query])
    D, I = index.search(np.array(q), k)
    return [docs[i] for i in I[0]]
