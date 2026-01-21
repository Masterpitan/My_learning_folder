# to embed our texts so they can generate answers from stored embeddings
import chromadb
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")

with open("k8s.txt", "r") as f:
    text = f.read()
collection.add(
    documents=[text],
    ids=["k8s"]
)
print("Embeddings stored in Chroma")