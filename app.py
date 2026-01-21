from fastapi import FastAPI
import chromadb
import ollama
import uuid
import os
app = FastAPI()

# 1. Initialize the Ollama Client to point to your host machine
# 'host.docker.internal' allows the container to talk to your computer
is_docker = os.path.exists('/.dockerenv')

if is_docker:
    # Docker mode
    ollama_host = 'http://host.docker.internal:11434'
else:
    # Local machine mode
    ollama_host = 'http://localhost:11434'
ollama_client = ollama.Client(host=ollama_host)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # 2. Use the 'ollama_client' instead of the global 'ollama'
    answer = ollama_client.generate(
        model="tinyllama",
        prompt=f"Context: \n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely: "
    )
    return {"answer": answer["response"]}

@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id])
        
        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }