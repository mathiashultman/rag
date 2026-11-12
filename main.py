import trafilatura
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from ollama import chat
from fastapi import FastAPI, Query

app = FastAPI()

with open("mine.html", encoding="utf-8") as f:
    html = f.read()

text = trafilatura.extract(html)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    separators=[
        "\n\n",   # stycken
        "\n",     # radbrytningar
        ". ",     # meningar
        " ",
        ""
    ]
)

chunks = splitter.split_text(text)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer(
    "BAAI/bge-m3",
    device=device
)

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

client = chromadb.PersistentClient(path="./chroma")

collection = client.get_or_create_collection("docs")

collection.add(
    ids=[str(i) for i in range(1,6)],
    documents=chunks,
    embeddings=embeddings.tolist()
)

@app.get("/query")
def query(query: str = Query(..., min_length=1)):
    #query = "Vem är VD?"
    print(query)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    context=results["documents"][0]

    augmented_prompt=f"""
    givet kontexten nedan, svara på frågan.

    Fråga: {query}
    Kontext: {context}
    Kom ihåg att bara svara utifrån kontexten som är given. Om frågan inte kan besvaras utifrån kontexten, svara "jag vet inte".
    """

    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )

    return {
        "query": query,
        "reponse": response["message"]["content"],
        "message": "Query received successfully"
    }

