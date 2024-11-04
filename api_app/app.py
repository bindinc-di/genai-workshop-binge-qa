from typing import List
import json
import os
from dotenv import load_dotenv, find_dotenv
# import vertexai # uncomment when you want to extend functionality and leverage GCP vertexai services
# from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.vertexai import VertexAIEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, Response
from fastapi.requests import Request
_ = load_dotenv(find_dotenv())

FAISS_PATH = "data/index"
CHROMA_PATH = "data/chromadb"

# One should use the same embeding function as for document ingestion into the vectorstore
# VERTEX_EMBEDDING_MODEL_NAME = "textembedding-gecko-multilingual@001"
VERTEX_EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"


class QueryInput(BaseModel):
    query: str

class SimilarityRecords(BaseModel):
    chunk: str
    metadata: dict

class SimilarityOutput(List[SimilarityRecords]):
    pass


app = FastAPI()

embeddings = VertexAIEmbeddings(
    model_name=VERTEX_EMBEDDING_MODEL_NAME,
    chunk_size=1
)

# store = FAISS.load_local(FAISS_PATH, embeddings, "index", allow_dangerous_deserialization=True)
store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name="popcorn")


@app.get('/')
async def home():
        return HTMLResponse(
        """<html>
<head>Vectorsearch API</head>
<body>
<h1>The API Is Up and Running!</h1>
<a href="/docs">API Documentation</a>
</body>
</html>
"""
    )


def is_authorized(request:Request):

    if os.getenv("API_DEBUG_MODE", "false").lower() == "true":
        return True

    api_key: str = os.getenv("SEARCH_API_KEY", "DUMMY_KEY")
    auth: str = request.headers.get("Authorization", "Bearer DUMMY_KEY")
    return api_key == auth.split()[-1]


@app.post('/similarity')
async def get_similarity(query_input: QueryInput):

    # if not is_authorized(request):
    #     return Response("Unable to authorize", 401)

    result = await store.asimilarity_search(query_input.query)

    return SimilarityOutput([{"chunk": r.page_content, "metadata": r.metadata}
              for r in result])

# Run locally:
# uvicorn app:app --reload

# Test:
# curl --header "Content-Type: application/json" --data '{"query": "Horror Series op Netflix"}' http://127.0.0.1:5000/similarity