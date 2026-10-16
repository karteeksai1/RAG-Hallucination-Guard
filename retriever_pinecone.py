import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

_api_key = os.getenv("PINECONE_API_KEY")
_index_name = os.getenv("PINECONE_INDEX_NAME", "squad-rag")

_pc = None
_index = None
_model = None


def _init_resources():
    global _pc, _index, _model
    if not _api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set in .env or environment.")
    if _pc is None:
        _pc = Pinecone(api_key=_api_key)
        _index = _pc.Index(_index_name)
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _index, _model


def retrieve(question: str, k: int = 4) -> list:
    index, model = _init_resources()
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    response = index.query(vector=query_vector, top_k=k, include_metadata=True)

    passages = []
    matches = getattr(response, "matches", []) or []
    if isinstance(response, dict):
        matches = response.get("matches", [])

    for match in matches:
        metadata = getattr(match, "metadata", None)
        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata", {})
        if metadata and "text" in metadata:
            passages.append(metadata["text"])

    return passages


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "When was the University of Notre Dame established?"
    try:
        results = retrieve(query)
        print(f"Query: {query}")
        print(f"Retrieved {len(results)} passage(s):\n")
        for i, text in enumerate(results, 1):
            print(f"[{i}]\n{text}\n")
    except Exception as e:
        sys.exit(f"Error during retrieval: {e}")
