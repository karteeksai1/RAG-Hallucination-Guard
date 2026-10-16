import argparse
import hashlib
import os
import sys
import time
from dotenv import load_dotenv
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest SQuAD 2.0 passages into Pinecone")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "squad-rag")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    if not api_key:
        sys.exit("Error: PINECONE_API_KEY is not set in the environment or .env file.")

    try:
        pc = Pinecone(api_key=api_key)
        index_list = pc.list_indexes()
        if hasattr(index_list, "names"):
            existing_indexes = index_list.names()
        else:
            existing_indexes = [
                item["name"] if isinstance(item, dict) else item.name
                for item in index_list
            ]

        if index_name not in existing_indexes:
            print(f"Creating serverless index '{index_name}' in {cloud}:{region}...")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Waiting for index '{index_name}' to be ready...")
            while True:
                desc = pc.describe_index(index_name)
                status = getattr(desc, "status", None)
                ready = False
                if isinstance(status, dict):
                    ready = status.get("ready", False)
                elif status is not None:
                    ready = getattr(status, "ready", False)
                if ready:
                    break
                time.sleep(2)
            print(f"Index '{index_name}' is ready.")
        else:
            print(f"Using existing index '{index_name}'.")

        index = pc.Index(index_name)
    except Exception as e:
        sys.exit(f"Error connecting to Pinecone: {e}")

    try:
        print(f"Loading SQuAD 2.0 dataset (split: '{args.split}')...")
        dataset = load_dataset("rajpurkar/squad_v2", split=args.split)
    except Exception as e:
        sys.exit(f"Error loading dataset: {e}")

    unique_passages = {}
    for row in dataset:
        ctx = row["context"]
        ctx_id = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
        if ctx_id not in unique_passages:
            unique_passages[ctx_id] = {
                "id": ctx_id,
                "text": ctx,
                "title": row.get("title", "")
            }
            if args.limit is not None and len(unique_passages) >= args.limit:
                break

    passages = list(unique_passages.values())
    print(f"Collected {len(passages)} unique passages for ingestion.")

    try:
        print("Loading embedding model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [p["text"] for p in passages]
        print("Encoding passages...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    except Exception as e:
        sys.exit(f"Error generating embeddings: {e}")

    vectors = []
    for p, emb in zip(passages, embeddings):
        vectors.append({
            "id": p["id"],
            "values": emb.tolist(),
            "metadata": {
                "text": p["text"],
                "title": p["title"],
                "source": "squad_v2"
            }
        })

    try:
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting passages"):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch)
    except Exception as e:
        sys.exit(f"Error upserting vectors to Pinecone: {e}")

    try:
        stats = index.describe_index_stats()
        total_count = getattr(stats, "total_vector_count", None)
        if total_count is None and isinstance(stats, dict):
            total_count = stats.get("total_vector_count")
        print("Ingestion completed successfully.")
        print(f"Passages ingested: {len(vectors)}")
        print(f"Total vector count in index: {total_count}")
    except Exception as e:
        sys.exit(f"Error fetching index stats: {e}")


if __name__ == "__main__":
    main()
