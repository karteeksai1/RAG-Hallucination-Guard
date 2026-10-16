import argparse
import hashlib
import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build SQuAD 2.0 evaluation set")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="eval/questions.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        print(f"Loading SQuAD 2.0 dataset (split: '{args.split}')...")
        dataset = load_dataset("rajpurkar/squad_v2", split=args.split)
    except Exception as e:
        sys.exit(f"Error loading dataset: {e}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in tqdm(dataset, desc="Writing questions"):
                gold_answers = row["answers"]["text"]
                context_id = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
                item = {
                    "question_id": row["id"],
                    "question": row["question"],
                    "gold_answers": gold_answers,
                    "is_answerable": len(gold_answers) > 0,
                    "context_id": context_id
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if args.limit is not None and count >= args.limit:
                    break

        print(f"Successfully saved {count} evaluation questions to {args.output}")
    except Exception as e:
        sys.exit(f"Error writing evaluation set: {e}")


if __name__ == "__main__":
    main()
