import json
import os
import sys
from datasets import load_dataset
from generator_groq import generate_answer
from guard_laya import check_guard


def main():
    print("Loading SQuAD validation dataset for context lookup...")
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    id_to_context = {row["id"]: row["context"] for row in dataset}

    eval_file = "eval/questions.jsonl"
    if not os.path.exists(eval_file):
        sys.exit(f"Error: {eval_file} does not exist. Run eval/build_eval_set.py first.")

    answerable = []
    unanswerable = []

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["is_answerable"] and len(answerable) < 10:
                answerable.append(item)
            elif not item["is_answerable"] and len(unanswerable) < 10:
                unanswerable.append(item)
            if len(answerable) == 10 and len(unanswerable) == 10:
                break

    selected_questions = answerable + unanswerable
    print(f"Selected {len(selected_questions)} questions ({len(answerable)} answerable, {len(unanswerable)} unanswerable).\n")

    results = []
    print("=" * 105)
    print(f"{'#':<3} | {'Type':<12} | {'P(Supported)':<14} | {'P(Sufficient)':<14} | {'Question / Drafted Answer'}")
    print("=" * 105)

    for idx, q_item in enumerate(selected_questions, 1):
        q_id = q_item["question_id"]
        question = q_item["question"]
        is_ans = q_item["is_answerable"]
        q_type = "Answerable" if is_ans else "Unanswerable"
        context = id_to_context.get(q_id, "")

        try:
            drafted_answer = generate_answer(question, [context])
        except Exception as e:
            drafted_answer = f"[Groq Error: {e}]"

        guard_scores = check_guard(question, context, drafted_answer)
        p_sup = guard_scores["p_supported"]
        p_suf = guard_scores["p_context_sufficient"]

        clean_ans = drafted_answer.replace("\n", " ")
        if len(clean_ans) > 75:
            clean_ans = clean_ans[:72] + "..."

        print(f"{idx:<3} | {q_type:<12} | {p_sup:<14.4f} | {p_suf:<14.4f} | Q: {question}")
        print(f"    | {'':<12} | {'':<14} | {'':<14} | A: {clean_ans}")
        print("-" * 105)

        results.append({
            "index": idx,
            "question_id": q_id,
            "type": q_type,
            "question": question,
            "gold_answers": q_item["gold_answers"],
            "drafted_answer": drafted_answer,
            "p_supported": p_sup,
            "p_context_sufficient": p_suf
        })

    print("\nSummary of Raw Probabilities:")
    ans_sup = [r["p_supported"] for r in results if r["type"] == "Answerable"]
    ans_suf = [r["p_context_sufficient"] for r in results if r["type"] == "Answerable"]
    unans_sup = [r["p_supported"] for r in results if r["type"] == "Unanswerable"]
    unans_suf = [r["p_context_sufficient"] for r in results if r["type"] == "Unanswerable"]

    print(f"Answerable Questions (n={len(ans_sup)}):")
    print(f"  P(Supported)  -> Min: {min(ans_sup):.4f}, Max: {max(ans_sup):.4f}, Avg: {sum(ans_sup)/len(ans_sup):.4f}")
    print(f"  P(Sufficient) -> Min: {min(ans_suf):.4f}, Max: {max(ans_suf):.4f}, Avg: {sum(ans_suf)/len(ans_suf):.4f}")

    print(f"Unanswerable Questions (n={len(unans_sup)}):")
    print(f"  P(Supported)  -> Min: {min(unans_sup):.4f}, Max: {max(unans_sup):.4f}, Avg: {sum(unans_sup)/len(unans_sup):.4f}")
    print(f"  P(Sufficient) -> Min: {min(unans_suf):.4f}, Max: {max(unans_suf):.4f}, Avg: {sum(unans_suf)/len(unans_suf):.4f}")


if __name__ == "__main__":
    main()
