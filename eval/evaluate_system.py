import argparse
import csv
import json
import os
import re
import string
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from generator_groq import generate_answer
from guard_laya import check_guard
from judge_groq import check_llm_judge
from rag_pipeline import run_rag_pipeline
from retriever_pinecone import retrieve


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_answer_match(prediction: str, gold_answers: List[str]) -> bool:
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return False
    pred_tokens = set(norm_pred.split())
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if not norm_gold:
            continue
        if norm_gold in norm_pred:
            return True
        gold_tokens = set(norm_gold.split())
        overlap = pred_tokens.intersection(gold_tokens)
        if len(gold_tokens) > 0 and len(overlap) / len(gold_tokens) >= 0.5:
            return True
    return False


def is_refusal_answer(action: str, answer_text: str) -> bool:
    if action.startswith("refused"):
        return True
    norm = answer_text.lower()
    refusal_cues = [
        "not contain enough information",
        "does not provide this information",
        "does not provide information",
        "cannot be determined",
        "i do not know",
        "i don't know",
        "not mentioned in the context",
        "the provided context does not",
        "no information provided"
    ]
    return any(cue in norm for cue in refusal_cues)


def build_split_sets(all_questions_file: str, sample_size: int = 30):
    if not os.path.exists(all_questions_file):
        sys.exit(f"Error: {all_questions_file} not found. Run eval/build_eval_set.py first.")

    ans_items = []
    unans_items = []
    with open(all_questions_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["is_answerable"]:
                ans_items.append(item)
            else:
                unans_items.append(item)

    half_sample = sample_size // 2
    tuning_items = ans_items[:half_sample] + unans_items[:half_sample]
    held_out_items = ans_items[half_sample:half_sample * 2] + unans_items[half_sample:half_sample * 2]

    os.makedirs("eval", exist_ok=True)
    with open("eval/tuning_set.jsonl", "w", encoding="utf-8") as f:
        for it in tuning_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open("eval/held_out_set.jsonl", "w", encoding="utf-8") as f:
        for it in held_out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Created eval splits: Tuning={len(tuning_items)} ({half_sample} ans, {half_sample} unans), "
          f"Held-out={len(held_out_items)} ({half_sample} ans, {half_sample} unans)")
    return tuning_items, held_out_items


def run_tuning_phase(tuning_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n--- Running Tuning Phase (Guard Disabled, Recording Raw Laya Scores) ---")
    tuning_records = []

    for item in tqdm(tuning_items, desc="Tuning Evaluation"):
        q = item["question"]
        t0 = time.perf_counter()
        passages = retrieve(q, k=4)
        drafted_answer = generate_answer(q, passages, prompt_variant="strict")
        scores = check_guard(q, passages, drafted_answer)
        elapsed = time.perf_counter() - t0

        tuning_records.append({
            "question_id": item["question_id"],
            "question": q,
            "is_answerable": item["is_answerable"],
            "gold_answers": item["gold_answers"],
            "passages": passages,
            "drafted_answer": drafted_answer,
            "p_supported": scores["p_supported"],
            "p_context_sufficient": scores["p_context_sufficient"],
            "time_taken": round(elapsed, 4)
        })

    with open("eval/tuning_raw_scores.json", "w", encoding="utf-8") as f:
        json.dump(tuning_records, f, indent=2)

    with open("eval/tuning_raw_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "is_answerable", "p_supported", "p_context_sufficient", "time_taken", "question", "drafted_answer"]
        )
        writer.writeheader()
        for r in tuning_records:
            writer.writerow({
                "question_id": r["question_id"],
                "is_answerable": r["is_answerable"],
                "p_supported": r["p_supported"],
                "p_context_sufficient": r["p_context_sufficient"],
                "time_taken": r["time_taken"],
                "question": r["question"],
                "drafted_answer": r["drafted_answer"]
            })

    return tuning_records


def sweep_thresholds(tuning_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n--- Sweeping Thresholds from 0.30 to 0.95 ---")
    unans = [r for r in tuning_records if not r["is_answerable"]]
    ans = [r for r in tuning_records if r["is_answerable"]]

    total_unans = len(unans)
    total_ans = len(ans)

    steps = [round(0.30 + i * 0.05, 2) for i in range(14)]
    sweep_results = []
    best_setting = None
    best_score = -1.0

    for th in steps:
        unans_refused = 0
        for r in unans:
            is_refused = (r["p_context_sufficient"] < 0.50) or (r["p_supported"] < th)
            if is_refused:
                unans_refused += 1

        ans_delivered = 0
        for r in ans:
            is_delivered = (r["p_context_sufficient"] >= 0.50) and (r["p_supported"] >= th)
            if is_delivered:
                ans_delivered += 1

        unans_refuse_pct = round((unans_refused / total_unans) * 100, 2) if total_unans > 0 else 0.0
        ans_deliver_pct = round((ans_delivered / total_ans) * 100, 2) if total_ans > 0 else 0.0

        composite = round(0.5 * unans_refuse_pct + 0.5 * ans_deliver_pct, 2)

        sweep_results.append({
            "threshold": th,
            "deliver_threshold": th,
            "retry_threshold": round(max(0.30, th - 0.25), 2),
            "context_sufficient_threshold": 0.50,
            "unanswerable_refused_pct": unans_refuse_pct,
            "answerable_delivered_pct": ans_deliver_pct,
            "composite_score": composite
        })

        if ans_deliver_pct >= 60.0 and composite > best_score:
            best_score = composite
            best_setting = sweep_results[-1]

    if not best_setting:
        best_setting = sweep_results[len(sweep_results) // 2]

    print(f"Optimal Threshold Selected: deliver_threshold={best_setting['deliver_threshold']}, "
          f"retry_threshold={best_setting['retry_threshold']}, context_threshold={best_setting['context_sufficient_threshold']}")
    print(f"  Unanswerable Refused: {best_setting['unanswerable_refused_pct']}% | Answerable Delivered: {best_setting['answerable_delivered_pct']}%")

    with open("eval/threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    with open("eval/threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "deliver_threshold", "retry_threshold", "context_sufficient_threshold", "unanswerable_refused_pct", "answerable_delivered_pct", "composite_score"]
        )
        writer.writeheader()
        writer.writerows(sweep_results)

    with open("eval/calibrated_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best_setting, f, indent=2)

    return best_setting


def run_benchmark_on_held_out(held_out_items: List[Dict[str, Any]], thresholds: Dict[str, Any]):
    print("\n--- Running Benchmark on Held-Out Split (3 Configurations x 2 Prompt Variants) ---")

    configs = [
        {"name": "no_guard", "use_guard": False, "guard_type": "laya"},
        {"name": "laya_guard", "use_guard": True, "guard_type": "laya"},
        {"name": "llm_judge", "use_guard": True, "guard_type": "llm_judge"}
    ]
    prompt_variants = ["strict", "plain"]

    benchmark_records = {}
    summary_rows = []

    for p_var in prompt_variants:
        for cfg in configs:
            run_key = f"{cfg['name']}_{p_var}"
            print(f"\nEvaluating: Configuration='{cfg['name']}', Prompt Variant='{p_var}'")

            details = []
            wrong_answers_delivered = 0
            correct_answers_wrongly_refused = 0
            total_latencies = []

            for item in tqdm(held_out_items, desc=f"{run_key}"):
                q = item["question"]
                is_ans = item["is_answerable"]
                gold_answers = item["gold_answers"]

                result = run_rag_pipeline(
                    question=q,
                    use_guard=cfg["use_guard"],
                    guard_type=cfg["guard_type"],
                    prompt_variant=p_var,
                    deliver_threshold=thresholds["deliver_threshold"],
                    retry_threshold=thresholds["retry_threshold"],
                    context_sufficient_threshold=thresholds["context_sufficient_threshold"]
                )

                ans_text = result["answer"]
                action = result["action"]
                latency = result["time_taken"]
                total_latencies.append(latency)

                is_refused = is_refusal_answer(action, ans_text)

                is_wrong_delivered = False
                is_correct_refused = False

                if not is_ans:
                    if not is_refused:
                        is_wrong_delivered = True
                        wrong_answers_delivered += 1
                else:
                    if is_refused:
                        is_correct_refused = True
                        correct_answers_wrongly_refused += 1
                    else:
                        matches = is_answer_match(ans_text, gold_answers)
                        if not matches:
                            is_wrong_delivered = True
                            wrong_answers_delivered += 1

                details.append({
                    "question_id": item["question_id"],
                    "question": q,
                    "is_answerable": is_ans,
                    "gold_answers": gold_answers,
                    "delivered_answer": ans_text,
                    "action": action,
                    "is_refused": is_refused,
                    "is_wrong_delivered": is_wrong_delivered,
                    "is_correct_refused": is_correct_refused,
                    "scores": result["scores"],
                    "time_taken": latency
                })

            n = len(held_out_items)
            avg_lat = round(sum(total_latencies) / n, 4) if n > 0 else 0.0

            benchmark_records[run_key] = {
                "config": cfg["name"],
                "prompt_variant": p_var,
                "total_queries": n,
                "wrong_answers_delivered": wrong_answers_delivered,
                "wrong_answers_delivered_pct": round((wrong_answers_delivered / n) * 100, 2),
                "correct_answers_wrongly_refused": correct_answers_wrongly_refused,
                "correct_answers_wrongly_refused_pct": round((correct_answers_wrongly_refused / n) * 100, 2),
                "avg_latency_sec": avg_lat,
                "details": details
            }

            summary_rows.append({
                "run_key": run_key,
                "config": cfg["name"],
                "prompt_variant": p_var,
                "total_queries": n,
                "wrong_answers_delivered": wrong_answers_delivered,
                "wrong_answers_delivered_pct": round((wrong_answers_delivered / n) * 100, 2),
                "correct_answers_wrongly_refused": correct_answers_wrongly_refused,
                "correct_answers_wrongly_refused_pct": round((correct_answers_wrongly_refused / n) * 100, 2),
                "avg_latency_sec": avg_lat
            })

    with open("eval/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    with open("eval/benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_key", "config", "prompt_variant", "total_queries",
                "wrong_answers_delivered", "wrong_answers_delivered_pct",
                "correct_answers_wrongly_refused", "correct_answers_wrongly_refused_pct",
                "avg_latency_sec"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 95)
    print(f"{'Run':<22} | {'Config':<12} | {'Prompt':<8} | {'Wrong Delivered':<16} | {'Wrongly Refused':<16} | {'Avg Latency'}")
    print("=" * 95)
    for s in summary_rows:
        wrong_str = f"{s['wrong_answers_delivered']} ({s['wrong_answers_delivered_pct']}%)"
        ref_str = f"{s['correct_answers_wrongly_refused']} ({s['correct_answers_wrongly_refused_pct']}%)"
        print(f"{s['run_key']:<22} | {s['config']:<12} | {s['prompt_variant']:<8} | {wrong_str:<16} | {ref_str:<16} | {s['avg_latency_sec']:.2f}s")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Full RAG Hallucination Guard Evaluation")
    parser.add_argument("--questions-file", default="eval/questions.jsonl")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    if args.benchmark_only and os.path.exists("eval/held_out_set.jsonl") and os.path.exists("eval/calibrated_thresholds.json"):
        with open("eval/calibrated_thresholds.json", "r", encoding="utf-8") as f:
            calibrated_thresholds = json.load(f)
        held_out_items = []
        with open("eval/held_out_set.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                held_out_items.append(json.loads(line))
        if args.sample_size and len(held_out_items) > args.sample_size:
            half = args.sample_size // 2
            ans = [it for it in held_out_items if it["is_answerable"]][:half]
            unans = [it for it in held_out_items if not it["is_answerable"]][:half]
            held_out_items = ans + unans
        run_benchmark_on_held_out(held_out_items, calibrated_thresholds)
    else:
        tuning_items, held_out_items = build_split_sets(args.questions_file, sample_size=args.sample_size)
        tuning_records = run_tuning_phase(tuning_items)
        calibrated_thresholds = sweep_thresholds(tuning_records)
        run_benchmark_on_held_out(held_out_items, calibrated_thresholds)
    print("\nEvaluation completed successfully. All artifacts saved in eval/ directory.")


if __name__ == "__main__":
    main()
