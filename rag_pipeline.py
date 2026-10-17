import time
from typing import Any, Dict, List, Optional
from generator_groq import generate_answer
from guard_laya import check_guard
from retriever_pinecone import retrieve

STANDARD_REFUSAL_MESSAGE = (
    "I'm sorry, but the provided context does not contain enough information to answer this question."
)


def run_rag_pipeline(
    question: str,
    use_guard: bool = True,
    k: int = 4,
    retry_k: int = 8,
    deliver_threshold: float = 0.80,
    retry_threshold: float = 0.50,
    context_sufficient_threshold: float = 0.50,
    refusal_message: str = STANDARD_REFUSAL_MESSAGE
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    passages = retrieve(question, k=k)
    drafted_answer = generate_answer(question, passages)

    if not use_guard:
        return {
            "answer": drafted_answer,
            "action": "baseline_delivered",
            "scores": None,
            "passages": passages,
            "time_taken": round(time.perf_counter() - t0, 4)
        }

    scores = check_guard(question, passages, drafted_answer)
    p_supported = scores["p_supported"]
    p_sufficient = scores["p_context_sufficient"]

    if p_sufficient < context_sufficient_threshold:
        return {
            "answer": refusal_message,
            "action": "refused_insufficient_context",
            "scores": scores,
            "passages": passages,
            "time_taken": round(time.perf_counter() - t0, 4)
        }

    if p_supported >= deliver_threshold:
        return {
            "answer": drafted_answer,
            "action": "delivered",
            "scores": scores,
            "passages": passages,
            "time_taken": round(time.perf_counter() - t0, 4)
        }

    if retry_threshold <= p_supported < deliver_threshold:
        passages_retry = retrieve(question, k=retry_k)
        drafted_answer_retry = generate_answer(question, passages_retry)
        scores_retry = check_guard(question, passages_retry, drafted_answer_retry)

        if scores_retry["p_context_sufficient"] < context_sufficient_threshold:
            action = "refused_insufficient_context_after_retry"
            final_answer = refusal_message
        elif scores_retry["p_supported"] >= deliver_threshold:
            action = "delivered_after_retry"
            final_answer = drafted_answer_retry
        else:
            action = "refused_unsupported_after_retry"
            final_answer = refusal_message

        return {
            "answer": final_answer,
            "action": action,
            "scores": scores_retry,
            "passages": passages_retry,
            "time_taken": round(time.perf_counter() - t0, 4)
        }

    return {
        "answer": refusal_message,
        "action": "refused_unsupported",
        "scores": scores,
        "passages": passages,
        "time_taken": round(time.perf_counter() - t0, 4)
    }


if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "When was the University of Notre Dame founded?"

    print("--- Running with Guard ENABLED ---")
    res_guard = run_rag_pipeline(q, use_guard=True)
    print(json.dumps(res_guard, indent=2))

    print("\n--- Running with Guard DISABLED (Baseline) ---")
    res_base = run_rag_pipeline(q, use_guard=False)
    print(json.dumps(res_base, indent=2))
