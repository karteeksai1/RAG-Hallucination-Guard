from typing import Dict, List, Union
from laya import Router

GUARD_QUESTIONS = {
    "supported_by_context": {
        "type": "noul",
        "instructions": "Is every claim in the answer supported by the context?",
        "criteria": {
            "true": "all claims and assertions in the answer are directly supported by the context",
            "false": "the answer contains unsupported claims, extrapolations, or details missing from the context"
        }
    },
    "context_has_info": {
        "type": "noul",
        "instructions": "Does the context contain enough information to answer the question?",
        "criteria": {
            "true": "the context provides sufficient facts and details to answer the question",
            "false": "the context lacks critical information or does not have enough detail to answer the question"
        }
    }
}

_router = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(default="multilingual")
        _router.preload(["multilingual"])
    return _router


def check_guard(
    question: str,
    passages: Union[str, List[str]],
    drafted_answer: str,
    max_len: int = 2048
) -> Dict[str, float]:
    router = get_router()

    if isinstance(passages, list):
        joined_context = "\n\n".join(p.strip() for p in passages if p.strip())
    else:
        joined_context = passages.strip()

    state = {
        "question": question.strip(),
        "context": joined_context,
        "answer": drafted_answer.strip()
    }

    result = router.predict(
        state=state,
        questions=GUARD_QUESTIONS,
        model="multilingual",
        max_len=max_len
    )

    answers = result.get("answers", {})
    p_supported = answers.get("supported_by_context", {}).get("noul", 0.0)
    p_context_sufficient = answers.get("context_has_info", {}).get("noul", 0.0)

    return {
        "p_supported": p_supported,
        "p_context_sufficient": p_context_sufficient
    }
