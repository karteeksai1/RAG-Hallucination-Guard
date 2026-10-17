import json
import os
import re
from typing import Dict, List, Union
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
_default_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None


def _get_client() -> Groq:
    global _client
    if not _api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    if _client is None:
        _client = Groq(api_key=_api_key)
    return _client


def check_llm_judge(
    question: str,
    passages: Union[str, List[str]],
    drafted_answer: str,
    model: str = None
) -> Dict[str, float]:
    client = _get_client()
    selected_model = model or _default_model

    if isinstance(passages, list):
        joined_context = "\n\n".join(passages)
    else:
        joined_context = passages

    system_prompt = (
        "You are an expert evaluator assessing factual groundedness in RAG systems. "
        "Evaluate the candidate answer against the context and question on two binary criteria:\n"
        "1. 'supported': true if all statements in the answer are supported by the context, false otherwise.\n"
        "2. 'sufficient': true if the context contains enough information to answer the question, false otherwise.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"supported": true, "sufficient": true}'
    )

    user_prompt = (
        f"Context:\n{joined_context}\n\n"
        f"Question:\n{question}\n\n"
        f"Drafted Answer:\n{drafted_answer}\n\n"
        "Evaluation JSON:"
    )

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
    except Exception:
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}
        else:
            data = {}

    p_supported = 1.0 if data.get("supported") is True else 0.0
    p_sufficient = 1.0 if data.get("sufficient") is True else 0.0

    return {
        "p_supported": p_supported,
        "p_context_sufficient": p_sufficient
    }
