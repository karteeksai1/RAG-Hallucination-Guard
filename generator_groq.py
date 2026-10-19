import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
_default_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None


def _get_client():
    global _client
    if not _api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
    if _client is None:
        _client = Groq(api_key=_api_key)
    return _client


def generate_answer(
    question: str,
    passages: list[str],
    model: str = None,
    prompt_variant: str = "strict"
) -> str:
    client = _get_client()
    selected_model = model or _default_model

    formatted_context = ""
    for idx, passage in enumerate(passages, 1):
        formatted_context += f"Passage [{idx}]:\n{passage.strip()}\n\n"

    if prompt_variant == "plain":
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based on the provided "
            "context passages. Provide a direct, confident, and complete answer."
        )
    else:
        system_prompt = (
            "You are a strictly factual assistant. Answer the user's question using ONLY the provided "
            "context passages. If the answer cannot be determined directly from the context, state that "
            "you do not know or that the context does not provide this information. Do not speculate, "
            "extrapolate, or use external knowledge."
        )

    user_prompt = f"Context:\n{formatted_context}\nQuestion: {question.strip()}\n\nAnswer:"

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(2 ** attempt)


def run_tests():
    test_cases = [
        {
            "name": "Fact directly present in context",
            "question": "When was the University of Notre Dame founded, and by whom?",
            "passages": [
                "The University of Notre Dame du Lac is a private Catholic research university in Notre Dame, Indiana, outside the city of South Bend. It was founded on November 26, 1842, by Father Edward Sorin."
            ],
            "expected_behavior": "Should answer November 26, 1842 by Father Edward Sorin."
        },
        {
            "name": "Multiple passages with relevant detail",
            "question": "Who was the lunar module pilot on Apollo 11?",
            "passages": [
                "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
                "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.",
                "Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the surface."
            ],
            "expected_behavior": "Should identify Buzz Aldrin as the lunar module pilot."
        },
        {
            "name": "Unanswerable from context (refusal test)",
            "question": "What is the capital of France?",
            "passages": [
                "The University of Notre Dame du Lac is a private Catholic research university in Notre Dame, Indiana, outside the city of South Bend. It was founded on November 26, 1842, by Father Edward Sorin."
            ],
            "expected_behavior": "Should state that it doesn't know or that the information is not provided in context."
        }
    ]

    print(f"Testing generator with model: {_default_model}\n" + "=" * 60)
    for i, tc in enumerate(test_cases, 1):
        print(f"Test {i}: {tc['name']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected_behavior']}")
        answer = generate_answer(tc["question"], tc["passages"])
        print(f"Generated Answer:\n{answer}")
        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    elif len(sys.argv) > 1:
        user_query = sys.argv[1]
        dummy_context = ["This is a test context passage."]
        print(generate_answer(user_query, dummy_context))
    else:
        run_tests()
