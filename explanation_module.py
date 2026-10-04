import os
from functools import lru_cache

from gemini_client import generate_text


MODEL_NAME = os.getenv(
    "LOCAL_EXPLANATION_MODEL",
    "MBZUAI/LaMini-Flan-T5-783M"
)


@lru_cache(maxsize=1)
def _get_local_pipeline():

    from transformers import pipeline

    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )


def _local_explanation(topic: str) -> str:

    generator = _get_local_pipeline()

    prompt = f"""
Explain the following educational topic
in simple language for a beginner.

Topic:
{topic}

Use:

1. Simple definition
2. Key points
3. One simple example
"""

    result = generator(
        prompt,
        max_new_tokens=220,
        do_sample=False,
        truncation=True
    )

    return result[0]["generated_text"].strip()


def explain_concept(topic: str) -> str:

    if not topic.strip():
        raise ValueError(
            "Topic cannot be empty."
        )

    use_local = os.getenv(
        "USE_LOCAL_EXPLANATION",
        "true"
    ).lower() == "true"

    if use_local:

        try:
            return _local_explanation(topic)

        except Exception:
            # If local model fails,
            # automatically use Gemini.
            pass

    return generate_text(
        f"""
Explain the following educational topic
for a beginner:

{topic}

Use:

- A simple definition
- 3 to 5 key points
- One easy example
""",
        temperature=0.3,
        max_output_tokens=700
    )