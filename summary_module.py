from gemini_client import generate_text


def summarize_text(text: str) -> str:

    if not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    prompt = f"""
Summarize the following educational passage.

Requirements:

- Keep the main ideas.
- Keep important facts.
- Keep important terminology.
- Remove unnecessary repetition.
- Use simple language.
- Make it useful for quick revision.
- Do not introduce information that
  is not present in the original passage.

Passage:

{text}
"""

    return generate_text(
        prompt,
        temperature=0.25,
        max_output_tokens=900
    )