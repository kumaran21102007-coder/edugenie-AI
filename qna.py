from gemini_client import generate_text


SYSTEM_PROMPT = """
You are EduGenie, an educational AI assistant.

Your job is to help students understand academic topics.

Rules:
- Give accurate answers.
- Use simple language.
- Be concise but useful.
- Explain unfamiliar terms.
- Use examples when helpful.
- For multi-step questions, organize the answer clearly.
- Do not invent facts.
- If a question is ambiguous, mention the ambiguity.
"""


def answer_question(question: str) -> str:

    prompt = f"""
{SYSTEM_PROMPT}

Student question:

{question}

Give a helpful educational answer.
"""

    return generate_text(
        prompt,
        temperature=0.3,
        max_output_tokens=1000
    )