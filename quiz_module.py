from pydantic import BaseModel, Field

from gemini_client import generate_structured


class QuizQuestion(BaseModel):

    question: str

    options: list[str] = Field(
        min_length=4,
        max_length=4
    )

    correct_answer: str

    explanation: str


class QuizResponse(BaseModel):

    questions: list[QuizQuestion] = Field(
        min_length=3,
        max_length=3
    )


def generate_quiz(text: str) -> dict:

    if not text.strip():
        raise ValueError(
            "Passage/topic cannot be empty."
        )

    prompt = f"""
Create exactly 3 educational
multiple-choice questions from
the material below.

Requirements:

- Exactly 3 questions.
- Exactly 4 options per question.
- Only one option should be correct.
- correct_answer must exactly match
  one of the four options.
- Include an explanation for the answer.
- Questions should test understanding.
- Do not add extra fields.
- Return valid JSON matching the schema.

Material:

{text}
"""

    raw = generate_structured(
        prompt,
        QuizResponse,
        temperature=0.25,
        max_output_tokens=1800
    )

    quiz = QuizResponse.model_validate_json(
        raw
    )

    return quiz.model_dump()