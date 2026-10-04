from gemini_client import generate_text


def get_learning_recommendations(
    topic: str
) -> str:

    if not topic.strip():
        raise ValueError(
            "Topic cannot be empty."
        )

    prompt = f"""
Create a personalized learning path
for the following topic:

{topic}

Organize it into:

1. Prerequisites
2. Beginner stage
3. Intermediate stage
4. Advanced stage
5. Suggested timeline
6. Practice exercises
7. Project ideas
8. Useful resource types

The learning path should progress
from beginner to advanced.

Make the recommendations practical
and easy for a learner to follow.

If the learner's current level is unknown,
state that the timeline is flexible.
"""

    return generate_text(
        prompt,
        temperature=0.4,
        max_output_tokens=1400
    )