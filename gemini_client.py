import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types


try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

load_dotenv()


DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)


class GeminiConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Create and cache the Gemini client.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=api_key
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 1024,
) -> str:
    """
    Generate normal text using Gemini.
    """

    client = get_client()

    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    text = response.text

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


def generate_structured(
    prompt: str,
    response_schema,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 1800,
) -> str:
    """
    Generate structured JSON using Gemini.
    """

    client = get_client()

    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty structured response."
        )

    return response.text.strip()