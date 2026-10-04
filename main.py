from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from qna import answer_question
from explanation_module import explain_concept
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0",
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


# HTML templates
templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


# -----------------------------
# Request Models
# -----------------------------

class QARequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=10000
    )


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=30000
    )


# -----------------------------
# Frontend
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "EduGenie"
    }


# -----------------------------
# Q&A
# -----------------------------

@app.post("/qa")
async def qa(payload: QARequest):
    answer = answer_question(payload.question)

    return {
        "answer": answer
    }


# -----------------------------
# Explanation
# -----------------------------

@app.post("/explain")
async def explain(payload: TextRequest):
    explanation = explain_concept(payload.text)

    return {
        "explanation": explanation
    }


# -----------------------------
# Quiz
# -----------------------------

@app.post("/quiz")
async def quiz(payload: TextRequest):
    return generate_quiz(payload.text)


# -----------------------------
# Summarization
# -----------------------------

@app.post("/summarize")
async def summarize(payload: TextRequest):
    summary = summarize_text(payload.text)

    return {
        "summary": summary
    }


# -----------------------------
# Learning Path
# -----------------------------

@app.post("/learn/recommendations")
async def learning_recommendations(payload: TextRequest):
    recommendations = get_learning_recommendations(
        payload.text
    )

    return {
        "recommendations": recommendations
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)