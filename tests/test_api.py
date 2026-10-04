import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "EduGenie"


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "EduGenie" in response.text


def test_qa_validation_error():
    response = client.post("/qa", json={"question": ""})
    assert response.status_code == 422


def test_explain_validation_error():
    response = client.post("/explain", json={"text": ""})
    assert response.status_code == 422


def test_quiz_validation_error():
    response = client.post("/quiz", json={"text": ""})
    assert response.status_code == 422


def test_summarize_validation_error():
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 422


def test_learning_recommendations_validation_error():
    response = client.post("/learn/recommendations", json={"text": ""})
    assert response.status_code == 422


@patch("main.answer_question")
def test_qa_endpoint(mock_qa):
    mock_qa.return_value = "Gravity is an attractive force between masses."
    response = client.post("/qa", json={"question": "What is gravity?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Gravity is an attractive force between masses."
    mock_qa.assert_called_once_with("What is gravity?")


@patch("main.explain_concept")
def test_explain_endpoint(mock_explain):
    mock_explain.return_value = "1. Definition: Quantum computing uses qubits..."
    response = client.post("/explain", json={"text": "Quantum Computing"})
    assert response.status_code == 200
    assert "Definition" in response.json()["explanation"]
    mock_explain.assert_called_once_with("Quantum Computing")


@patch("main.generate_quiz")
def test_quiz_endpoint(mock_quiz):
    mock_quiz.return_value = {
        "questions": [
            {
                "question": "What is H2O?",
                "options": ["Water", "Oxygen", "Hydrogen", "Carbon"],
                "correct_answer": "Water",
                "explanation": "H2O is water."
            }
        ]
    }
    response = client.post("/quiz", json={"text": "Water is H2O."})
    assert response.status_code == 200
    data = response.json()
    assert len(data["questions"]) == 1
    assert data["questions"][0]["correct_answer"] == "Water"


@patch("main.summarize_text")
def test_summarize_endpoint(mock_summarize):
    mock_summarize.return_value = "Brief summary of the passage."
    response = client.post("/summarize", json={"text": "Long text passage to summarize."})
    assert response.status_code == 200
    assert response.json()["summary"] == "Brief summary of the passage."
    mock_summarize.assert_called_once_with("Long text passage to summarize.")


@patch("main.get_learning_recommendations")
def test_learning_recommendations_endpoint(mock_learn):
    mock_learn.return_value = "Roadmap: 1. Python basics, 2. Data structures"
    response = client.post("/learn/recommendations", json={"text": "Python"})
    assert response.status_code == 200
    assert "Roadmap" in response.json()["recommendations"]
    mock_learn.assert_called_once_with("Python")
