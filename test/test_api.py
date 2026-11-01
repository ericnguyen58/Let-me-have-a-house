import pytest
from fastapi.testclient import TestClient

from api import main
from src.components import assistant


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(assistant, "warm_up", lambda: None)  # no data or model loading at startup
    with TestClient(main.app) as client:
        yield client


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_chat_passes_the_question_and_history(client, monkeypatch):
    calls = []
    monkeypatch.setattr(assistant, "ask", lambda question, history: calls.append((question, history)) or "answer")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    response = client.post("/api/chat", json={"question": "worth?", "history": history})
    assert response.json() == {"answer": "answer"}
    assert calls == [("worth?", history)]


def test_missing_model_server_is_a_503(client, monkeypatch):
    def no_ollama(question, history):
        raise SystemExit("Ollama is not running at http://localhost:11434")

    monkeypatch.setattr(assistant, "ask", no_ollama)
    response = client.post("/api/chat", json={"question": "worth?"})
    assert response.status_code == 503
    assert "Ollama is not running" in response.json()["detail"]


def test_bad_history_is_rejected(client):
    response = client.post("/api/chat", json={"question": "x", "history": [{"role": "system", "content": "y"}]})
    assert response.status_code == 422
