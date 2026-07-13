from fastapi.testclient import TestClient
from main import app, get_agent

client = TestClient(app)


class FakeResult:
    def __init__(self, content):
        self.content = content


class FakeAgent:
    def invoke(self, question):
        return FakeResult(f"Echo: {question}")


# Override the real ChatGroq dependency so tests don't hit the network / need a live model
app.dependency_overrides[get_agent] = lambda: FakeAgent()


def test_ask_returns_200():
    resp = client.post("/ask", json={"question": "What is DevOps?"})
    assert resp.status_code == 200


def test_ask_response_shape():
    resp = client.post("/ask", json={"question": "hello"})
    body = resp.json()
    assert "answer" in body
    assert body["answer"] == "Echo: hello"


def test_ask_missing_question_returns_422():
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_wrong_type_returns_422():
    resp = client.post("/ask", json={"question": 123, "extra": True})
    # pydantic coerces/validates; 123 is not a valid str -> 422
    assert resp.status_code == 422


def test_real_agent_builds_with_key(monkeypatch):
    # Ensures get_agent constructs when GROQ_API_KEY is present (repo secret in CI)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    agent = get_agent()
    assert agent is not None
