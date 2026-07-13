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
    resp = client.post("/ask",
