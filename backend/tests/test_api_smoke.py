from fastapi.testclient import TestClient

from app.main import app


def test_core_api_routes_are_available():
    client = TestClient(app)
    root = client.get("/")
    health = client.get("/health")
    config = client.get("/config")
    assert root.status_code == 200
    assert health.status_code == 200
    assert config.status_code == 200
    assert health.json()["status"] == "ok"
    assert config.json()["embedding_model"]
