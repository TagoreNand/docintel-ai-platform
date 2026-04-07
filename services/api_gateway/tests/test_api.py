from app.db.database import init_db
from app.main import app
from fastapi.testclient import TestClient


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_ingest_sample_and_search():
    init_db()
    with TestClient(app) as client:
        response = client.post("/api/v1/documents/ingest-sample")
        assert response.status_code == 200

        search_response = client.post("/api/v1/search/query", json={"question": "Which invoice mentions Nova Industrial Supplies?"})
        assert search_response.status_code == 200
        payload = search_response.json()
        assert "answer" in payload
        assert isinstance(payload["evidence"], list)
