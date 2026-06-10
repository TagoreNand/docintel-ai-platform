from app.main import app
from fastapi.testclient import TestClient


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_system_info_and_metrics():
    with TestClient(app) as client:
        info = client.get("/api/v1/system/info").json()
        assert info["embedding"]["backend"] == "hashing"
        assert info["vector_store"]["backend"] == "local"
        assert "rrf_k" in info["retrieval"]

        metrics = client.get("/api/v1/metrics")
        assert metrics.status_code == 200
        assert "docintel_documents" in metrics.text


def test_ingest_sample_and_search():
    with TestClient(app) as client:
        response = client.post("/api/v1/documents/ingest-sample")
        assert response.status_code == 200
        assert len(response.json()) >= 1

        search = client.post(
            "/api/v1/search/query",
            json={"question": "Which invoice mentions Nova Industrial Supplies?"},
        )
        assert search.status_code == 200
        payload = search.json()
        assert "answer" in payload
        assert isinstance(payload["evidence"], list)
        assert payload["strategy"]["embedding_backend"] == "hashing"
