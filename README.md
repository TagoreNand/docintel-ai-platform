# DocIntel AI Platform

![Build](https://img.shields.io/badge/build-ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)
![ML](https://img.shields.io/badge/ML-hybrid%20document%20intelligence-purple)
![License](https://img.shields.io/badge/license-MIT-black)

A senior-level, production-inspired AI/ML project for enterprise document intelligence. The platform ingests documents, classifies them, extracts structured fields, indexes semantic chunks, supports grounded question answering, scores anomalies, and routes low-confidence outputs to human review.

## Why this project stands out



- **Document ingestion and parsing** for TXT, MD, JSON, CSV, and PDF
- **Hybrid document classification** (rule-weighted ML-friendly architecture)
- **Structured information extraction** with confidence scoring
- **Semantic retrieval** over chunked evidence
- **Grounded QA** with evidence-backed responses
- **Anomaly scoring** for invoices and claims
- **Human-in-the-loop review queue**
- **Operational analytics** for approvals, anomalies, and queue depth
- **Docker, Kubernetes, CI, Postman, and system design docs**
- **A React dashboard** for upload, search, review, and analytics

## Core use cases

- Invoice intelligence and duplicate detection
- Contract lifecycle extraction
- Claims triage and risk review
- Enterprise search over uploaded document corpora
- AI-assisted reviewer workflows

## Architecture

See [`docs/architecture.md`](docs/architecture.md) and [`docs/architecture_diagram.png`](docs/architecture_diagram.png).

```text
Client Apps / Integrations
        ↓
API Gateway (FastAPI)
        ↓
Ingestion Pipeline
  ├─ Parser / OCR-ready layer
  ├─ Document Classifier
  ├─ Field Extractor
  ├─ Chunking + Retrieval Index
  ├─ Anomaly Scorer
  └─ Review Router
        ↓
Persistence Layer (SQL + file store + vector-ready index)
        ↓
Analytics / Search / Review APIs
```

## Project structure

```text
docintel_ai_platform/
├── services/api_gateway/     # FastAPI backend and ML pipeline
├── frontend/web/             # React dashboard
├── docs/                     # Architecture, API spec, resume bullets
├── infra/                    # Docker, Kubernetes, Terraform scaffolding
├── postman/                  # API collection
├── sample_data/              # Demo documents
├── scripts/                  # Bootstrap, OpenAPI export
└── tests/                    # Integration tests (via backend package)
```

## Fast local start

### Backend

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r services/api_gateway/requirements.txt
cd services/api_gateway
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend/web
npm install
npm run dev
```

### Demo data bootstrap

```bash
python scripts/bootstrap_demo.py
```

### Tests

```bash
cd services/api_gateway
pytest -q
```

## API highlights

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/documents/upload` | POST | Upload and process a document |
| `/api/v1/documents` | GET | List documents |
| `/api/v1/documents/{id}` | GET | Detailed document record |
| `/api/v1/search/query` | POST | Semantic search / grounded QA |
| `/api/v1/review/tasks` | GET | Reviewer queue |
| `/api/v1/review/tasks/{id}/resolve` | POST | Resolve a task |
| `/api/v1/analytics/overview` | GET | Operational metrics |

## Discussion points

- **ML systems design**: modular pipeline with clean service boundaries
- **Reliability**: confidence thresholds and human review routing
- **Search quality**: chunking, scoring, entity grounding, evidence returns
- **Production readiness**: Docker, K8s manifests, health endpoints, CI
- **MLOps**: registry- and vector-db-ready architecture, analytics hooks, retraining roadmap
- **Product thinking**: measurable reviewer efficiency and anomaly detection workflow

## Production upgrade path

This repo is intentionally runnable in local development without external infrastructure. For production, swap the dev implementations with:

- PostgreSQL + Alembic migrations
- Redis/Celery or Kafka workers
- Qdrant / pgvector for vector search
- Layout-aware models for PDFs and scanned images
- MLflow experiment tracking and model registry
- OCR workers using Tesseract / PaddleOCR
- OpenTelemetry, Prometheus, and Grafana

## Included deliverables

- Complete backend code
- Frontend dashboard
- Docker and Kubernetes manifests
- Postman collection
- Architecture diagram
- System design docs
- Resume bullets
- Sample data
- Automated tests

## Progress 

See [`docs/resume_bullets.md`](docs/resume_bullets.md).

## License

MIT
