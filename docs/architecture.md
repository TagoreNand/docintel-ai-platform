# Architecture Overview

## System goals

- Process enterprise documents end to end (including scanned PDFs / images via OCR)
- Extract structured intelligence and index it for semantic + lexical search
- Answer grounded questions with ranked, citable, tenant-scoped evidence
- Detect anomalies with explainable rules **and** an unsupervised model
- Route uncertain / risky cases to human review
- Scale horizontally (event-driven worker) and stay observable in production
- Remain local-dev friendly (zero services required) via graceful fallbacks

## Core components

1. **API Gateway** (`app/api`) — routers for documents, search, review, analytics,
   health, system; request-id middleware, structured logging, Prometheus + optional OTel.
2. **Ingestion Pipeline** (`app/services/pipeline.py`) — parse → classify → extract →
   chunk → anomaly score → persist → incrementally index → route to review.
3. **Parser + OCR** (`parser.py`, `ocr.py`) — text/JSON/CSV/PDF parsing with a
   Tesseract OCR fallback for image files and image-only (scanned) PDFs.
4. **Embeddings** (`embeddings.py`) — neural `SentenceTransformerEmbedder` (GPU-aware)
   or deterministic `HashingEmbedder` fallback.
5. **Vector store** (`vector_store.py`) — `QdrantVectorStore` or persisted `LocalVectorStore`.
6. **Hybrid retrieval** (`retrieval.py`, `sparse_index.py`, `reranker.py`) — dense +
   BM25 Okapi → Reciprocal Rank Fusion → optional cross-encoder rerank → extractive answer.
7. **Classification** (`classification.py`, `ml/train_classifier.py`) — calibrated
   TF-IDF + LogReg blended with rules; rule fallback.
8. **Anomaly detection** (`anomaly.py`, `ml/train_anomaly.py`) — business rules + IsolationForest.
9. **Persistence** (`app/db`) — SQLAlchemy models (documents incl. `tenant_id`, entities,
   chunks, review tasks); SQLite locally, Postgres in compose.

## Enterprise capabilities

- **Multi-tenancy & auth** (`core/security.py`) — `get_tenant` resolves the caller from an
  `X-API-Key` (mapped to a tenant) or a Bearer JWT. Documents carry a `tenant_id`; list,
  detail, search, review and analytics are all tenant-scoped. Disabled by default
  (single open tenant) and enforced when `ENABLE_AUTH=true`.
- **Event streaming** (`services/events.py`) — domain events (`document.uploaded`,
  `document.processed`, `document.needs_review`, `review.resolved`) to Kafka, with an
  in-memory ring-buffer fallback and a `/events/recent` feed.
- **Distributed worker** (`app/worker.py`) — when `INGESTION_MODE=kafka`, uploads publish
  `document.uploaded` and a separate worker process consumes and runs the pipeline,
  decoupling ingestion from the API for horizontal scale.
- **GPU inference** — embedder and reranker call `resolve_device()` to auto-detect CUDA
  (fp16) and otherwise run on CPU.
- **Observability** (`core/observability.py`) — Prometheus counters/histograms (requests,
  pipeline, retrieval) + gauges at `/metrics`; structured JSON logs with request ids;
  optional OpenTelemetry OTLP tracing. Prometheus + Grafana ship in docker-compose.
- **Experiment tracking** (`services/tracking.py`) — optional MLflow run/metric/artifact logging.

## Graceful degradation matrix

| Component | Preferred | Fallback |
|---|---|---|
| Embeddings | sentence-transformers (GPU/CPU) | deterministic hashing |
| Vector store | Qdrant | local numpy store |
| Reranking | cross-encoder | fused RRF order |
| Classifier | calibrated LogReg | keyword rules |
| Anomaly | rules + IsolationForest | rules only |
| OCR | Tesseract | skipped (text-layer only) |
| Ingestion | Kafka + worker | in-process background task |
| Events | Kafka | in-memory ring buffer |
| Auth | API key / JWT | open, single default tenant |
| Metrics | prometheus-client | plain-text gauges |
| Tracing | OpenTelemetry | disabled |
| Tracking | MLflow (+ model registry) | no-op |
| Layout OCR | Donut / LayoutLM | Tesseract -> none |
| Tenancy/scale | per-tenant Qdrant collections + HNSW/quantization | per-tenant local numpy dirs |
| Quality gate | CI fails on metric regression | n/a |
| Drift/learning | PSI + embedding drift, feedback retraining | n/a |

## Production decomposition

The modular monolith maps cleanly onto `ingestion`, `extraction`, `retrieval`,
`anomaly`, `review` and `analytics` services. The Kafka worker is the first real
split: ingestion already runs as an independent, separately-scalable process.

## Continual learning & scale

- **Per-tenant indexes** — each tenant gets its own Qdrant collection (or local index
  directory); HNSW (`m`, `ef_construct`) and scalar quantization are configurable.
- **Layout OCR** — `LayoutOcrEngine` (Donut / LayoutLM) sits ahead of Tesseract in the
  OCR chain when `ENABLE_LAYOUT_OCR=true`.
- **Model governance** — training logs to MLflow and can register models; a CI gate
  (`scripts/check_model_quality.py`) fails the build if accuracy / macro-F1 regress.
- **Drift & feedback** — `services/drift.py` reports PSI over predictions/confidence plus
  embedding-centroid drift against a training-time baseline; reviewer corrections are
  captured (`services/feedback.py`) and folded back in by `scripts/retrain_from_feedback.py`.
