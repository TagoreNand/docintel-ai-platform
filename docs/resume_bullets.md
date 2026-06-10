# Key Points

- Built a production-inspired document intelligence platform (FastAPI, SQLAlchemy,
  React) with a real ML core: hybrid retrieval, calibrated classification,
  unsupervised anomaly detection, and human-in-the-loop review.
- Engineered a **hybrid retrieval stack** — neural sentence-embedding dense search +
  hand-implemented **BM25 Okapi**, fused with **Reciprocal Rank Fusion** and refined by
  an optional **cross-encoder reranker** (retrieve-then-rerank) — returning grounded
  extractive answers with ranked evidence.
- Implemented a **persisted vector store** (Qdrant + local numpy fallback), a calibrated
  **TF-IDF + logistic-regression** classifier (Platt scaling) blended with rules, and an
  **IsolationForest** anomaly detector alongside deterministic checks, all with model cards.
- Added enterprise capabilities: **OCR** (Tesseract) for scanned PDFs/images,
  **multi-tenant auth** (API key / JWT) with per-tenant data + search isolation,
  **Kafka event streaming** with a **distributed ingestion worker**, and **GPU-aware**
  inference (CUDA/fp16 auto-detection).
- Built an **observability stack**: Prometheus metrics (request/pipeline/retrieval),
  structured JSON logging with request-correlation ids, optional **OpenTelemetry** tracing,
  and optional **MLflow** experiment tracking — with Prometheus + Grafana in docker-compose.
- Designed everything **fallback-first**: neural models, GPU, Qdrant, Tesseract, Kafka,
  MLflow and OTel are all optional and auto-detected, so the system runs fully offline in
  CI (37 tests) and on laptops while production enables the full path with no code changes.
- Shipped Docker, a scalable Kubernetes worker deployment, Terraform, and GitHub Actions CI.
