# System Design Notes

## Design choices

### Why a modular monolith for local development?
A senior engineer optimizes for both developer velocity and production migration.
This repo packages the services into one FastAPI deployment locally, but isolates
logic into service modules so each module can later become an independent service.

### Why hybrid retrieval (and how)?
Pure semantic retrieval misses exact entities (IDs, amounts); pure lexical retrieval
misses paraphrase. DocIntel runs **both**: neural bi-encoder embeddings for semantic
recall and a hand-implemented **BM25 Okapi** for lexical recall, then merges them
with **Reciprocal Rank Fusion** (rank-based, score-scale agnostic). A **cross-encoder**
reranker optionally re-scores the top fused candidates — the standard
*retrieve-then-rerank* architecture used in modern search.

### Why calibrated probabilities for classification?
Operating thresholds (auto-approve / human-review) are only meaningful if the
confidence is trustworthy. The classifier is a TF-IDF + logistic regression model
wrapped in `CalibratedClassifierCV` (Platt scaling), and its probability is blended
with rule-engine agreement so the final confidence reflects both signals.

### Why rules **and** an unsupervised anomaly model?
Rules are precise and auditable (e.g. `subtotal + tax != total`), but they only
catch known failure modes. An **IsolationForest** trained on the joint feature
distribution flags "weird-looking" documents that break no single rule. Combining
them keeps high precision while adding novelty detection.

### Why a pluggable, fallback-first design?
Reproducibility and portability. Every heavy dependency (torch, Qdrant, MLflow,
cross-encoder) is optional and auto-detected, so CI and laptops run offline and
deterministically, while production lights up the neural path with no code changes.

### Why rule-driven extraction?
High-precision deterministic fields still matter for enterprise documents. Regex
extraction keeps the project runnable; the interface is designed so LayoutLM, Donut,
or LLM extraction can be swapped in later.

## Enterprise design choices

### Why fallback-first multi-tenancy?
`get_tenant` resolves the caller from an API key or JWT and is the single seam for
isolation; documents carry a `tenant_id` and every read is scoped. With auth off the
resolver returns one default tenant, so dev/CI stay frictionless and production flips a
flag — no route rewrites, no schema fork.

### Why event-driven ingestion + a worker?
Document processing (embedding, OCR, model inference) is heavier than an HTTP request
should carry. Publishing `document.uploaded` to Kafka and consuming it in a separate
worker decouples ingestion from the API, lets the worker scale independently, and is the
first concrete step from modular monolith to microservices. In-process background tasks
remain the zero-dependency default.

### Why Prometheus + OpenTelemetry (and a custom fallback)?
Operating an ML system means watching latency and throughput per stage. Prometheus
client metrics (counters/histograms for requests, pipeline and retrieval) give RED-style
signals scraped by the bundled Prometheus/Grafana; OpenTelemetry adds distributed traces
when enabled. When the client libraries are absent, `/metrics` still serves plain-text
gauges so the endpoint never disappears.

### Why GPU auto-detection rather than a hard requirement?
`resolve_device()` picks CUDA (with fp16) when available and CPU otherwise, so the same
image runs on a laptop or a GPU node. Batch sizes and half precision engage only on GPU.

## Production evolution roadmap

1. OCR + layout-aware parsing for scanned documents
2. Tune Qdrant ANN params / quantization at scale; per-tenant collections
3. Celery or Kafka-based async ingestion at high throughput
4. Replace synthetic training data with labelled production corpora; add active learning
5. Promote MLflow from optional logging to a model registry + CI model gates
6. Embedding / label drift monitoring and reviewer feedback loops
