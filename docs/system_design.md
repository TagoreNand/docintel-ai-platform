# System Design Notes

## Design choices

### Why a modular monolith for local development?
A senior engineer optimizes for both developer velocity and production migration. This repo packages the services into one FastAPI deployment locally, but isolates logic into service modules so each module can later become an independent service.

### Why hybrid retrieval?
Pure semantic retrieval can miss exact entities; pure keyword retrieval can miss paraphrased meaning. The project combines TF-IDF cosine similarity with keyword overlap to emulate a hybrid retriever.

### Why confidence-based review?
Trustworthy AI systems need operating thresholds. The pipeline uses:
- auto-approve: high confidence + low anomaly risk
- manual review: lower confidence or elevated risk

### Why rule-driven extraction?
For enterprise documents, high-precision deterministic fields still matter. This repo uses regex extraction to keep the project runnable, but the interfaces are designed so you can later swap in LayoutLM, Donut, or LLM extraction.

## Production evolution roadmap

1. Replace parser with OCR and layout-aware parsing
2. Move indexing to pgvector/Qdrant
3. Add Celery or Kafka-based async processing
4. Replace heuristic classification with trained multimodal models
5. Log model runs into MLflow
6. Add drift monitoring and reviewer feedback loops
