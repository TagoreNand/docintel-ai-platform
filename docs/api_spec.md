# API Summary

Base path: `/api/v1`

When `ENABLE_AUTH=true`, all data endpoints require `X-API-Key: <key>` or
`Authorization: Bearer <jwt>` and are scoped to the caller's tenant.

## Documents

`POST /documents/upload` — multipart `file`; stores the file and queues async
processing. Returns document metadata.

`POST /documents/ingest-sample` — loads demo files from `sample_data/` and
processes them synchronously.

`GET /documents` — list documents (newest first).

`GET /documents/{id}` — document detail incl. extracted entities and chunks.

## Search / Grounded QA

`POST /search/query`

```json
{ "question": "Which invoice mentions Nova Industrial Supplies?", "top_k": 5, "rerank": null }
```

Returns:
- `answer` — extractive answer grounded in the best document
- `evidence[]` — ranked chunks with `score`, `dense_score`, `sparse_score`, `rerank_score`
- `related_entities` — entities from the matched documents
- `strategy` — `{ embedding_backend, vector_backend, reranker, dense_candidates, sparse_candidates, fused_candidates, reranked, latency_ms }`

`rerank` may be `true`/`false` to override the configured reranker per request.

## Review queue

`GET /review/tasks` — list review tasks.

`POST /review/tasks/{task_id}/resolve`

```json
{ "outcome": "approved", "notes": "Validated by reviewer." }
```

## Analytics

`GET /analytics/overview` — totals by type/status, review status, avg anomaly score.

## System & observability

`GET /health` · `GET /ready` — liveness / readiness.

`GET /system/info` — live backend status: embedding backend + dim, vector store
backend + vector count, retrieval config, and whether the classifier / anomaly
models are loaded (with their model-card metrics).

`GET /metrics` — Prometheus-format gauges (`docintel_documents_total`,
`docintel_chunks_total`, `docintel_review_tasks_open`, `docintel_index_vectors`,
`docintel_avg_anomaly_score`).

`GET /events/recent?limit=50` — most recent domain events from the event bus
(`document.uploaded`, `document.processed`, `document.needs_review`, `review.resolved`).
