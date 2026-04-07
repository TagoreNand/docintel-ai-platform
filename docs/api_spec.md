# API Summary

## Upload a document

`POST /api/v1/documents/upload`

Form-data:
- `file`: uploaded file

Returns document metadata and queues background processing.

## Ingest sample documents

`POST /api/v1/documents/ingest-sample`

Loads demo files from `sample_data/`.

## Search / QA

`POST /api/v1/search/query`

Body:
```json
{
  "question": "Which invoice mentions Nova Industrial Supplies?",
  "top_k": 5
}
```

Returns:
- answer
- ranked evidence
- related entities from matched documents

## Review queue

`GET /api/v1/review/tasks`

## Resolve a review task

`POST /api/v1/review/tasks/{task_id}/resolve`

Body:
```json
{
  "outcome": "approved",
  "notes": "Validated by reviewer."
}
```

## Analytics

`GET /api/v1/analytics/overview`
