.PHONY: setup backend frontend test demo worker retrain model-gate train train-classifier train-anomaly index-rebuild openapi zip

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r services/api_gateway/requirements.txt

backend:
	cd services/api_gateway && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend/web && npm install && npm run dev

test:
	cd services/api_gateway && EMBEDDING_BACKEND=hashing VECTOR_BACKEND=local ENABLE_RERANKER=false pytest -q

train: ## Train classifier + anomaly models into data/models/
	cd services/api_gateway && python -m app.ml.train_classifier && python -m app.ml.train_anomaly

train-classifier:
	cd services/api_gateway && python -m app.ml.train_classifier

train-anomaly:
	cd services/api_gateway && python -m app.ml.train_anomaly

index-rebuild: ## Rebuild the vector index from the database
	python scripts/rebuild_index.py

demo: ## Train models, ingest sample docs, build the index
	python scripts/bootstrap_demo.py

worker: ## Run the distributed Kafka ingestion worker
	python scripts/worker.py

retrain: ## Retrain the classifier, folding in reviewer feedback
	python scripts/retrain_from_feedback.py

model-gate: ## Fail if classifier metrics regress below thresholds (CI gate)
	python scripts/check_model_quality.py

openapi:
	cd services/api_gateway && python ../../scripts/export_openapi.py

zip:
	cd .. && zip -r docintel_ai_platform.zip docintel_ai_platform
