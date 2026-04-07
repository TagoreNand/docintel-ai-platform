.PHONY: setup backend frontend test demo openapi zip

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r services/api_gateway/requirements.txt

backend:
	cd services/api_gateway && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend/web && npm install && npm run dev

test:
	cd services/api_gateway && pytest -q

demo:
	cd services/api_gateway && python ../../scripts/bootstrap_demo.py

openapi:
	cd services/api_gateway && python ../../scripts/export_openapi.py

zip:
	cd .. && zip -r docintel_ai_platform.zip docintel_ai_platform
