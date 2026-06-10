from pathlib import Path
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration.

    Values are read from environment variables (or a local ``.env`` file) and
    fall back to local-dev friendly defaults. Every external dependency
    (neural embeddings, Qdrant, MLflow, cross-encoder reranking, OCR, Kafka,
    OpenTelemetry) is optional and degrades gracefully to an offline
    implementation when unavailable, so the platform runs anywhere with zero
    services running.
    """

    # ----- Core app -------------------------------------------------------
    app_name: str = "DocIntel AI Platform"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"
    debug: bool = True
    secret_key: str = "change-me"
    log_level: str = "INFO"
    log_json: bool = False

    # ----- Storage --------------------------------------------------------
    database_url: str = "sqlite:///./data/docintel.db"
    upload_dir: str = "./data/uploads"
    model_dir: str = "./data/models"
    index_dir: str = "./data/index"

    allowed_origins: List[str] | str = ["http://localhost:5173", "http://localhost:3000"]

    # ----- Decisioning thresholds ----------------------------------------
    auto_approve_threshold: float = 0.92
    human_review_threshold: float = 0.70
    anomaly_review_threshold: float = 0.25
    anomaly_high_priority_threshold: float = 0.5

    # ----- Embeddings -----------------------------------------------------
    # "auto" prefers a real neural sentence-transformer and silently falls
    # back to the deterministic hashing embedder when torch / the model is
    # unavailable. Force a backend with "sentence_transformer" or "hashing".
    embedding_backend: Literal["auto", "sentence_transformer", "hashing"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_embedding_dim: int = 384
    embedding_batch_size: int = 32

    # ----- Inference device (GPU acceleration when available) ------------
    inference_device: Literal["auto", "cpu", "cuda"] = "auto"
    inference_fp16: bool = True  # use half precision on CUDA

    # ----- Vector store ---------------------------------------------------
    vector_backend: Literal["auto", "qdrant", "local"] = "auto"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docintel_chunks"
    qdrant_timeout: float = 3.0

    # ----- Hybrid retrieval ----------------------------------------------
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    answer_max_sentences: int = 3

    # ----- Cross-encoder reranking ---------------------------------------
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidates: int = 20

    # ----- Classifier -----------------------------------------------------
    classifier_version: str = "tfidf-logreg-v1"
    classifier_min_confidence: float = 0.45  # below this we trust rules more
    classifier_rule_blend: float = 0.25  # weight of rule agreement in blended score

    # ----- Anomaly model --------------------------------------------------
    anomaly_model_version: str = "iforest-v1"
    anomaly_ml_weight: float = 0.2  # contribution of the unsupervised model

    # ----- OCR ------------------------------------------------------------
    ocr_enabled: bool = True
    ocr_languages: str = "eng"
    ocr_min_chars: int = 24  # parsed PDFs under this are treated as scanned -> OCR

    # ----- Authentication / multi-tenancy --------------------------------
    enable_auth: bool = False
    default_tenant: str = "default"
    auth_api_keys: str = ""  # comma-separated "key:tenant" pairs
    jwt_secret: str = "change-me-jwt"
    jwt_algorithm: str = "HS256"

    # ----- Ingestion mode / Kafka events ---------------------------------
    ingestion_mode: Literal["inproc", "kafka"] = "inproc"
    enable_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "docintel"
    kafka_consumer_group: str = "docintel-workers"

    # ----- Observability stack -------------------------------------------
    enable_prometheus: bool = True
    enable_otel: bool = False
    otel_service_name: str = "docintel-api"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # ----- Experiment tracking -------------------------------------------
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "docintel"

    # ----- Layout-aware OCR (Donut / LayoutLM) ---------------------------
    enable_layout_ocr: bool = False
    layout_model: str = "naver-clova-ix/donut-base"

    # ----- Vector store scale / per-tenant -------------------------------
    qdrant_per_tenant: bool = True
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
    qdrant_quantization: Literal["none", "scalar"] = "none"

    # ----- MLflow registry + model-quality gates -------------------------
    mlflow_register_models: bool = False
    classifier_accuracy_gate: float = 0.85
    classifier_macro_f1_gate: float = 0.85

    # ----- Drift monitoring ----------------------------------------------
    drift_psi_warn: float = 0.1
    drift_psi_alert: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".json", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    @property
    def image_extensions(self) -> set[str]:
        return {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    @property
    def api_key_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for pair in self.auth_api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                key, tenant = pair.split(":", 1)
                mapping[key.strip()] = tenant.strip()
            else:
                mapping[pair] = self.default_tenant
        return mapping

    @property
    def project_root(self) -> Path:
        # Resolve the base dir for data/models/index across layouts:
        #  - local repo:  <repo>/services/api_gateway/app/core/config.py -> <repo>
        #  - container:   /app/app/core/config.py (service copied to /app) -> /app
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "sample_data").is_dir() or (parent / "docker-compose.yml").exists():
                return parent
        return here.parents[2] if len(here.parents) > 2 else here.parents[-1]

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def model_path(self) -> Path:
        return self._resolve(self.model_dir)

    @property
    def index_path(self) -> Path:
        return self._resolve(self.index_dir)

    @property
    def upload_path(self) -> Path:
        return self._resolve(self.upload_dir)

    @property
    def classifier_path(self) -> Path:
        return self.model_path / "document_classifier.joblib"

    @property
    def classifier_card_path(self) -> Path:
        return self.model_path / "document_classifier.card.json"

    @property
    def anomaly_model_path(self) -> Path:
        return self.model_path / "anomaly_iforest.joblib"

    @property
    def anomaly_card_path(self) -> Path:
        return self.model_path / "anomaly_iforest.card.json"

    @property
    def drift_baseline_path(self) -> Path:
        return self.model_path / "drift_baseline.json"

    @property
    def feedback_path(self) -> Path:
        return self.model_path / "feedback.jsonl"


settings = Settings()
