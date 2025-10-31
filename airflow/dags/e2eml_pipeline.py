"""Airflow DAG skeleton for the OpenFoodFacts end-to-end ML pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from minio import Minio
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from project root .env if present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Environment configuration
OPENFOODFACTS_KAGGLE_DATASET = os.getenv("OPENFOODFACTS_KAGGLE_DATASET")
STARBUCKS_KAGGLE_DATASET = os.getenv("STARBUCKS_KAGGLE_DATASET")
MCDONALDS_KAGGLE_DATASET = os.getenv("MCDONALDS_KAGGLE_DATASET")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
RAW_PATH = os.getenv("RAW_PATH")
SILVER_PATH = os.getenv("SILVER_PATH")
GOLD_PATH = os.getenv("GOLD_PATH")


def _parse_s3_uri(uri: str | None) -> tuple[str | None, str | None]:
    """Return (bucket, key_prefix) for S3-style URIs or bare key strings."""
    if not uri:
        return None, None
    raw = uri.strip()
    if not raw:
        return None, None
    if "://" in raw:
        scheme, remainder = raw.split("://", 1)
        if scheme.lower() != "s3":
            raise ValueError(f"Unsupported URI scheme: {uri!r}")
    else:
        remainder = raw
    parts = [part for part in remainder.split("/") if part]
    if not parts:
        return None, None
    bucket = parts[0]
    prefix = "/".join(parts[1:]) if len(parts) > 1 else None
    return bucket, prefix


def _join_keys(*segments: str | os.PathLike[str] | None) -> str:
    """Join path segments with '/', skipping empty or None segments."""
    cleaned: list[str] = []
    for segment in segments:
        if segment is None:
            continue
        value = os.fspath(segment).strip("/")
        if value:
            cleaned.append(value)
    return "/".join(cleaned)


# Default arguments shared by all tasks
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="e2eml_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["e2eml", "mlops"],
)
def e2eml_pipeline() -> None:
    """Define the data-to-production pipeline stages as Airflow tasks."""

    @task(task_id="ingestion_data")
    def ingestion_data() -> str:
        """Download configured Kaggle datasets and upload them to MinIO."""
        ds_nodash = get_current_context()["ds_nodash"]

        dataset_configs = [
            ("openfoodfacts", OPENFOODFACTS_KAGGLE_DATASET, "OPENFOODFACTS_KAGGLE_DATASET"),
            ("starbucks", STARBUCKS_KAGGLE_DATASET, "STARBUCKS_KAGGLE_DATASET"),
            ("mcdonalds", MCDONALDS_KAGGLE_DATASET, "MCDONALDS_KAGGLE_DATASET"),
        ]

        datasets_to_fetch: list[dict[str, str]] = []
        for folder_name, dataset_ref, env_name in dataset_configs:
            if not dataset_ref:
                continue
            slug = dataset_ref.rstrip("/")  # normalize accidental trailing slash
            if slug.lower().endswith(".zip"):
                slug = slug[:-4]
            datasets_to_fetch.append(
                {"folder": folder_name, "ref": dataset_ref, "slug": slug, "env": env_name}
            )

        if not datasets_to_fetch:
            raise ValueError(
                "Configure at least one Kaggle dataset via OPENFOODFACTS_KAGGLE_DATASET, "
                "STARBUCKS_KAGGLE_DATASET, or MCDONALDS_KAGGLE_DATASET."
            )

        staging_dir = Path(tempfile.mkdtemp(prefix="openfoodfacts_", dir="/tmp"))

        if KAGGLE_USERNAME and KAGGLE_KEY:
            os.environ.setdefault("KAGGLE_USERNAME", KAGGLE_USERNAME)
            os.environ.setdefault("KAGGLE_KEY", KAGGLE_KEY)
        else:
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_json.exists():
                raise RuntimeError(
                    "Kaggle credentials are missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
                    "environment variables or mount ~/.kaggle/kaggle.json inside the Airflow container."
                )
            os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_json.parent))

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except OSError as exc:
            raise RuntimeError(
                "Failed to import Kaggle API because authentication could not be established. "
                "Ensure KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json are available to the container."
            ) from exc

        api = KaggleApi()
        api.authenticate()

        for dataset_cfg in datasets_to_fetch:
            target_dir = staging_dir / dataset_cfg["folder"]
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Downloading Kaggle dataset %s (env %s) into %s",
                dataset_cfg["ref"],
                dataset_cfg["env"],
                target_dir,
            )
            try:
                api.dataset_download_files(
                    dataset_cfg["slug"],
                    path=target_dir,
                    unzip=True,
                    force=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download dataset {dataset_cfg['ref']} "
                    f"(environment variable {dataset_cfg['env']})"
                ) from exc

        endpoint = MINIO_ENDPOINT
        access_key = MINIO_ACCESS_KEY
        secret_key = MINIO_SECRET_KEY
        if not all([endpoint, access_key, secret_key]):
            raise ValueError("Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.")

        bucket = MINIO_BUCKET
        if not bucket:
            raise ValueError("Set MINIO_BUCKET in the environment configuration.")

        base_prefix = RAW_PATH

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=MINIO_SECURE,
        )

        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        uploaded = 0
        for dataset_cfg in datasets_to_fetch:
            dataset_dir = staging_dir / dataset_cfg["folder"]
            dataset_prefix = (
                f"{base_prefix}/{dataset_cfg['folder']}"
                if base_prefix
                else dataset_cfg["folder"]
            )
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(dataset_dir)
                    object_name = f"{dataset_prefix}/{relative_path}".replace("\\", "/")
                    client.fput_object(bucket, object_name, file_path.as_posix())
                    uploaded += 1

        dataset_summary = ", ".join(cfg["ref"] for cfg in datasets_to_fetch)
        target_path = f"s3://{bucket}/{base_prefix}" if base_prefix else f"s3://{bucket}"
        logger.info(
            "Fetched Kaggle datasets [%s] and uploaded %d files to %s/%s",
            dataset_summary,
            uploaded,
            bucket,
            base_prefix or "",
        )
        return target_path.rstrip("/")

    @task(task_id="preprocess_data")
    def preprocess_data(raw_path: str) -> str:
        """Clean and validate raw data, persist to processed bucket."""
        # TODO: call preprocessing scripts / notebooks
        processed_path = "s3://openfoodfacts-processed/{{ ds_nodash }}/"
        logger.info(
            "Preprocessed dataset from %s and saved to %s", raw_path, processed_path
        )
        return processed_path

    @task(task_id="build_features")
    def build_features(processed_path: str) -> str:
        """Generate feature table ready for model training."""
        # TODO: implement feature engineering pipeline
        feature_path = "s3://openfoodfacts-features/{{ ds_nodash }}/"
        logger.info(
            "Built feature table from %s and stored at %s", processed_path, feature_path
        )
        return feature_path

    @task(task_id="train_model")
    def train_model(feature_path: str) -> str:
        """Train model and capture metrics + artifacts via MLflow."""
        # TODO: integrate with training script and MLflow tracking URI
        model_uri = "mlflow://experiments/openfoodfacts/models/{{ ds_nodash }}"
        logger.info("Trained model with features at %s -> %s", feature_path, model_uri)
        return model_uri

    @task(task_id="register_model")
    def register_model(model_uri: str) -> str:
        """Register the trained model in the model registry."""
        # TODO: call MLflow model registry or custom registry logic
        model_version = "openfoodfacts:{{ ds_nodash }}"
        logger.info("Registered model %s from %s", model_version, model_uri)
        return model_version

    @task(task_id="deploy_model")
    def deploy_model(model_version: str) -> None:
        """Deploy model for batch/online inference."""
        # TODO: implement deployment to serving environment
        logger.info("Deployment triggered for model %s", model_version)

    @task(task_id="monitor_pipeline")
    def monitor_pipeline(model_version: str) -> None:
        """Log post-deployment monitoring checkpoint."""
        # TODO: push metrics to Prometheus/Grafana or trigger drift checks
        logger.info("Monitoring checks scheduled for %s", model_version)

    # Task dependencies
    raw_dataset = ingestion_data()
    processed_dataset = preprocess_data(raw_dataset)
    feature_table = build_features(processed_dataset)
    trained_model = train_model(feature_table)
    registered_model = register_model(trained_model)
    deployment = deploy_model(registered_model)
    monitoring = monitor_pipeline(registered_model)

    raw_dataset >> processed_dataset >> feature_table >> trained_model >> registered_model
    registered_model >> deployment
    registered_model >> monitoring


dag = e2eml_pipeline()
