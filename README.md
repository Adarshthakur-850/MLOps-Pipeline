# MLOps Pipeline

End-to-End MLOps project demonstrating code versioning, experiment tracking (MLflow), containerization (Docker), and CI/CD (GitHub Actions).

## Components
- **Training**: Scikit-learn Random Forest on Diabetes dataset.
- **Tracking**: MLflow for metrics, params, and model registry.
- **Serving**: FastAPI for real-time inference.
- **DevOps**: Docker & Docker Compose for orchestration.

## Setup & Run

### 1. Prerequisites
- Docker & Docker Compose installed.
- Python 3.9+ installed.

### 2. Run Locally (Docker Compose)
This starts both the MLflow server and the Inference API.

```bash
docker-compose up --build
```

- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
- **API**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Workflow
1.  **Train Model**:
    When the container starts or via manual trigger:
    ```bash
    python src/train.py
    ```
    (Ensure MLFLOW_TRACKING_URI is set correctly if running outside docker).

2.  **Inference**:
    Send a POST request to `http://localhost:8000/predict` with JSON body corresponding to the 10 input features.

## CI/CD
The `.github/workflows/mlops.yml` defines the pipeline that runs on every push to `main`.
