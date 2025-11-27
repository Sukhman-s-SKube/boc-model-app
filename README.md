# BoC Model Serve

FastAPI microservice for serving Bank of Canada policy rate predictions from the latest XGBoost model snapshot stored in S3-compatible object storage. The service streams macroeconomic features from ClickHouse, featurizes rolling windows, exposes inference endpoints, and ships with everything needed to containerize, deploy, and automate the workflow via GitHub Actions.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [API](#api)
- [Docker](#docker)
- [Kubernetes Manifests](#kubernetes-manifests)
- [CI/CD Workflow](#cicd-workflow)
  - [Calling the Workflow](#calling-the-workflow)
  - [Inputs](#inputs)
  - [Secrets](#secrets)
  - [How the Workflow Works](#how-the-workflow-works)
  - [Example Invocation](#example-invocation)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## Features
- **Production-ready FastAPI app** that warms the model at startup and provides `/predict` and `/reload` endpoints.
- **Automated feature preparation**: pulls macro time series from ClickHouse, builds rolling windows, and applies normalization, clipping, and drift diagnostics.
- **Model lifecycle on S3**: fetches and caches the newest `boc_policy_classifier` booster stored under `MODELS_BUCKET`.
- **Container + Kubernetes ready**: lightweight Python 3.10 image (`Dockerfile`) and templated manifests in `k8s/` where `IMAGE_PLACEHOLDER` is patched by CI.
- **Reusable GitHub Actions pipeline** to build, push, and deploy via Harbor and any Kubernetes cluster.

## Architecture
```
Client → FastAPI router (/api/serve) → ModelService
        ↘ ClickHouse (macro features) → window builder → feature matrix
         ↘ S3 (models + metadata)     → XGBoost booster
```
- `app/server/app.py` wires the FastAPI application, CORS, and startup hook.
- `app/server/routes/serve.py` defines REST endpoints for health, prediction, and hot-reload.
- `app/server/services/model_service.py` orchestrates ClickHouse ingestion, feature engineering, inference, and drift stats.
- `app/server/utils/s3_util.py` configures the boto3 client so the latest model artifact can be fetched securely.

## Local Development
1. **Install prerequisites**: Python 3.10+, uvicorn, and access to your ClickHouse + S3 endpoints.
2. **Clone & create a virtual env**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Configure environment**: create a `.env` file with the variables listed below (or export them directly). `python-dotenv` auto-loads them.
4. **Run the server**
   ```bash
   uvicorn app.server.app:app --host 0.0.0.0 --port 8080 --reload
   ```
5. **Call the API**
   ```bash
   curl http://localhost:8080/api/serve/predict/2024-01-31
   ```

## Environment Variables
| Name | Required | Default | Purpose |
| --- | --- | --- | --- |
| `AWS_ENDPOINT_URL` | ✅ | — | Optional custom endpoint for S3-compatible storage (e.g., MinIO).
| `AWS_REGION` | ✅ | `us-east-1` | Region passed to the boto3 client.
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | ✅ | — | Credentials for downloading models.
| `MODELS_BUCKET` | ✅ | `models` | Bucket containing serialized XGBoost boosters.
| `S3_BUCKET` | ⛔️ | `dagster` | Included for compatibility if additional assets are stored; unused by default.
| `CH_HOST` | ✅ | `localhost` | ClickHouse host providing macro data.
| `CH_PORT` | ✅ | `9000` | ClickHouse TCP port.
| `CH_USER` / `CH_PASS` | ⛔️ | — | ClickHouse credentials (leave blank for default user).
| `CH_DB` | ✅ | `default` | Database to query for macro features.
| `CH_TABLE` | ✅ | `macro_daily` | Table supplying time-series signals.
| `CH_SECURE` | ✅ | `false` | Set to `true` when using TLS-enabled ClickHouse endpoints.
| `FEATURE_COLUMNS` | ✅ | `rate,cpi,y2,y5,y10,spread_2_10,oil,unemploy` | Ordered columns pulled from ClickHouse.
| `RATE_MOVE_TOL` | ⛔️ | `0.0125` | Threshold used when summarizing prediction tolerance.

`AWS_*` and `CH_*` values can be injected via Kubernetes secret `boc-model-app` or a local `.env` file.

## API
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Root heartbeat returning `{"Message": "Server is working"}`. |
| `GET` | `/api/serve/test` | Router-level health probe. |
| `GET` | `/api/serve/predict/{date}` | Scores the specified ISO date (or the most recent available) and returns class probabilities, drift stats, and metadata. |
| `POST` | `/api/serve/reload` | Forces a reload of the most recent model artifact from S3. |

Use the automatically generated Swagger UI at `http://localhost:8080/docs` for interactive testing.

## Docker
Build and run the container image defined in `Dockerfile`:

```bash
docker build -t boc-model-app:dev .
docker run --rm -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=xxx -e AWS_SECRET_ACCESS_KEY=yyy \
  -e MODELS_BUCKET=models -e CH_HOST=clickhouse \
  boc-model-app:dev
```

The image installs system build tools for `xgboost`, copies `app/`, and starts Uvicorn on port 8080.

## Kubernetes Manifests
The `k8s/` directory contains a simple deployment stack:
- `1-namespace.yaml` – isolates the `boc-model-app` namespace.
- `2-sealed-secrets.yaml` – stores ClickHouse & AWS credentials.
- `3-deployment.yaml` – single-replica Deployment that references `IMAGE_PLACEHOLDER` and your secret.
- `4-service.yaml` – `LoadBalancer` service exposing port 80 → 8080.

Apply them manually once the image is published (replace `IMAGE_PLACEHOLDER`):

```bash
kubectl apply -f k8s/1-namespace.yaml
kubectl apply -f k8s/2-sealed-secrets.yaml
kubectl set image deployment/boc-model-app boc-model-app=<registry>/boc-model-app:<tag> -n boc-model-app
kubectl apply -f k8s/4-service.yaml
```

## CI/CD Workflow
This repository is deployed through a reusable workflow housed in [`Sukhman-s-SKube/gh-actions`](https://github.com/Sukhman-s-SKube/gh-actions). The local workflow (`.github/workflows/pipeline.yaml`) builds the Docker image, pushes it to Harbor, and rolls out the updated manifests.

### Calling the Workflow
Create or update a workflow in your repo that calls the reusable action:

```yaml
name: CI/CD

on:
  push:
    branches:
      - main

jobs:
  deploy:
    uses: Sukhman-s-SKube/gh-actions/.github/workflows/build-deploy.yaml@main
    with:
      IMAGE_NAME:    boc-model-app/boc-model-app
      DOCKERFILE_PATH: ./Dockerfile
      MANIFEST_PATH:  ./k8s
      KUBE_NAMESPACE: boc-model-app
    secrets:
      HARBOR_URL:      ${{ secrets.HARBOR_URL }}
      HARBOR_USERNAME: ${{ secrets.HARBOR_USERNAME }}
      HARBOR_PASSWORD: ${{ secrets.HARBOR_PASSWORD }}
      KUBE_CONFIG:     ${{ secrets.KUBECONFIG }}
```

### Inputs
| Name | Required | Default | Description |
| --- | --- | --- | --- |
| `IMAGE_NAME` | ✅ | — | Repository/name for the image in Harbor (supports nested paths). |
| `DOCKERFILE_PATH` | ⛔️ | `./Dockerfile` | Location of the Dockerfile to build. |
| `MANIFEST_PATH` | ✅ | — | File or directory containing Kubernetes YAML (supports substitution of `IMAGE_PLACEHOLDER`). |
| `KUBE_NAMESPACE` | ⛔️ | `default` | Namespace passed to `kubectl apply`. |
| `EXTRA_BUILD_ARGS` | ⛔️ | — | Additional `--build-arg` key-value pairs forwarded to Docker Buildx. |

### Secrets
| Name | Required | Description |
| --- | --- | --- |
| `HARBOR_URL` | ✅ | Base URL of your Harbor registry (supports HTTP with `--tls-verify=false`). |
| `HARBOR_USERNAME` / `HARBOR_PASSWORD` | ✅ | Credentials or robot token for pushing to Harbor. |
| `KUBECONFIG` | ✅ | Base64-encoded kubeconfig granting access to the target cluster. |

### How the Workflow Works
1. Checks out the caller repository.
2. Sets up Docker Buildx on the runner.
3. Logs in to Harbor (HTTP registries supported with TLS verify disabled).
4. Builds the `linux/amd64` image and tags it with both `latest` and the commit SHA.
5. Pushes the image to Harbor.
6. Writes the decoded `KUBECONFIG` to disk and configures kubectl.
7. Replaces `IMAGE_PLACEHOLDER` inside the manifests with the pushed image reference.
8. Applies every YAML in `MANIFEST_PATH` to the chosen namespace and waits for rollout success.

### Example Invocation
Trigger the workflow manually (`workflow_dispatch`) and deploy a production namespace:

```yaml
name: Release

on:
  workflow_dispatch:

jobs:
  release:
    uses: Sukhman-s-SKube/gh-actions/.github/workflows/build-deploy.yaml@main
    with:
      IMAGE_NAME:     boc-model-app/boc-model-app
      MANIFEST_PATH:  ./k8s
      KUBE_NAMESPACE: production
      EXTRA_BUILD_ARGS: ARG1=foo ARG2=bar
    secrets:
      HARBOR_URL:      ${{ secrets.HARBOR_URL }}
      HARBOR_USERNAME: ${{ secrets.HARBOR_USERNAME }}
      HARBOR_PASSWORD: ${{ secrets.HARBOR_PASSWORD }}
      KUBECONFIG:      ${{ secrets.KUBECONFIG }}
```

## Requirements
- Python 3.10+
- Access to a ClickHouse cluster populated with the macro features referenced by `FEATURE_COLUMNS`.
- Access to an S3-compatible bucket containing serialized XGBoost boosters + metadata JSON.
- Docker & Buildx (for local builds or CI runners).
- `kubectl` configured for the destination cluster.

## Troubleshooting
- **Prediction returns 500** – ensure ClickHouse credentials are set and that enough history exists to build a `seq_len`-day window.
- **Model not found** – confirm the models bucket contains keys prefixed with `boc_policy_classifier` and that the GitHub runner/Pod can reach S3.
- **Drift stats missing** – expected if the stored metadata lacks `feature_stats`; retrain and upload with the new metadata payload.
- **Kubernetes deploy stuck** – run `kubectl describe deployment boc-model-app -n boc-model-app` to confirm image pull secrets and environment variables are correct.

Enjoy shipping new rate forecasts automatically! 🎯
