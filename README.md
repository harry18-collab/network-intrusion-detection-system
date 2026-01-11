# Network Security — Phishing Detection

AI-powered pipeline to ingest network/phishing datasets, validate & transform them, train ML models, and serve predictions via a FastAPI web UI.

## Contents
- Overview
- Requirements
- Environment variables
- Quickstart (local & Docker)
- Running the pipeline & API
- Project layout
- Docker on EC2 (commands)
- CI / Deployment notes
- Troubleshooting / Tips
- Contributing / License

## Overview
This project provides a full ML lifecycle:
- Data ingestion (MongoDB → DataFrame / CSV)
- Data validation and drift checks
- Feature transformation (imputation / preprocessing)
- Model training, evaluation and artifact persistence
- Web UI / API (FastAPI) to upload CSVs and get predictions
- Optional S3 / ECR sync utilities

## Requirements
- Python 3.8+
- pip, Docker (optional)
- MongoDB (or MongoDB URI)
- (Optional) AWS CLI for ECR / S3 operations

Install Python deps:
```sh
pip install -r requirements.txt
```

## Environment variables
Create a `.env` (or set CI secrets) with at least:
- MONGO_DB_URL (or MONGODB_URL used by ingestion)
- MONGODB_URL_KEY (if used by app)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (example: us-east-1)
- AWS_ECR_LOGIN_URI (ECR registry URI)
- ECR_REPOSITORY_NAME

## Quickstart — Local
1. Install deps:
```sh
pip install -r requirements.txt
```
2. Run web app:
```sh
python app.py
```
3. Run full training pipeline:
```sh
python main.py
# or programmatically
from networksecurity.pipeline.training_pipeline import TrainingPipeline
TrainingPipeline().run_pipeline()
```
4. Upload CSV → MongoDB:
```sh
python push_data.py
```

## API / Web Endpoints
- GET / → Home UI
- GET /predict-ui → Upload / predict UI
- POST /predict → Upload CSV and get predictions (saves to prediction_output/output.csv)
- GET /train → Trigger training pipeline
- GET /training-status → Check for trained artifacts
- GET /prediction_output/output.csv → Download predictions

(See app.py for exact routes and template files under templates/)

## Project layout (high level)
- app.py, main.py, push_data.py — entrypoints & web UI
- networksecurity/
  - pipeline/ — TrainingPipeline orchestration
  - components/ — data_ingestion, data_validation, data_transformation, model_trainer
  - utils/ — save/load helpers, model evaluation, estimator wrappers
  - cloud/ — s3_syncer, ECR helpers
  - logging/, exception/ — helpers
- data_schema/schema.yaml — expected dataset schema
- final_model/ — saved preprocessor & final model
- Artifacts/ — intermediate artifacts, reports, models
- prediction_output/ — outputs for user downloads
- templates/, static/ — web UI assets
- Dockerfile, requirements.txt, .github/workflows/ — CI

## Docker
Build and run:
```sh
docker build -t networksecurity .
docker run -p 8080:8080 networksecurity
```
Push to ECR: log in and push using AWS CLI (ensure AWS creds in env / CI).

## Docker Setup on EC2 (commands)
Run these on an Ubuntu EC2 (example):
```sh
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker $USER
newgrp docker
```
Then build/run your container as above. If pushing to ECR, authenticate with `aws ecr get-login-password` and `docker login`.

## CI / CD
- See .github/workflows for example pipeline (build, test, push to registry).
- Store AWS secrets (ACCESS_KEY, SECRET_KEY, ECR URI) in GitHub Secrets.

## Troubleshooting & Tips
- Verify MongoDB URI and collection names if ingestion fails.
- Check `final_model/` and `Artifacts/` for trained outputs.
- Use `test_mongodb.py` to validate DB connectivity.
- Ensure compatible Python package versions in requirements.txt.

## Contributing
- Fork, create a feature branch, add tests, open a PR.
- Follow existing code structure and docstrings.

## License
Include your preferred license (e.g., MIT) at project root.
