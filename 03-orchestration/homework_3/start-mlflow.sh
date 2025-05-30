#!/bin/bash

cd "$(dirname "$0")/mlflow_data"

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 \
  --port 5000
