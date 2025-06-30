#!/bin/bash

echo "🔧 Setting up environment..."

export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL=http://localhost:4566

echo "🪣 Creating S3 bucket if not exists..."
aws --endpoint-url=$S3_ENDPOINT_URL s3 mb s3://nyc-duration || true

echo "🧪 Running integration test..."
python tests/integration_test.py
