#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import numpy as np
import boto3


# Accept year and month from CLI
year = int(sys.argv[1])
month = int(sys.argv[2])

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Load input data for given year/month
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(input_file)

# Prepare features
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Print mean predicted duration (Q5)
print("Mean predicted duration:", round(y_pred.mean(), 2))

# Save output file (optional for Q2)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = f'output_{year:04d}_{month:02d}.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


def upload_to_s3(local_file, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, s3_key)
    print(f"✅ Uploaded to s3://{bucket}/{s3_key}")


output_file = f"{year:04d}-{month:02d}_predicted.parquet"
df_result.to_parquet(output_file, engine='pyarrow', index=False)

upload_to_s3(output_file, 'mlflow-models-falcon09099', f'homework-output/{output_file}')
