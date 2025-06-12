from prefect import task, flow
import pandas as pd
import pickle
import sys
import boto3
import numpy as np


@task
def load_model(path='model.bin'):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


@task
def read_data(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


@task
def run_inference(df, dv, model):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("Mean predicted duration:", round(np.mean(y_pred), 2))
    return y_pred


@task
def save_results(df, y_pred, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = f'{year:04d}-{month:02d}_predicted.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    return output_file


@task
def upload_to_s3(local_file, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, s3_key)
    print(f"✅ Uploaded to s3://{bucket}/{s3_key}")


@flow
def batch_predict_flow(year: int, month: int):
    dv, model = load_model()
    df = read_data(year, month)
    y_pred = run_inference(df, dv, model)
    output_file = save_results(df, y_pred, year, month)
    upload_to_s3(output_file, 'mlflow-models-falcon09099', f'homework-output/{output_file}')


# Local execution
if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    batch_predict_flow(year, month)
