#!/usr/bin/env python
# coding: utf-8

# ### 🧭 ML Pipeline Overview – Module 3
# 
# | **Stage**                  | **Description**                                        | **Output**                    |
# |---------------------------|--------------------------------------------------------|-------------------------------|
# | 1. Download the data      | Ingest from S3, GCS, or URL                            | Raw data file(s)              |
# | 2. Transforming data      | Filter, clean, drop columns, remove outliers           | Cleaned DataFrame             |
# | 3. Preparing data for ML  | Feature engineering, vectorization, label split        | `X_train`, `y_train`, `X_val`, `y_val` |
# | 4. Hyperparameter tuning  | Hyperopt, Optuna, or GridSearch                        | Best model parameters         |
# | 5. Train the final model  | Retrain using best parameters                          | Fitted model                  |
# | 6. Model registry         | Log final model to MLflow Model Registry               | Versioned model entry         |

# # 📦 Imports

# In[18]:


import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from pathlib import Path


# In[4]:


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# Ensure 'models/' folder exists in current directory
models_folder = Path("models")
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df



df_train = read_dataframe(year=2021, month=1)   
df_val = read_dataframe(year=2021, month=2) 

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)  

    return X, dv


def train_model(X_train, y_train, X_val, y_val,dv):

    with mlflow.start_run() as run:
        
        mlflow.set_tag("developer", "Michelangelo")
        # Prepare DMatrix for XGBoost
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        
        # Use provided best hyperparameters
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }


        mlflow.log_params(best_params)

        # Train model
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        # Predict and log RMSE
        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        # Save DictVectorizer
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log the XGBoost model itself
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        # return run_id from MLflow
        return run.info.run_id


def run(year,month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    #train_model(X_train, y_train, X_val, y_val, dv)
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")   
    return run_id     

if __name__ == "__main__":
    # use argparse to get year and month from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train a duration prediction model.")
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    # run(year=args.year, month=args.month)
    run_id = run(year=args.year, month=args.month)   

    # save the run_id to a file
    with open("models/run_id.txt", "w") as f:
        f.write(run_id) 