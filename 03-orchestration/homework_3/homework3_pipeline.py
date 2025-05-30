#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load March 2023 Yellow Taxi data
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
df = pd.read_parquet(url)

# Print number of records
print(f"Number of rows: {len(df):,}")


# In[2]:


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

df_prepared = read_dataframe(url)

print(f"Number of rows after filtering: {len(df_prepared):,}")


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# Train/test data prep
train_dicts = df_prepared[['PULocationID', 'DOLocationID']].to_dict(orient='records')

# Vectorization
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df_prepared['duration'].values

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Print intercept
print(f"Intercept of the model: {round(lr.intercept_, 2)}")


# In[6]:


import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

# Predict on validation set
y_pred = lr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))

# Log params, metrics, and model
params = {"fit_intercept": True}

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        lr,
        artifact_path="model",
        registered_model_name="lin_reg_model"
    )


# In[8]:


import os

model_path = "mlflow_data/artifacts/0/1d8e684553f6410bb61e605ab497cd9b/artifacts/model/model.pkl"
size_bytes = os.path.getsize(model_path)

print(f"Model size: {size_bytes} bytes")


# In[ ]:




