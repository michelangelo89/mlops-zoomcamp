from datetime import datetime
import pandas as pd
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = os.getenv("INPUT_FILE_PATTERN").format(year=2023, month=1)
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
    }
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

print(f"✅ Correct test data written to {input_file}")




# Step 1: Run batch.py on the fake January 2023 input
print("🚀 Running batch.py...")
os.system("python batch.py 2023 1")

# Step 2: Load output from Localstack S3 and sum predicted durations
output_file = os.getenv("OUTPUT_FILE_PATTERN").format(year=2023, month=1)
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
    }
}

df_result = pd.read_parquet(output_file, storage_options=options)
total_duration = df_result["predicted_duration"].sum()

print("✅ Sum of predicted durations:", round(total_duration, 2))