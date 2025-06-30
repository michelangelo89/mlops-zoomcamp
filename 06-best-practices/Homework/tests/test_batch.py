import pandas as pd
from datetime import datetime
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),         # invalid: null locations
        (1, 1, dt(1, 2), dt(1, 10)),               # valid: 8 minutes
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),      # valid: 59 seconds
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),          # invalid: over 60 min
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    assert len(actual_df) == 2
    assert actual_df['duration'].between(1, 60).all()