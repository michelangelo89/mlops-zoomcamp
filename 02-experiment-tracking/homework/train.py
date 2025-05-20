import os
import pickle
import click
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # ✅ Connect to tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-homework")

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # ✅ Start manual tracking
    with mlflow.start_run():

        # ✅ Set tag and log params manually
        mlflow.set_tag("developer", "michelangelo")
        mlflow.set_tag("model_type", "RandomForest")

        params = {
            "max_depth": 10,
            "random_state": 0
        }
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse", rmse)

        # ✅ Save the model manually
        model_path = os.path.join(data_path, "rf_model.pkl")
        with open(model_path, "wb") as f_out:
            pickle.dump(rf, f_out)

        mlflow.log_artifact(model_path, artifact_path="models")

        print(f"✅ RMSE: {rmse:.3f}")


if __name__ == '__main__':
    run_train()