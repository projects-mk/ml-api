import pandas as pd
import mlflow.sklearn
import os
from dotenv import load_dotenv
from utils import get_mlflow_uri

load_dotenv()


class Predictor:
    def __init__(self, model_name: str, transformer_name: str):
        self.model_name = model_name
        self.transformer_name = transformer_name
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=get_mlflow_uri())

    def load_model(self):
        model_version = self.mlflow_client.get_latest_versions(self.model_name)[0]
        self.model = mlflow.sklearn.load_model(
            model_uri=f"models:/{self.model_name}/{model_version.version}"
        )

    def load_transformer(self):
        transformer_version = self.mlflow_client.get_latest_versions(
            self.transformer_name
        )[0]
        self.transformer = mlflow.sklearn.load_model(
            model_uri=f"models:/{self.transformer_name}/{transformer_version.version}"
        )

    def load_models(self):
        mlflow.set_tracking_uri(get_mlflow_uri())
        self.load_model()
        self.load_transformer()

    def predict(self, input_data: dict, vars_to_ohe: list):
        self.load_models()

        input_data = pd.DataFrame([input_data])

        input_data_to_encode = input_data[vars_to_ohe]
        input_data_remaining = input_data.drop(columns=vars_to_ohe)

        input_data_encoded = self.transformer.transform(input_data_to_encode.values)
        input_data_encoded_df = pd.DataFrame(
            input_data_encoded.toarray(),
            columns=self.transformer.get_feature_names_out(vars_to_ohe),
        )

        input_data_encoded = pd.concat(
            [input_data_encoded_df, input_data_remaining.reset_index(drop=True)], axis=1
        )
        prediction = self.model.predict(input_data_encoded)

        return prediction[0]
