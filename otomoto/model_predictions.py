import pandas as pd
import mlflow.sklearn
import os


class Predictor:
    def __init__(self, model_name, transformer_name):
        self.model_name = model_name
        self.transformer_name = transformer_name
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))

    def load_model(self):
        model_version = self.mlflow_client.get_latest_versions(self.model_name)[0]
        self.model = mlflow.sklearn.load_model(model_uri=f"models:/{self.model_name}/{model_version.version}")

    def load_transformer(self):
        transformer_version = self.mlflow_client.get_latest_versions(self.transformer_name)[0]
        self.transformer = mlflow.sklearn.load_model(model_uri=f"models:/{self.transformer_name}/{transformer_version.version}")

    def load_models(self):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment("otomoto_price_predictor")
        self.load_model()
        self.load_transformer()

    def predict(self, data_dict):
        self.load_models()
        data_df = pd.DataFrame([data_dict])
        data_transformed = self.transformer.transform(data_df)
        prediction = self.model.predict(data_transformed)

        return prediction[0]