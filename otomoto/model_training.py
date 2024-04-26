from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import mlflow
import mlflow.sklearn
import datetime
from warnings import filterwarnings
import os
filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, df, target_column):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        self.df = df
        self.target_column = target_column
        self.models = [('RandomForestRegressor', RandomForestRegressor()), 
                       ('XGBRegressor', XGBRegressor()), 
                       ('LGBMRegressor', LGBMRegressor())]
        self.parameters = {'n_estimators': [50, 100, 200], 
                           'max_depth': [6, 10, 15], 
                           'min_samples_split': [2, 5, 10]}
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def preprocess(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("otomoto_price_predictor")

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        with mlflow.start_run() as encoder_run:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 
            mlflow.set_tag("mlflow.runName", run_name + '_encoder')
            self.encoder.fit(X)
            mlflow.sklearn.log_model(self.encoder, "otomoto_data_encoder")
            mlflow.register_model("runs:/{}/otomoto_data_encoder".format(mlflow.active_run().info.run_id), "otomoto_data_encoder")
        mlflow.end_run()

        X_encoded = self.encoder.transform(X).toarray()
        y = y.values

        self.train, self.test, self.y_train, self.y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
        self.valid, self.test, self.y_valid, self.y_test = train_test_split(self.test, self.y_test, test_size=0.33, random_state=42)

    def train_models(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("otomoto_price_predictor")

        for model_name, model in self.models:
            with mlflow.start_run() as run:
                run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")    
                mlflow.set_tag("mlflow.runName", run_name)

                cv = KFold(n_splits=5, random_state=42, shuffle=True)
                grid_cv = GridSearchCV(model, self.parameters, cv=cv)
                model_name = model_name + '_otomoto_price_predictor'

                grid_cv.fit(self.train, self.y_train)

                for name, dataset, y_true in [('Train', self.train, self.y_train), ('Validation', self.valid, self.y_valid), ('Test', self.test, self.y_test)]:
                    pred = grid_cv.predict(dataset)
                    r2 = r2_score(y_true, pred)
                    mae = mean_absolute_error(y_true, pred)
                    rmse = np.sqrt(mean_squared_error(y_true, pred))
                    mlflow.log_metric(f'{model_name} {name} R2', r2)
                    mlflow.log_metric(f'{model_name} {name} MAE', mae)
                    mlflow.log_metric(f'{model_name} {name} RMSE', rmse)

                params = {f"{model_name}_{k}": v for k, v in grid_cv.best_params_.items()}
                mlflow.log_params(params)

                model_info = mlflow.sklearn.log_model(grid_cv.best_estimator_, "model")
                model_uri = "runs:/" + run.info.run_id + "/" + model_info.artifact_path
                mlflow.register_model(model_uri, f"{model_name}")