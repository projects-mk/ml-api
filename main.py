from fastapi import FastAPI
import pandas as pd
from otomoto.model_predictions import Predictor
from otomoto.input_models import OtomotoInputData
from fastapi import Body
from utils import generate_conn_string
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title=os.getenv("APP_NAME"), version=os.getenv("APP_VERSION"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/otomoto/", tags=["otomoto"])
async def otomoto_marka():
    try:
        marka = "SELECT DISTINCT marka FROM otomoto.preprocessed"
        options = sorted(
            pd.read_sql_query(marka, con=generate_conn_string("projects"))[
                "marka"
            ].values
        )
        return {"response": options}

    except Exception as e:
        return {"response": str(e)}


@app.get("/api/v1/otomoto/{marka}", tags=["otomoto"])
async def otomoto_dropdown_values(marka: str):
    try:
        model = (
            f"SELECT DISTINCT model FROM otomoto.preprocessed WHERE marka = '{marka}'"
        )
        model = sorted(
            pd.read_sql_query(model, con=generate_conn_string("projects"))[
                "model"
            ].values
        )

        rodzaj_paliwa = "SELECT DISTINCT rodzaj_paliwa FROM otomoto.preprocessed"
        rodzaj_paliwa = sorted(
            pd.read_sql_query(rodzaj_paliwa, con=generate_conn_string("projects"))[
                "rodzaj_paliwa"
            ].values
        )

        skrzynia_biegow = "SELECT DISTINCT skrzynia_biegow FROM otomoto.preprocessed"
        skrzynia_biegow = sorted(
            pd.read_sql_query(skrzynia_biegow, con=generate_conn_string("projects"))[
                "skrzynia_biegow"
            ]
            .dropna()
            .values
        )

        naped = "SELECT DISTINCT naped FROM otomoto.preprocessed"
        naped = sorted(
            pd.read_sql_query(naped, con=generate_conn_string("projects"))["naped"]
            .dropna()
            .values
        )

        nadwozie = "SELECT DISTINCT nadwozie FROM otomoto.preprocessed"
        nadwozie = sorted(
            pd.read_sql_query(nadwozie, con=generate_conn_string("projects"))[
                "nadwozie"
            ]
            .dropna()
            .values
        )

        bezwypadkowy = "SELECT DISTINCT bezwypadkowy FROM otomoto.preprocessed"
        bezwypadkowy = sorted(
            pd.read_sql_query(bezwypadkowy, con=generate_conn_string("projects"))[
                "bezwypadkowy"
            ]
            .dropna()
            .values
        )

        serwisowany_w_aso = (
            "SELECT DISTINCT serwisowany_w_aso FROM otomoto.preprocessed"
        )
        serwisowany_w_aso = sorted(
            pd.read_sql_query(serwisowany_w_aso, con=generate_conn_string("projects"))[
                "serwisowany_w_aso"
            ]
            .dropna()
            .values
        )

        stan = "SELECT DISTINCT stan FROM otomoto.preprocessed"
        stan = sorted(
            pd.read_sql_query(stan, con=generate_conn_string("projects"))["stan"]
            .dropna()
            .values
        )

        return {
            "response": {
                "model": model,
                "rodzaj_paliwa": rodzaj_paliwa,
                "skrzynia_biegow": skrzynia_biegow,
                "naped": naped,
                "nadwozie": nadwozie,
                "bezwypadkowy": bezwypadkowy,
                "serwisowany_w_aso": serwisowany_w_aso,
                "stan": stan,
            }
        }

    except Exception as e:
        return {"response": str(e)}


@app.post("/api/v1/otomoto/predict", tags=["model_predictions"])
async def otomoto_predict(data: OtomotoInputData = Body(...)):

    try:
        transformer_name = "otomoto_car_price_predictor_data_encoder"
        model_name = "xgboost_otomoto_car_price_predictor_price_predictor"
        vars_to_ohe = [
            "marka",
            "model",
            "rodzaj_paliwa",
            "skrzynia_biegow",
            "naped",
            "nadwozie",
            "bezwypadkowy",
            "serwisowany_w_aso",
            "stan",
        ]

        predictor = Predictor(model_name=model_name, transformer_name=transformer_name)
        predictor.load_models()

        data = data.model_dump()
        response = predictor.predict(data, vars_to_ohe)

        return {"response": float(response)}

    except Exception as e:
        return {"response": str(e)}
