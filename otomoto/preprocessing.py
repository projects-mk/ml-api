
from sqlalchemy import create_engine
import os
import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import numpy as np
import logging
from typing import Union
from sqlalchemy import create_engine
from typing import Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessing:
    def __init__(self):
        self.engine = create_engine(os.getenv('DB_CONNECTION_STRING'))
        self.df = pd.read_sql_table('otomoto_data', con=self.engine)\
            .drop(columns=['index','Oblicz','Kup ten pojazd na ratyOblicz','Pokaż oferty z numerem VIN','Oferta od','Ma numer rejestracyjny'])\
            .drop_duplicates()

    @staticmethod
    def _remove_na_columns(df: pd.DataFrame) -> pd.DataFrame:
        n_total = len(df)
        for col in df.columns:
            n_nulls = len(df.loc[df[col].isna()])
            if n_nulls/n_total > 0.35:
                df = df.drop(columns=[col])
        return df


    @staticmethod
    def _drop_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
        columns_to_check = ['Cena', 'Przebieg', 'Pojemność skokowa', 'Moc', 'Spalanie W Mieście']
        existing_columns = [col for col in columns_to_check if col in df.columns]
        df = df.dropna(how='any', subset=existing_columns)        
        return df


    @staticmethod
    def _keep_selected_columns(df: pd.DataFrame) -> pd.DataFrame:
        columns_to_keep = ['Cena', 'Marka pojazdu', 'Model pojazdu', 'Wersja', 'Generacja',
            'Rok produkcji', 'Przebieg', 'Pojemność skokowa', 'Rodzaj paliwa',
            'Moc', 'Skrzynia biegów', 'Napęd', 'Spalanie W Mieście', 'Typ nadwozia','Liczba drzwi',
            'Liczba miejsc', 'Kolor', 'Kraj pochodzenia','Stan', 'Uszkodzony']
        return df[columns_to_keep]
    

    @staticmethod
    def _convert_to_type(x: Any, type: Union[int, float, str]) -> Union[int, float, str]:
        try:
            return type(x)
        except Exception:
            return np.nan


    def _convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Uszkodzony'].fillna('No', inplace=True)
        df['Uszkodzony'].replace('Tak', 'Yes', inplace=True)
        df['Uszkodzony'].replace('Nie', 'No', inplace=True)

        df['Rok produkcji'] = df['Rok produkcji'].apply(lambda x: self._convert_to_type(x, int))
        df['Przebieg'] = df['Przebieg'].astype(str).str.replace(' km', '').str.replace(' ', '').astype(float)
        df['Pojemność skokowa'] = df['Pojemność skokowa'].str.replace(' cm3', '').str.replace(' ', '').astype(float)
        df['Moc'] = df['Moc'].str.replace(' KM', '').str.replace(' ', '').astype(float)
        df['Spalanie W Mieście'] = df['Spalanie W Mieście'].str.replace(' l/100km', '').str.replace(',', '.').str.replace(' ', '').astype(float)
        df['Liczba drzwi'] = df['Liczba drzwi'].apply(lambda x: self._convert_to_type(x, int))
        df['Liczba miejsc'] = df['Liczba miejsc'].apply(lambda x: self._convert_to_type(x, int))
        return df

    @staticmethod
    def _translate_to_eng(df: pd.DataFrame) -> pd.DataFrame:
        translator = Translator()
        translator = Translator(service_urls=[
            'translate.google.com',
            'translate.google.pl',
            ])

        columns_to_translate = ['Rodzaj paliwa', 'Skrzynia biegów', 'Napęd', 'Typ nadwozia', 'Kolor', 'Kraj pochodzenia', 'Stan']
        logger.info("Translating column contents...")
        for col in tqdm(columns_to_translate):
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                translations = {}
                for value in unique_values:
                    try:
                        translations[value] = translator.translate(value).text
                    except AttributeError:
                        translations[value] = value
                df[col] = df[col].map(translations)

        df.columns = ['Price', 'Vehicle brand', 'Vehicle model', 'Vehicle Version', 'Vehicle Generation',
            'Year of production', 'Mileage', 'Engine Capacity', 'Fuel Type', 'Horse Power',
            'Transmission Type', 'Drive Type', 'Gas Usage per 100km', 'Car Body Type',
            'Number of doors', 'Number of seats', 'Color', 'Country of origin',
            'New/Used', 'Damaged']

        return df


    def preprocess_data(self) -> pd.DataFrame:
        # self.df = self._remove_na_columns(self.df)
        self.df = self._keep_selected_columns(self.df)
        self.df = self._drop_rows_with_nulls(self.df)
        self.df = self._convert_datatypes(self.df)
        self.df = self._translate_to_eng(self.df)
        
        self.save_to_sql('otomoto_data_cleaned', os.getenv('DB_CONNECTION_STRING'))
        return self.df


    def get_data(self) -> pd.DataFrame:

        self.df.columns = [
            'price', 'vehicle_brand', 'vehicle_model', 'vehicle_version',
            'vehicle_generation', 'year_of_production', 'mileage',
            'engine_capacity', 'fuel_type', 'horse_power', 'transmission_type',
            'drive_type', 'gas_usage_per_100km', 'car_body_type', 'number_of_doors',
            'number_of_seats', 'color', 'country_of_origin', 'new_used', 'damaged'
        ]

        return self.df


    def save_to_sql(self, table_name: str, sql_engine: str):
        engine = create_engine(sql_engine)
        self.df.to_sql(table_name, engine, if_exists='replace', index=False)