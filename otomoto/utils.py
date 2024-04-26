import os
import pandas as pd
from sqlalchemy import create_engine
from typing import Dict


def get_unique_values() -> Dict:
    table_name = 'otomoto_data_cleaned'
    db_connection_string = os.getenv('DB_CONNECTION_STRING')

    engine = create_engine(db_connection_string)
    df = pd.read_sql_table(table_name, engine)

    cols_to_include = ['Vehicle brand', 'Vehicle model', 'Vehicle Version',
        'Fuel Type','Transmission Type', 'Drive Type', 'Car Body Type',
        'Number of doors', 'Number of seats', 'Color', 'Country of origin',
        'Vehicle Generation', 'New/Used']
    
    df = df[cols_to_include]

    unique_values_dict = {col: df[col].dropna().unique().tolist() for col in df.columns}

    return unique_values_dict