import os
import requests
from dotenv import load_dotenv

load_dotenv()


def generate_conn_string(db: str) -> str:

    url = os.environ["VAULT_URL"]
    token = os.environ["VAULT_TOKEN"]

    resp = requests.get(url, headers={"X-Vault-Token": token}).json()
    if not os.getenv("IS_TEST_ENV"):
        return resp["data"]["data"]["postgres"] + db

    return resp["data"]["data"]["postgres"] + "test_db"


def get_mlflow_uri() -> str:

    url = os.environ["VAULT_URL"]
    token = os.environ["VAULT_TOKEN"]

    resp = requests.get(url, headers={"X-Vault-Token": token}).json()

    return resp["data"]["data"]["mlflow_uri"]
