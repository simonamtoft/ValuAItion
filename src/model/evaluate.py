import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv


def compute_metrics():
    # TODO:
    # 1. Load the model from models/model.pkl
    # 2. Load the test data from models/test.parquet
    # 3. Predict the prices on the test data
    df_test = pd.DataFrame()
    # 4. Upload the results to the API
    upload_results(df_test)


def upload_results(df_test: pd.DataFrame):
    # load name and email from .env file
    load_dotenv(override=True)
    submit_name = os.environ['SUBMIT_NAME']
    submit_email = os.environ['SUBMIT_EMAIL']
    assert submit_name is not None, 'Please set SUBMIT_NAME in .env file.'
    assert submit_email is not None, 'Please set SUBMIT_EMAIL in .env file.'

    predictions = df_test[["TRANSACTION_ID", "PRICE"]].to_dict("records")
    r = requests.post(
        url="https://api.resights.dk/hackathon/avm/ejerlejligheder/v1",
        json={
            "name": submit_name,
            "email": submit_email,
            "predictions": predictions,
        })

    print(r.json())
    with open('models/upload_result.json', 'w') as f:
        json.dump(r.json(), f, indent=4)
