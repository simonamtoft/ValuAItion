import os
import json
import torch
import click
import joblib
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from src.data.process import get_data
from src.model.utils import get_device


@click.command()
@click.option('--run_id', help='Run ID of trained model.')
@click.option('--submit', is_flag=True,
              help='Whether to submit test predictions to endpoint.')
def main(run_id: str, submit: bool):
    model_dir = os.path.join('models', run_id)
    save_path = os.path.join(model_dir, 'df_test.csv')
    if os.path.exists(save_path):
        logging.info('Loading previously computed predictions.')
        df_test = pd.read_csv(save_path)
    else:
        logging.info('Compute predictions.')
        df_test = compute_metrics(model_dir)
        df_test.to_csv(save_path)

    if submit:
        predictions = df_test[["TRANSACTION_ID", "PRICE"]].to_dict("records")
        upload_results(predictions, model_dir)


def compute_metrics(model_dir: str):
    device = get_device()

    # load test data
    df_test = get_data('test')
    if 'PRICE' in df_test.columns:
        X_test = df_test.drop('PRICE', axis=1)

    # load feature names used for training
    with open(os.path.join(model_dir, 'train_features.txt'), 'r') as f:
        train_features = [x.strip() for x in f.readlines()]

    # add missing features
    missing_features = [x for x in train_features if x not in X_test.columns]
    for feature in missing_features:
        X_test.loc[:, feature] = 0

    # remove extra features and reorder
    X_test = X_test[train_features]

    # scale features if applicable for model
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)

    # move data to device
    X_test = torch.from_numpy(X_test.values).to(device)

    # load model
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))

    # predict
    df_test = df_test.reset_index(drop=False)
    df_test['PRICE'] = model.predict(X_test)
    return df_test


def upload_results(predictions: dict, model_dir: str):
    # load name and email from .env file
    load_dotenv(override=True)
    submit_name = os.getenv('SUBMIT_NAME', None)
    submit_email = os.getenv('SUBMIT_EMAIL', None)
    assert submit_name is not None, 'Please set SUBMIT_NAME in .env file.'
    assert submit_email is not None, 'Please set SUBMIT_EMAIL in .env file.'

    logging.info('Submitting results to API with name: %s and email: %s',
                 submit_name, submit_email)

    r = requests.post(
        url="https://api.resights.dk/hackathon/avm/ejerlejligheder/v1",
        json={
            "name": submit_name,
            "email": submit_email,
            "predictions": predictions,
        })

    with open(os.path.join(model_dir, 'upload_result.json'), 'w') as f:
        json.dump(r.json(), f, indent=4)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main()
