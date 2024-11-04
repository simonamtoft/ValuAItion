import os
import json
import torch
import joblib
import logging
import numpy as np
from uuid import uuid4
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.data.process import get_data
from src.model.utils import get_device


def main():
    device = get_device()

    # setup model run dir
    run_id = str(uuid4())
    model_dir = os.path.join('models', run_id)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # load data config
    with open('data_config.json', 'r') as f:
        data_config = json.load(f)

    # load training data
    df_train = get_data('train', run_id, **data_config)
    df_train = df_train.astype('float32')

    # load model and data configuration
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)
    with open('data_config.json', 'r') as f:
        data_config = json.load(f)

    # split into features and labels
    X = df_train.drop('PRICE', axis=1)
    y = df_train['PRICE'].values

    if data_config['log_y']:
        y = np.log(y)

    # store training features in list
    with open(os.path.join(model_dir, 'train_features.txt'), 'w') as f:
        f.writelines('\n'.join(X.columns.tolist()))

    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, shuffle=True, test_size=0.1)

    # move all arrays to device
    X_train = torch.from_numpy(X_train).to(device)
    X_val = torch.from_numpy(X_val).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_val = torch.from_numpy(y_val).to(device)

    # define model
    model = XGBRegressor(**model_config, device=device)

    # fit on training data
    model = model.fit(X_train, y_train)

    # compute scores
    model_scores = {
        'train_score': np.round(model.score(X_train, y_train), 5),
        'val_score': np.round(model.score(X_val, y_val), 5)
    }

    # save model along with other stuff
    logging.info('Saving run with ID %s', run_id)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(model_dir, 'scores.json'), 'w') as f:
        json.dump(model_scores, f, indent=4)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main()
