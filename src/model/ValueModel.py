import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from src.data.process import process_data


class ValueModel:
    def __init__(self, run_id: str):
        self.run_id = run_id
        model_dir = os.path.join('models', run_id)

        # load feature names used for training
        with open(os.path.join(model_dir, 'train_features.txt'), 'r') as f:
            self.train_features = [x.strip() for x in f.readlines()]

        # load data config
        with open('data_config.json', 'r') as f:
            self.data_config = json.load(f)

        # load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        # load model
        self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))

    def predict(self, df: pd.DataFrame):
        df = process_data(df, 'test', self.run_id, **self.data_config)

        # add missing features
        missing_features = [x for x in self.train_features if x not in df.columns]
        for feature in missing_features:
            df.loc[:, feature] = 0

        # remove extra features and reorder
        df = df[self.train_features]

        # scale features if applicable for model
        if self.scaler is not None:
            df = self.scaler.transform(df)

        # move data to device
        x = torch.from_numpy(df.values)

        # predict
        y_hat = self.model.predict(x)
        if self.data_config['log_y']:
            y_hat = np.exp(y_hat)

        return y_hat
