import os
import json
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


def main():
    # load training data
    df_train = pd.read_csv(os.path.join('data', 'processed', 'train.parquet'))

    # load model configuration
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)

    # split into features and labels
    X = df_train.drop('PRICE', axis=1)
    y = df_train['PRICE'].values

    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, shuffle=True, test_size=0.1)

    # define model
    model = XGBRegressor(**model_config)

    # fit on training data
    model = model.fit(X_train, y_train)

    # save model
    joblib.dump(model, os.path.join('models', 'model.joblib'))


if __name__ == '__main__':
    main()
