import json
import os
import pandas as pd
from .transforms import zip_code_mapper, municipality_code_mapper, \
    transform_floor, facility_clean
from .constants import AREA_COLUMNS, FACILITY_COLUMNS, DROP_COLUMNS, \
    CAT_COLUMNS


def get_data(split: str) -> pd.DataFrame:
    dataset_path = os.path.join(
        'datasets', f'Resights_Hackathon_Ejerlejligheder_{split.upper()}.csv')
    df = pd.read_csv(dataset_path, sep=',')
    df = transform_values(df, split)
    df = handle_missing_values(df)
    df = remove_unused_columns(df)

    # expand TRADE_DATE to year, month, and day of week
    trade_dates = pd.to_datetime(df['TRADE_DATE'])
    df['TRADE_YEAR'] = trade_dates.dt.year.values
    df['TRADE_MONTH'] = trade_dates.dt.month.values
    df['TRADE_DOW'] = trade_dates.dt.dayofweek.values
    df = df.drop(['TRADE_DATE'], axis=1)

    # one-hot encode columns
    df = pd.get_dummies(df, columns=CAT_COLUMNS, dtype='int8')

    # drop columns that are not for training anymore
    df = df.drop(['ZIP_CODE', 'MUNICIPALITY_CODE'], axis=1)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # impute CONSTRUCTION_YEAR and FLOOR with mean based on ZIP_AREA
    df = groupby_mean_impute(df, 'ZIP_AREA', 'CONSTRUCTION_YEAR')
    df = groupby_mean_impute(df, 'ZIP_AREA', 'FLOOR')

    # impute REBUILDING_YEAR to be same as CONSTRUCTION_YEAR if it is missing
    missing_rebuilding = df['REBUILDING_YEAR'].isna()
    df.loc[missing_rebuilding, 'REBUILDING_YEAR'] = \
        df.loc[missing_rebuilding, 'CONSTRUCTION_YEAR']

    # impute AREA columns with 0
    df[AREA_COLUMNS] = df[AREA_COLUMNS].fillna(0).values
    return df


def groupby_mean_impute(df: pd.DataFrame, groupby_col: str,
                        impute_col: str) -> pd.DataFrame:
    df[impute_col] = df[impute_col].astype(float)
    df[impute_col] = df.groupby(groupby_col)[impute_col].transform(
        lambda x: x.fillna(x.mean()))
    return df


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove some unused columns
    drop_cols = [x for x in DROP_COLUMNS if x in df.columns]
    df = df.drop(drop_cols, axis=1)

    # set 'TRANSACTION_ID' to be the index
    df = df.set_index('TRANSACTION_ID')
    return df


def transform_values(df: pd.DataFrame, split: str) -> pd.DataFrame:
    for col in FACILITY_COLUMNS:
        df[col] = df[col].apply(facility_clean)

    df['FLOOR'] = df['FLOOR'].apply(transform_floor)
    df['ZIP_AREA'] = df['ZIP_CODE'].apply(zip_code_mapper)
    df['MUNICIPALITY'] = df['MUNICIPALITY_CODE'].apply(
        municipality_code_mapper)
    df['HAS_ELEVATOR'] = df['HAS_ELEVATOR'].astype('float16')

    df = calculate_street_price_sqm(df, split)

    return df


def calculate_street_price_sqm(df: pd.DataFrame, split: str) -> pd.DataFrame:
    with open('data_config.json', 'r') as f:
        data_config = json.load(f)

    if data_config["calculate_street_price_sqm"]:
        # during train: calculate the average sqm price for each street code based off the entire dataset
        if split == 'train':
            street_mean_sqm_price = df.groupby("STREET_CODE")["SQM_PRICE"].mean()
            street_mean_sqm_price.to_csv("street_code_mean_sqm_price.csv", header=True)
        # during test: load the average sqm price for each street code from the train phase 
        else:
            street_mean_sqm_price = pd.read_csv("street_code_mean_sqm_price.csv", index_col="STREET_CODE").squeeze()
        df["STREET_CODE_MEAN_SQM_PRICE"] = df["STREET_CODE"].map(street_mean_sqm_price)
        df["STREET_CODE_MEAN_SQM_PRICE"] = df["STREET_CODE_MEAN_SQM_PRICE"].fillna(df["SQM_PRICE"].mean())
        
    return df
