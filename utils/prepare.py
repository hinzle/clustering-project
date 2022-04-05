# prepare.py

from utils.imports import *


def handle_missing_values(df, percent_missing_col, percent_missing_row):
    n_required_column = round(df.shape[0] * percent_missing_col)
    n_required_row = round(df.shape[1] * percent_missing_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df