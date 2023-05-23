import numpy as np
import pandas as pd


def get_bin_columns(dataframe):
    columnas_binarias = []
    columnas_no_binarias = []

    for columna in dataframe.columns:
        valores_unicos = np.unique(dataframe[columna])
        
        if len(valores_unicos) == 2 and np.array_equal(valores_unicos, [0, 1]):
            columnas_binarias.append(columna)
        else:
            columnas_no_binarias.append(columna)
    
    return columnas_binarias, columnas_no_binarias




def fill_with_mode(dataframe: pd.DataFrame, columns: list):
    for column in columns:
        most_frequent_value = dataframe[column].mode()[0]
        dataframe[column].fillna(most_frequent_value, inplace=True)
    
    return dataframe


def fill_with_encoding(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    # Reemplazar los 0 por -1
    dataframe.replace(0, -1, inplace=True)
    
    # Reemplazar los NaN por 0
    dataframe.fillna(0, inplace=True)
    
    return dataframe


def fill_with_mean(dataframe: pd.DataFrame, columns: list):
    for column in columns:
        mean = dataframe[column].mean()
        dataframe[column].fillna(mean, inplace=True)
    
    return dataframe


def fast_fill(dataframe: pd.DataFrame):
    binary_cols, non_binary_cols = get_bin_columns(dataframe)
    dataframe = fill_with_mode(dataframe, binary_cols)
    dataframe = fill_with_mean(dataframe, non_binary_cols)
    
    return dataframe


def fast_fill_2(dataframe: pd.DataFrame):
    binary_cols, non_binary_cols = get_bin_columns(dataframe)
    dataframe = fill_with_encoding(dataframe, binary_cols)
    dataframe = fill_with_mean(dataframe, non_binary_cols)
    
    return dataframe


def get_corr_columns(dataframe: pd.DataFrame, column: str, x: bool=False):
    correlations = dataframe.corr()[column].abs().sort_values(ascending=x)
    correlated_columns = correlations.index[1:25].tolist()
    
    return correlated_columns