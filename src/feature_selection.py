import pandas as pd


def get_bin_columns(dataframe):
    """
    Get the binary and non-binary columns from a dataframe.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
    
    Returns:
        tuple: A tuple containing two lists - binary columns and non-binary columns.
    
    """
    columnas_binarias = []
    columnas_no_binarias = []
    
    for columna in dataframe.columns:
        valores_unicos = dataframe[columna].dropna().unique()
        
        if len(valores_unicos) == 2:
            columnas_binarias.append(columna)
        else:
            columnas_no_binarias.append(columna)
    
    return columnas_binarias, columnas_no_binarias



def fill_with_encoding(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Fill specified columns in a dataframe with encoding values.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        columns (list): List of column names to fill.
    
    Returns:
        pd.DataFrame: The modified dataframe with filled columns.
    
    """
    dataframe[columns] = dataframe[columns].replace(0, -1)
    
    dataframe[columns] = dataframe[columns].fillna(0)
    
    return dataframe[columns]



def fill_with_mean(dataframe: pd.DataFrame, columns: list):
    """
    Fill specified columns in a dataframe with their mean values.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        columns (list): List of column names to fill.
    
    Returns:
        pd.DataFrame: The modified dataframe with filled columns.
    """
    for column in columns:
        mean = dataframe[column].mean()
        dataframe[column].fillna(mean, inplace=True)
    
    return dataframe[columns]



def fast_fill(dataframe: pd.DataFrame):
    """
    Perform fast filling of binary and non-binary columns in a dataframe.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The modified dataframe after fast filling.
    """
    binary_cols, non_binary_cols = get_bin_columns(dataframe)
    dataframe[binary_cols] = fill_with_encoding(dataframe, binary_cols)
    dataframe[non_binary_cols] = fill_with_mean(dataframe, non_binary_cols)
    
    return dataframe