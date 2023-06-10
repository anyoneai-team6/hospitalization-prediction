import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
#smote

def find_column_name(df,x):
    columnas_elegibles = []
    for columna in df.columns:
        if x in columna:
            columnas_elegibles.append(columna)
    
    return columnas_elegibles


def find_name_list(l,x):
    columnas_elegibles = []
    for columna in l:
        if x in columna:
            columnas_elegibles.append(columna)
    
    return columnas_elegibles


def find_with_null_col(df, x):
    columnas_con_nulos = []
    nulos_por_columna = df.isnull().sum().sort_values(ascending=False)
    
    for columna, cantidad in nulos_por_columna.items():
        if cantidad > x:
            columnas_con_nulos.append(columna)
    
    return columnas_con_nulos


def fill_column(df, x, strategy='most_frequent'):
    imputer = SimpleImputer(strategy=strategy)
    print(df[x].value_counts())
    print(f'nulls: {df[x].isnull().sum()}')
    df[[x]] = imputer.fit_transform(df[[x]])  
    
    return df[x]


def values_columns(df, columnas_elegibles):
    
    for column in columnas_elegibles:
        nan_percentage = np.mean(pd.isnull(df[column])) * 100
        nan_number = df[column].isnull().sum()
        zero_percentage = np.mean(df[column] == 0) * 100
        zero_number = (df[column] == 0).sum()
        one_percentage = np.mean(df[column] == 1) * 100
        one_number = (df[column] == 1).sum()
        
        print(f"{column}:")
        print(f"Number of NaN values: {nan_number}")
        print(f"Percentage of NaN values: {round(nan_percentage,2)}%")
        print(f"Number of 0s: {zero_number}")
        print(f"Percentage of 0s: {round(zero_percentage,2)}%")
        print(f"Number of 1s: {one_number}")
        print(f"Percentage of 1s: {round(one_percentage,2)}%")
        print("")


def null_per_column(df, x):
    print(f'{x} nulls: {df[x].isnull().sum()}')


def contar_nulos(dataframe):
    nulos_por_columna = dataframe.isnull().sum()
    return nulos_por_columna


def fill_binary_with_knn(dataframe, columns):
    for column in columns:
        # Separar los datos con valores no nulos y nulos en la columna actual
        not_null_data = dataframe[dataframe[column].notnull()]
        null_data = dataframe[dataframe[column].isnull()]
        
        # Dividir los datos en características (X) y el valor objetivo (y)
        X = not_null_data.drop(column, axis=1)
        y = not_null_data[column]
        
        # Crear un clasificador k-NN
        knn = KNeighborsClassifier(n_neighbors=1)
        
        # Entrenar el clasificador k-NN
        knn.fit(X, y)
        
        # Predecir los valores faltantes en la columna actual
        predicted_values = knn.predict(null_data.drop(column, axis=1))
        
        # Asignar los valores predichos a los valores faltantes en la columna actual
        dataframe.loc[dataframe[column].isnull(), column] = predicted_values
    
    return dataframe
