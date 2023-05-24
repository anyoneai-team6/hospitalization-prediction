import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder,RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def apply_resample(X_train, y_train,v=1.35):
    # Concatenar X_train e y_train en un solo dataframe
    df_train = pd.concat([X_train, y_train], axis=1)
    
    # Separar la clase mayoritaria y la clase minoritaria
    class_majority = df_train[y_train == 0]
    class_minority = df_train[y_train == 1]
    
    n_samples=int(len(class_minority)*v)
    
    # Aplicar resample a la clase mayoritaria
    class_majority_resampled = resample(class_majority,
                                        replace=True,  # Permitir duplicados
                                        n_samples=n_samples,  # Igualar número de muestras de la clase minoritaria
                                        random_state=42)  # Reproducibilidad
    
    # Combinar la clase minoritaria con la clase mayoritaria remuestreada
    df_resampled = pd.concat([class_minority, class_majority_resampled])
    
    # Separar características (X_train_resampled) y etiquetas (y_train_resampled)
    X_train_resampled = df_resampled.drop(y_train.name, axis=1)
    y_train_resampled = df_resampled[y_train.name]
    
    return X_train_resampled, y_train_resampled


def preprocess_no_bin(X_train, X_test, strategy="knn"):
    if strategy == "simple":
        si = SimpleImputer(strategy='median')
        si.fit(X_test)
        X_test = si.transform(X_test)
        X_train = si.transform(X_train)
    elif strategy == "knn":
        knn = KNNImputer(n_neighbors=8)
        knn.fit(X_test)
        X_test = knn.transform(X_test)
        X_train = knn.transform(X_train)


def apply_standard_scaler(X_train, X_test):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def apply_power_transform(X_train, X_test):
    transformer = PowerTransformer(method='yeo-johnson')
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    return X_train_transformed, X_test_transformed


def clean_categoricals(df):
    # This function cleans categorical columns containing 0,1 and nans, transforming them to F, T and NC
    # and then retunrs ONLY the columns cleaned
    selected_columns = []
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        if set(unique_values) <= {0, 1, np.nan}:
            selected_columns.append(column)
    
    cleaned_cats = df[selected_columns].replace({0: "F", 1: "T", np.nan: "NC"})
    
    return cleaned_cats