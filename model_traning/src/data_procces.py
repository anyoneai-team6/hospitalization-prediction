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


def feature_importance(model, X: pd.DataFrame, y: pd.Series, n: int):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    feature_names_original = list(X.columns)
    
    target_name = y.name  # Obtener el nombre de la columna objetivo desde la Serie y
    
    if target_name in feature_names_original:
        feature_names_original.remove(target_name)
    
    max_importance = list(indices[:n])
    
    nueva_lista = [feature_names_original[i] for i in max_importance]
    
    return nueva_lista