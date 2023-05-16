import pandas as pd
from sklearn.utils import resample

def equalize_class_imbalance(df, target_column):
    df_1 = df[df[target_column] == 1]
    df_0 = df[df[target_column] == 0]
    
    df_0_resampled = resample(df_0, replace=False, n_samples=(15000), random_state=42)
    
    df_equalized = pd.concat([df_1, df_0_resampled])
    
    df_equalized = df_equalized.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(df_equalized[target_column].value_counts())
    
    y = df_equalized[target_column]
    x = df_equalized.drop(target_column, axis=1)
    
    return x, y
