import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def correlation_filter(df, y_column, threshold=0.5):
    correlation_matrix = df.corr()
    
    correlation_with_target = correlation_matrix[y_column].abs()
    
    selected_features = correlation_with_target[correlation_with_target > threshold]
    
    return selected_features


def backward_elimination(X, y, significance_level=0.05):
    num_features = X.shape[1]
    for i in range(num_features):
        regressor = sm.OLS(y, X).fit()
        max_pvalue = max(regressor.pvalues)
        if max_pvalue > significance_level:
            max_pvalue_index = np.argmax(regressor.pvalues)
            X = np.delete(X, max_pvalue_index, 1)
        else:
            break
    return X



def forward_selection(X, y, significance_level=0.05):
    num_features = X.shape[1]
    selected_features = []
    best_error = float('inf')
    
    for i in range(num_features):
        remaining_features = list(set(range(X.shape[1])) - set(selected_features))
        best_feature = None
        best_model = None
        
        for feature in remaining_features:
            model = LinearRegression()
            model.fit(X[:, selected_features + [feature]], y)
            y_pred = model.predict(X[:, selected_features + [feature]])
            error = mean_squared_error(y, y_pred)
            
            if error < best_error:
                best_error = error
                best_feature = feature
                best_model = model
                
        if best_feature is not None:
            selected_features.append(best_feature)
        else:
            break
        
        if best_model is not None:
            model = best_model
        else:
            model = LinearRegression()
            model.fit(X[:, selected_features], y)
            
    return X[:, selected_features]