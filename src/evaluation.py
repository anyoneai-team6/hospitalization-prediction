import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score 
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

def get_model_performance(y_test, predictions, table=False): 
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    performance_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Score': [accuracy, precision, recall, f1]
    })
    
    if not table:
        sns.barplot(x='Metric', y='Score', data=performance_table, color='skyblue')
        for index, row in performance_table.iterrows():
            plt.text(index, row['Score'], f"{row['Score']:.2f}", ha='center', va='bottom')
        
        plt.title('Model Performance')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.show()
    
    if  table:
        return performance_table


def plot_roc(y_true, y_score, style='darkgrid'):
    sns.set_style(style)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    sns.despine()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, normalize=True, cmap='Blues'):
    labels = ['Negative', 'Positive']
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

########################################################################################

from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize



def get_performance(predictions, y_test, labels = [1, 0],) -> Tuple[float, float, float, float]:
    """
    Get model performance using different metrics.

    Args:
        predictions : Union[List, np.ndarray]
            Predicted labels, as returned by a classifier.
        y_test : Union[List, np.ndarray]
            Ground truth (correct) labels.
        labels : Union[List, np.ndarray]
            Optional display names matching the labels (same order).
            Used in `classification_report()`.

    Return:
        accuracy : float
        precision : float
        recall : float
        f1_score : float
    """
    # TODO: Compute metrics
    # Use sklearn.metrics.accuracy_score
    accuracy = metrics.accuracy_score(y_test,predictions)
    # Use sklearn.metrics.precision_score
    precision = metrics.precision_score(y_test,predictions)
    # Use sklearn.metrics.recall_score
    recall = metrics.recall_score(y_test,predictions)
    # Use sklearn.metrics.f1_score
    f1_score = metrics.f1_score(y_test,predictions)
    # Use sklearn.metrics.classification_report
    report = metrics.classification_report(y_test,predictions)

    # TODO: Get Confusion Matrix, use sklearn.metrics.confusion_matrix
    cm = metrics.confusion_matrix(y_test,predictions)

    # Convert Confusion Matrix to pandas DataFrame, don't change this code!
    cm_as_dataframe = pd.DataFrame(data=cm)
    # Print metrics, don't change this code!
    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    print(cm_as_dataframe)

    # Return resulting metrics, don't change this code!
    return accuracy, precision, recall, f1_score

def plot_roc_ai(model, y_test, features) -> float:
    """
    Plot ROC Curve graph.

    Args:
        model : BaseEstimator
            Classifier model.
        y_test : Union[List, np.ndarray]
            Ground truth (correct) labels.
        features : List[int]
            Dataset features used to evaluate the model.

    Return:
        roc_auc : float
            ROC AUC Score.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_labels = model.classes_
    y_test = label_binarize(y_test, classes=class_labels)

    prob = model.predict_proba(features)
    y_score = prob[:, prob.shape[1] - 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc})", linewidth=2.5)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

