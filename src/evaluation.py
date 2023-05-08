import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score 
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score


def get_model_performance(predictions, y_test, table=False): 
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


def plot_confusion_matrix(y_true, y_pred, normalize=False, cmap='Blues'):
    labels = ['Negative', 'Positive']
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()