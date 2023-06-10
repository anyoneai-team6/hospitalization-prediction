import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency



def plot_missing_values(dataframe:pd.DataFrame, sorted=True ,display_xticklabels=False):
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_
        sorted (bool, optional): _description_. Defaults to True.
        display_xticklabels (bool, optional): _description_. Defaults to False.
    """

    # Calculate the percentage of missing values
    missing_values_series = dataframe.isna().mean() * 100
    df = pd.DataFrame(missing_values_series, columns=["Percentage of Missings"])
    if sorted:
        df = df.sort_values(by="Percentage of Missings", ascending=False)

    with plt.style.context('bmh'):
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=df.index, y="Percentage of Missings", data=df, linewidth=0)

        if display_xticklabels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.set_xticklabels([])

        ax.set_xlabel("Feature")
        ax.set_ylabel("Percentage of Missings")
        ax.set_title("Data Missing Values")

        # Adjust the layout to prevent label overlap
        plt.tight_layout()
        plt.show()


def plot_multi_missing_values(*dfs, display_xticklabels=False):
    with plt.style.context('bmh'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        for i, df in enumerate(dfs):
            missing_values_series = df.isna().sum() / df.shape[0] * 100
            missing_values_df = pd.DataFrame(missing_values_series, columns=["Percentage of Missings"])
            missing_values_df = missing_values_df.sort_values(by="Percentage of Missings", ascending=False)

            ax = axes[i // 3, i % 3]

            sns.barplot(x=missing_values_df.index, 
                        y="Percentage of Missings", 
                        data=missing_values_df, 
                        ax=ax,
                        color="#ff931e",
                        linewidth=0)
            
            if display_xticklabels:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("Feature")
            ax.set_ylabel("Percentage of Missings")
            ax.set_title(f"Data Missing Values - DataFrame {i+1}")

        # Hide remaining subplots if there are fewer DataFrames than subplots
        for j in range(len(dfs), 6):
            axes[j // 3, j % 3].axis("off")

        plt.show()



def plot_uniques(df, threshold=1, display_xticklabels=False):
    # Get the number of unique values per column
    unique_values_series = df.nunique()
    df = pd.DataFrame(unique_values_series, columns=["Number of Unique Values"])
    df = df.sort_values(by="Number of Unique Values", ascending=False)
    df = df[df["Number of Unique Values"] >= threshold]
    

    with plt.style.context('bmh'):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=df.index, y="Number of Unique Values", data=df)

        if display_xticklabels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.set_xticklabels([])

        ax.set_xlabel("Feature")
        ax.set_ylabel("Number of Unique Values")
        ax.set_title("Data Unique Values")
        # Adjust the layout to prevent label overlap
        plt.tight_layout()

        # Display the chart
        plt.show()

def plot_multi_uniques(*dfs, threshold=1 ,display_xticklabels=False):

    with plt.style.context('bmh'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        for i, df in enumerate(dfs):

            unique_values_series = df.nunique()
            uniques_df = pd.DataFrame(unique_values_series, columns=["Number of Unique Values"])
            uniques_df = uniques_df.sort_values(by="Number of Unique Values", ascending=False)
            uniques_df = uniques_df[uniques_df["Number of Unique Values"] >= threshold]
            ax = axes[i // 3, i % 3]

            sns.barplot(x=uniques_df.index, 
                        y="Number of Unique Values", 
                        data=uniques_df, 
                        ax=ax,
                        color="#ff6ea9",
                        linewidth=0)
            
            if display_xticklabels:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("Feature")
            ax.set_ylabel("n uniques values")
            ax.set_title(f"Unique Values - DataFrame {i+1}")

        # Hide remaining subplots if there are fewer DataFrames than subplots
        for j in range(len(dfs), 6):
            axes[j // 3, j % 3].axis("off")

        plt.show()


def plot_multi_ndistribution(*data_series):
    with plt.style.context('bmh'):
        num_plots = len(data_series)
        num_rows = (num_plots - 1) // 3 + 1  # Calculate the number of rows based on the number of plots
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(18, num_rows * 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        for i, series in enumerate(data_series):
            ax = axes[i // 3, i % 3]  # Select the appropriate subplot

            sns.histplot(series, ax=ax, linewidth=0)

            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution - Data Series {i+1}")

        # Hide remaining subplots if there are fewer data series than subplots
        for j in range(num_plots, num_rows * 3):
            axes[j // 3, j % 3].axis("off")

        plt.show()

def plot_multi_cdistribution(*data_series:pd.Series, display_percentage=True):
    with plt.style.context('bmh'):
        num_plots = len(data_series)
        num_rows = (num_plots - 1) // 3 + 1  # Calculate the number of rows based on the number of plots
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(18, num_rows * 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        for i, series in enumerate(data_series):
            value_counts = series.value_counts()
            value_counts.sort_index(ascending=True ,inplace=True)

            ax = axes[i // 3, i % 3]  # Select the appropriate subplot

            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, order=value_counts.index)

            ax.set_xlabel("Unique Values")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution - Data Series {i+1}")

            if display_percentage:
                total = value_counts.sum()
                for j, count in enumerate(value_counts.values):
                    percentage = (count / total) * 100
                    ax.text(j, count, f"{percentage:.2f}%", ha='center', va='bottom')

        # Hide remaining subplots if there are fewer data series than subplots
        for j in range(num_plots, num_rows * 3):
            axes[j // 3, j % 3].axis("off")

        plt.show()


def binary_corr(var1:pd.Series, var2:pd.Series)->tuple[float, pd.DataFrame]:
    """Calculate the binary correlation between two categorical variables.
    Args:
        var1 (pd.Series): First categorical variable.
        var2 (pd.Series): Second categorical variable.
    Returns:
        tuple[float, pd.DataFrame]: The first element is the Cramer's V value 
        and the second element is the contingency table.
    """
    contingency_table=pd.crosstab(var1, var2)
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()  # Total sample size
    phi_c = np.sqrt(chi2 / n)
    V = np.sqrt(phi_c / min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1))

    return round(V,2), contingency_table


def plot_categorical_correlation(df: pd.DataFrame, target_column: str):
    """
    Plot the correlation between each categorical variable and the target variable.
    Args:
        df (pd.DataFrame): The DataFrame containing the categorical variables.
        target_column (str): The name of the target variable.
        
        Returns:
        a bar plot showing the correlation."""

    corr_scores = []
    columns = []

    for column in df.columns:
        if column != target_column:
            corr, _ = binary_corr(df[column], df[target_column])
            corr_scores.append(corr)
            columns.append(column)

    corr_df = pd.DataFrame({"Column": columns, "Correlation": corr_scores})
    corr_df = corr_df.sort_values("Correlation", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Correlation", y="Column", data=corr_df, palette="viridis")
    ax.set_xlabel("Correlation (Cramer's V)")
    # do not display the y ticks
    ax.set_yticks([])
    ax.set_ylabel("Column")
    ax.set_title("Categorical Correlation with Target Variable")
    plt.show()