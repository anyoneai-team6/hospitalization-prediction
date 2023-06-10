
import pandas as pd


def select_elements(features: list, code: str) -> list:
    """Select the elements of a list that start with a specific code.

    Args:
        features (list) : List of elements to be filtered.
        code (str): Code to filter the elements of the list.

    Returns:
        list: List of elements that start with the code.
        """
    features_selected = [feature for feature in features
                         if str(feature).startswith(code)]
    return features_selected


def rename_columns(features: list, new_code: str) -> list:
    """Rename the columns of a dataframe

    Args:
        features (list): list with features names
        new_code (str): new code to assign to the features

    Returns:
        list: list with the new features names
    """

    features_renamed = [new_code+column_name[2:] for column_name in features]
    return features_renamed


def column_validation(column_list: list, dataframe: pd.DataFrame) -> tuple[list]:
    """
    Args:
        column_list (list): _description_
        dataframe (pd.DataFrame): _description_

    Returns:
        tuple[list]: _description_
    """

    existing_columns = []
    non_existing_columns = []

    for column in column_list:
        if column in dataframe.columns:
            existing_columns.append(column)
        else:
            non_existing_columns.append(column)

    return existing_columns, non_existing_columns



def process_data(df_r: pd.DataFrame, df: pd.DataFrame, encode: str) -> pd.DataFrame:
    """this function process the data for the respondent and the spouse

    Args:
        df_r (pd.DataFrame): _description_
        df (pd.DataFrame): _description_
        encode (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_r_new = pd.DataFrame()

    col = rename_columns(df_r.columns, encode)
    exist, no_exist = column_validation(col, df)

    df_r_new[rename_columns(exist, 'pa')] = df[exist]
    df_r_new[rename_columns(no_exist, 'pa')
             ] = df_r[rename_columns(no_exist, 'pa')]

    return df_r_new


def extract(df: pd.DataFrame) -> pd.DataFrame:

    all_columns = df.columns
    selected = [feature for feature in all_columns if str(
        feature).startswith("r1")]

    # delete: Household Analysis Weight
    if 'r1wthh' in selected:
        selected.remove('r1wthh')

    p1 = ["pa"+feature[2:] for feature in selected]
    s1 = ["s1"+feature[2:] for feature in selected]

    df_s1 = pd.DataFrame()
    df_r1 = pd.DataFrame()

    # select the data for spouse in the wave 1
    df_s1[p1] = df[s1].copy()
    # select the data for respondent in the wave 1
    df_r1[p1] = df[selected].copy()

    # create: Household Analysis Weight, for spouse and respondent
    df_s1['pawthh'] = df['r1wthh'].copy()
    df_r1['pawthh'] = df['r1wthh'].copy()
    # create: gender, for spouse and respondent
    df_s1['pagender'] = df['s1gender'].copy()
    df_r1['pagender'] = df['ragender'].copy()

    df_r2=process_data(df_r1, df, 'r2')
    df_r3=process_data(df_r2, df, 'r3')
    df_r4=process_data(df_r3, df, 'r4')
    df_r5=process_data(df_r4, df, 'r5')

    df_s2=process_data(df_s1, df, 's2')
    df_s3=process_data(df_s2, df, 's3')
    df_s4=process_data(df_s3, df, 's4')
    df_s5=process_data(df_s4, df, 's5')

    df_r1['cpindex']=63.29
    df_r2['cpindex']=73.90
    df_r3['cpindex']=107.69
    df_r4['cpindex']=119.40
    df_r5['cpindex']=136.6

    df_s1['cpindex']=63.29
    df_s2['cpindex']=73.90
    df_s3['cpindex']=107.69
    df_s4['cpindex']=119.40
    df_s5['cpindex']=136.6

    df_concatenated = pd.concat([df_r1, df_r2, df_r3, df_r4,
                                 df_r5, df_s1, df_s2, df_s3, 
                                 df_s4, df_s5])

    return df_concatenated