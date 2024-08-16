import numpy as np
import pandas as pd

def binarize_feature(df: pd.DataFrame, feature: str, threshold: float) -> pd.DataFrame:
    """
    Binarize a continuous feature based on a threshold value.

    Args:
        df (pd.DataFrame): The input dataframe containing the feature.
        feature (str): The name of the feature to binarize.
        threshold (float): The threshold value to determine binarization.

    Returns:
        pd.DataFrame: The dataframe with the new binarized feature added.
    
    Raises:
        ValueError: If the feature does not exist in the dataframe.
        TypeError: If the dataframe is not a pandas DataFrame or the threshold is not a float.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' does not exist in the dataframe.")

    df[f'{feature}_binarized'] = (df[feature] >= threshold).astype(int)
    return df


def log_transform_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Apply a log transformation to a feature, handling negative values by replacing them with NaN.

    Args:
        df (pd.DataFrame): The input dataframe containing the feature.
        feature (str): The name of the feature to log-transform.

    Returns:
        pd.DataFrame: The dataframe with the log-transformed feature added.
    
    Raises:
        ValueError: If the feature does not exist in the dataframe.
        TypeError: If the dataframe is not a pandas DataFrame.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' does not exist in the dataframe.")

    df[feature] = df[feature].apply(lambda x: np.nan if x < 0 else x)
    df[f'{feature}_log'] = np.log1p(df[feature])
    return df

def add_polynomial_features(df: pd.DataFrame, feature: str, degree: int = 2) -> pd.DataFrame:
    """
    Add polynomial features to a given feature up to a specified degree.

    Args:
        df (pd.DataFrame): The input dataframe containing the feature.
        feature (str): The name of the feature to add polynomial features to.
        degree (int): The maximum degree of the polynomial features. Default is 2.

    Returns:
        pd.DataFrame: The dataframe with the polynomial features added.
    
    Raises:
        ValueError: If the feature does not exist in the dataframe.
        TypeError: If the dataframe is not a pandas DataFrame or degree is not an integer.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' does not exist in the dataframe.")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("Degree must be a positive integer.")

    for d in range(2, degree + 1):
        df[f'{feature}_poly{d}'] = df[feature] ** d
    return df

def impute_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Impute missing values in the dataframe using a specified strategy.

    Args:
        df (pd.DataFrame): The input dataframe containing missing values.
        strategy (str): The imputation strategy. Can be 'mean', 'median', or 'constant'. Default is 'mean'.

    Returns:
        pd.DataFrame: The dataframe with missing values imputed.
    
    Raises:
        ValueError: If the strategy is not one of 'mean', 'median', or 'constant'.
        TypeError: If the dataframe is not a pandas DataFrame.
    """
    if strategy not in ['mean', 'median', 'constant']:
        raise ValueError("Strategy must be one of 'mean', 'median', or 'constant'.")

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if strategy == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'constant':
                df[column] = df[column].fillna(0)
    return df