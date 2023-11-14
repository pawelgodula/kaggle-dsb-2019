import json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def load_features_and_params(path_to_opt_settings: str) -> Tuple[List[str], Dict]:
    """
    Loads the list of unimportant features and optimal LightGBM parameters from specified files.
    Parameters:
    - path_to_opt_settings (str): The path to the directory containing the pre-computed 'unimportant_features.txt' and optimal_lgbm_params.json' files.
    Returns:
    tuple: A tuple containing two elements:
        - unimportant_features (list of str): A list of feature names considered unimportant.
        - optimal_lgb_params (dict): A dictionary with the optimal LightGBM parameters.
    """    
    with open(f'{path_to_opt_settings}/unimportant_features.txt', 'r') as file:
        unimportant_features = [line.strip() for line in file.readlines()]
    with open(f'{path_to_opt_settings}/optimal_lgbm_params.json', 'r') as file:
         optimal_lgb_params = json.load(file)
    return unimportant_features, optimal_lgb_params

def dispersion(x: np.ndarray) -> float:
    """
    Calculate the dispersion of an array, defined as the difference between its maximum and minimum values,
    while ignoring NaN values.

    Parameters:
    x (np.ndarray): A NumPy array.

    Returns:
    float: The dispersion of the array. Returns NaN if the array is empty or contains only NaN values.
    """
    if np.all(np.isnan(x)) or len(x) == 0:
        return np.nan
    return np.nanmax(x) - np.nanmin(x)

def share_na(x: pd.Series) -> float:
    """
    Calculate the proportion of NaN values in a pandas Series.

    Parameters:
    x (pd.Series): A pandas Series.

    Returns:
    float: The proportion of NaN values in the Series.
    """
    return sum(x.isnull()) / len(x)
    
def light_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Divide two arrays element-wise and return the result as a float32 array.

    Parameters:
    numerator (np.ndarray): The numerator array.
    denominator (np.ndarray): The denominator array.

    Returns:
    np.ndarray: The element-wise division of the two arrays, cast to float32.
    """
    return np.divide(numerator, denominator).astype(np.float32)

def reduce_column_names(multi_level_df: pd.DataFrame, prefix: str) -> list:
    """
    Reduces the column names in a multi-level pandas DataFrame by concatenating all levels with a prefix.
    
    Parameters:
    multi_level_df (pd.DataFrame): A multi-level pandas DataFrame.
    prefix (str): A prefix to prepend to each column name.

    Returns:
    list: A list of new column names where each column name is a concatenation of the prefix and the original multi-level names.

    Raises:
    ValueError: If the input is not a multi-level pandas DataFrame.
    """
    if not isinstance(multi_level_df, pd.DataFrame):
        raise ValueError("The first argument must be a pandas DataFrame.")

    if not isinstance(multi_level_df.columns, pd.MultiIndex):
        raise ValueError("The DataFrame does not have multi-level columns.")

    new_columns = [
        f"{prefix}_{'_'.join(map(str, levels))}".replace(' ', '_') for levels in multi_level_df.columns
    ]

    return new_columns
