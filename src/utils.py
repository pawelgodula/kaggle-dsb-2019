import json
from typing import List, Dict, Tuple

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
