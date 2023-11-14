import json

def load_features_and_params(path_to_opt_settings):
    with open(f'{path_to_opt_settings}/unimportant_features.txt', 'r') as file:
        unimportant_features = [line.strip() for line in file.readlines()]
    with open(f'{path_to_opt_settings}/optimal_lgbm_params.json', 'r') as file:
         optimal_lgb_params = json.load(file)
    return unimportant_features, optimal_lgb_params
