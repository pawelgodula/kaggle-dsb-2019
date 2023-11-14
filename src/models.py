from typing import List, Tuple, Set, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc

class TrainerLGBM:
    """
    A LightGBM model trainer class for running K-fold cross-validation,
    feature importance plotting, and identifying unimportant features.
    
    Attributes:
        seed (int): Random seed for reproducibility.
        default_params (dict): Default hyperparameters for the models.
    """
    
    def __init__(self, seed: int) -> None:
        """
        Initializes the TrainerLGBM class with a seed and default parameters.
        Args:
            seed (int): The seed for random number generation to ensure reproducibility.            
        """
        self.seed = seed
        self.default_params = {'n_estimators': 100,
                               'learning_rate': 0.1,
                               'num_leaves': 32,
                               'max_depth': -1,
                               'feature_fraction': 1.0}
    
    def set_seed(self, seed: int) -> 'TrainerLGBM':
        """
        Set the random seed for the trainer class.
        Args:
            seed (int): The seed to set for random number generation.
        Returns:
            TrainerLGBM: The instance of the TrainerLGBM with updated seed.
        """
        self.seed = seed
        return self
    
    def set_tr_val_indexes(self, full_df: pd.DataFrame, n_fold: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indexes for K-fold cross-validation.
        Args:
            full_df (pd.DataFrame): The dataframe containing the data to be split.
            n_fold (int): The number of folds to use for cross-validation.
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples of length = n_fold, with each tuple containing train and validation indexes.
        """
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=self.seed)
        return list(kf.split(full_df))
    
    def init_model(self, task_type: str, params: Optional[Dict[str, Any]] = None) -> lgb.LGBMModel:
        """
        Initialize a LightGBM model with the given or default parameters.
        Args:
            task_type (str): The type of task ('classification' or 'regression').
            params (Optional[Dict[str, Any]]): Additional parameters for the model. If None, default parameters are used.
        Returns:
            lgb.LGBMModel: The initialized LightGBM model.
        """
        # If no parameters are provided, use the defaults
        effective_params = self.default_params.copy()
        if params:
            effective_params.update(params) 
            
        if task_type == 'regression':
            lgb_model = lgb.LGBMRegressor(**effective_params)
        elif task_type == 'classification':
            lgb_model = lgb.LGBMClassifier(**effective_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return lgb_model
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, task_type: str, params: dict, features: List[str], target: str, eval_metric: str) -> lgb.LGBMModel:
        """
        Fit a LightGBM model on the training set and evaluate it on the validation set.
        Args:
            train_df (pd.DataFrame): The training data.
            val_df (pd.DataFrame): The validation data.
            task_type (str): The type of task ('classification' or 'regression').
            params (Dict[str, Any]): Parameters for the LightGBM model.
            features (List[str]): List of feature names used for training.
            target (str): The target variable name.
            eval_metric (str): The evaluation metric to use.
        Returns:
            lgb.LGBMModel: The trained LightGBM model.
        """
        model = self.init_model(task_type, params)
        model.fit(
            X=train_df[features], 
            y=train_df[target].values,
            eval_set=[(train_df[features].values, train_df[target].values), (val_df[features].values, val_df[target].values)],
            verbose=20, 
            early_stopping_rounds=20, 
            eval_metric=eval_metric
        )
        return model
    
    def predict(self, model: lgb.LGBMModel, data: pd.DataFrame, task_type: str) -> np.ndarray:
        """
        Make predictions using the trained LightGBM model.
        Args:
            model (lgb.LGBMModel): The trained LightGBM model.
            data (pd.DataFrame): The data on which to make predictions.
            task_type (str): The type of task ('classification' or 'regression').
        Returns:
            np.ndarray: The predicted values.
        """
        if task_type == 'regression':
            return model.predict(data)
        elif task_type == 'classification':
            return model.predict_proba(data)[:, 1]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def plot_importances(self, model: lgb.LGBMModel, n_top_feats: int = 25, figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot the feature importances of the trained model.
        Args:
            model (lgb.LGBMModel): The trained LightGBM model.
            n_top_feats (int): Number of top features to display in the plot. Defaults to 25.
            figsize (Tuple[int, int]): The size of the plot. Defaults to (10, 5).
        """
        feature_imp = pd.DataFrame(sorted(zip(model.booster_.feature_importance(importance_type='gain'), model.booster_.feature_name())), columns=['Value','Feature'])
        plt.figure(figsize=figsize)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:n_top_feats, :])
        plt.title('LightGBM Feature Importances')
        plt.tight_layout()
        plt.show()
    
    def validate_categoricals(self, full_df: pd.DataFrame, categoricals: List[str]) -> pd.DataFrame:
        """
        Ensure that categorical features have the correct dtype in the dataframe.
        Args:
            full_df (pd.DataFrame): The dataframe containing the features.
            categoricals (List[str]): List of categorical feature names.
        Returns:
            pd.DataFrame: The updated dataframe with corrected categorical dtypes.
        """
        for cat_feature in categoricals:
            if not pd.api.types.is_categorical_dtype(full_df[cat_feature]):
                full_df[cat_feature] = full_df[cat_feature].astype('category')
        return full_df
    
    def fit_kfold(self, full_df: pd.DataFrame, task_type: str, params: dict, features: List[str], target: str, eval_metric: str, n_fold: int, print_results: bool = True,
                  categoricals: List[str] = None) -> Tuple[List[lgb.LGBMModel], np.ndarray, np.float64]:
        """
        Perform K-fold cross-validation.
        Args:
            full_df (pd.DataFrame): The full dataset.
            task_type (str): The type of task ('classification' or 'regression').
            params (Dict[str, Any]): Parameters for the LightGBM model.
            features (List[str]): List of feature names used for training.
            target (str): The target variable name.
            eval_metric (str): The evaluation metric to use.
            n_fold (int): The number of folds for cross-validation.
            print_results (bool): Whether to print results and plot feature importances. Defaults to True.
            categoricals (List[str], optional): List of categorical feature names.
        Returns:
            Tuple[List[lgb.LGBMModel], np.ndarray, float]: A tuple containing the list of trained models, out-of-fold predictions, and validation score.
        """
        if categoricals:
            full_df = self.validate_categoricals(full_df, categoricals)
        tr_val_idx = self.set_tr_val_indexes(full_df, n_fold)                
        val_preds = np.zeros(full_df.shape[0])
        models = []
        val_score = 0
        for i, (tr_idx, val_idx) in enumerate(tr_val_idx):
            print('FOLD', i)
            train_df, val_df = full_df.iloc[tr_idx, :], full_df.iloc[val_idx, :]
            model = self.fit(train_df, val_df, task_type, params, features, target, eval_metric)
            if print_results:
                self.plot_importances(model)
            models.append(model)
            val_preds[val_idx] = self.predict(model, val_df[features], task_type)
            cur_score = model.best_score_['valid_1'][eval_metric]
            val_score += cur_score/n_fold
            gc.collect()
        if print_results:    
            print(f'Overall {eval_metric}', val_score)  
        return models, val_preds, val_score
    
    def find_unimportant_features(self, full_df: pd.DataFrame, task_type: str, params: dict, features: List[str], target: str, eval_metric: str, n_fold: int, categoricals: List[str] = None) -> List[str]:
        """
        Find features that have zero importance in at least one fold of the cross-validation.
        Args:
            full_df (pd.DataFrame): The full dataset.
            task_type (str): The type of task ('classification' or 'regression').
            params (Dict[str, Any]): Parameters for the LightGBM model.
            features (List[str]): List of feature names used for training.
            target (str): The target variable name.
            eval_metric (str): The evaluation metric to use.
            n_fold (int): The number of folds for cross-validation.
            categoricals (List[str], optional): List of categorical feature names.
        Returns:
            Set[str]: A set of features that are found to be unimportant across at least one fold.
        """

        models, _, _ = self.fit_kfold(full_df, task_type, params, features, target, eval_metric, n_fold, False, categoricals)
        unimportant_features = set()
        for model in models:
            feat_imp_df = pd.DataFrame({
                'Feature': model.booster_.feature_name(),
                'Value': model.booster_.feature_importance(importance_type='gain')
            })
            zero_importance_features = feat_imp_df[feat_imp_df['Value'] == 0]['Feature'].tolist()
            unimportant_features.update(zero_importance_features)
        return list(unimportant_features)
    
    def optimize_hyperparameters(self, full_df: pd.DataFrame, task_type: str, features: List[str], target: str, eval_metric: str, n_fold: int, n_trials: int, timeout: Optional[int], categoricals: List[str] = None) -> dict:
        """
        Optimize hyperparameters using Optuna.
        Args:
            full_df (pd.DataFrame): Full dataset containing features and target.
            task_type (str): Type of task ('classification' or 'regression').
            features (List[str]): List of feature names.
            target (str): Target column name.
            eval_metric (str): Evaluation metric for optimization.
            n_fold (int): Number of folds for K-fold cross-validation.
            n_trials (int): Number of trials for optimization.
            timeout (int): Time in seconds to timeout the optimization.
            categoricals (List[str], optional): List of categorical feature names.

        Returns:
            dict: The best hyperparameters found.
        """
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log = True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log = True),
            }
            _,_, val_score = self.fit_kfold(full_df, task_type, params, features, target, eval_metric, n_fold, False, categoricals)
            return val_score            
        
        study = optuna.create_study(direction='maximize' if eval_metric in ['accuracy', 'f1', 'auc'] else 'minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return study.best_params

    def optimize_features_params(self, full_df: pd.DataFrame, task_type: str, params: dict, features: List[str], target: str, eval_metric: str, n_fold: int, n_trials: int, timeout: Optional[int], categoricals: List[str] = None) -> Tuple[List, dict]:
        """
        Finds unimportant features and searches for the best hyperparameters sequentially
        Args:
            full_df (pd.DataFrame): Full dataset containing features and target.
            task_type (str): Type of task ('classification' or 'regression').
            features (List[str]): List of feature names.
            target (str): Target column name.
            eval_metric (str): Evaluation metric for optimization.
            n_fold (int): Number of folds for K-fold cross-validation.
            n_trials (int): Number of trials for optimization.
            timeout (int): Time in seconds to timeout the optimization.
            categoricals (List[str], optional): List of categorical feature names.

        Returns:
            Tuple[List, dict]: A tuple containing unimportant features and the best hyperparameters found
        """
        unimportant_features = self.find_unimportant_features(full_df, task_type, params, features, target, eval_metric, n_fold, categoricals)
        full_df.drop(columns=unimportant_features, inplace = True)
        best_params = self.optimize_hyperparameters(full_df, task_type, features, target, eval_metric, n_fold, n_trials, timeout, categoricals)
            
        return unimportant_features, best_params
