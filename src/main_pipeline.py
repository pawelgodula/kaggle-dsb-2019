import gc
import json
import argparse
from models import TrainerLGBM
from data_processors import MainData, BureauData, PreviousApplicationData, InstallmentsPaymentsData, POSCashBalanceData, CreditCardBalanceData, BureauBalanceData
from utils import load_features_and_params

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.*")

def parse_args():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk Prediction")    
    parser.add_argument('--path_to_data', required=True, type=str, default='/kaggle/input/home-credit-default-risk/', help='Path to the data directory')
    parser.add_argument('--path_to_opt_settings', required=True, type=str, default='/kaggle/working/kagglehomecredit/src/opt_settings', help='Path to the optimization settings directory')
    parser.add_argument('--sample_rate', type=float, default=0.01, help='Sample rate for data processing')
    parser.add_argument('--num_parallel_processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--use_precomputed_optimal_settings', type=bool, default=True, help='Whether to use precomputed optimal settings')
    parser.add_argument('--n_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--metric', type=str, default='auc', help='Evaluation metric')
    parser.add_argument('--task_type', type=str, default='classification', help='classification or regression')
    args=parser.parse_args()
    return args

def main_pipeline(args):
    path_to_data = args.path_to_data
    path_to_opt_settings = args.path_to_opt_settings
    sample_rate = args.sample_rate
    num_parallel_processes = args.num_parallel_processes
    use_precomputed_optimal_settings = args.use_precomputed_optimal_settings
    n_fold = args.n_fold
    seed = args.seed
    metric = args.metric
    task_type = args.task_type

    ### Feature engineering
  
    main_data_processor = MainData(path_to_data, sampling = 0.1)
    df, target_col, y, categorical_feats = main_data_processor.process()
    del main_data_processor; gc.collect()

    bureau_processor = BureauData(path_to_data, num_parallel_processes, sample_rate)
    bureau_features = bureau_processor.process()
    bureau_id_map = bureau_processor.get_id_mapping()
    for feat_df in bureau_features:
        df = df.merge(feat_df, on="SK_ID_CURR", how='left')
    del bureau_processor, bureau_features; gc.collect()

    # prev_app_processor = PreviousApplicationData(path_to_data, num_parallel_processes, sample_rate)
    # prev_credit_app_features = prev_app_processor.process()
    # for feat_df in prev_credit_app_features:
    #     df = df.merge(feat_df, on="SK_ID_CURR", how='left')
    # del prev_app_processor, prev_credit_app_features; gc.collect()

    # inst_pmt_processor = InstallmentsPaymentsData(path_to_data, num_parallel_processes, sample_rate)
    # inst_pmt_features = inst_pmt_processor.process()
    # for feat_df in inst_pmt_features:
    #     df = df.merge(feat_df, on="SK_ID_CURR", how='left')
    # del inst_pmt_processor, inst_pmt_features; gc.collect()

    # pos_bal_processor = POSCashBalanceData(path_to_data, num_parallel_processes, sample_rate)
    # pos_bal_features = pos_bal_processor.process()
    # for feat_df in pos_bal_features:
    #     df = df.merge(feat_df, on="SK_ID_CURR", how='left')    
    # del pos_bal_processor, pos_bal_features; gc.collect()

    # cc_bal_processor = CreditCardBalanceData(path_to_data, num_parallel_processes, sample_rate)
    # cc_bal_features = cc_bal_processor.process()
    # for feat_df in cc_bal_features:
    #     df = df.merge(feat_df, on="SK_ID_CURR", how='left')
    # del cc_bal_processor, cc_bal_features; gc.collect()

    # buro_bal_processor = BureauBalanceData(path_to_data, bureau_id_map, num_parallel_processes, sample_rate)
    # buro_bal_features = buro_bal_processor.process()
    # for feat_df in buro_bal_features:
    #     df = df.merge(feat_df, on="SK_ID_CURR", how='left')
    # del buro_bal_processor, buro_bal_features; gc.collect()

    ### Feature selection & hyperparameter optimization
    
    if use_precomputed_optimal_settings:
        unimportant_features, optimal_lgb_params = load_features_and_params(path_to_opt_settings)
    else:
        full_df = df.iloc[:y.shape[0],:].copy()
        full_df['TARGET']=y
        trainer_lgb = TrainerLGBM(seed=seed)
        unimportant_features, optimal_lgb_params = trainer_lgb.optimize_features_params(full_df, task_type, None, full_df.columns.drop('TARGET'), 'TARGET', metric, 2, 1, 3600, categoricals = categorical_feats)

    df.drop(columns=unimportant_features, inplace=True)

    ### Train the main model

    optimal_lgb_params = {'n_estimators': 100}

    full_df = df.iloc[:y.shape[0],:].copy()
    full_df['TARGET']=y
    trainer_lgb = TrainerLGBM(seed=seed)
    models, _, val_metric = trainer_lgb.fit_kfold(full_df, task_type, optimal_lgb_params, full_df.columns.drop('TARGET'), 'TARGET', metric, n_fold, False, categorical_feats)
    print(f'CV {metric}: {val_metric}')

    ### Make a submission

    pred_df = df.iloc[y.shape[0]:,:][full_df.columns.drop('TARGET')]
    id_ = df.iloc[y.shape[0]:,:]["SK_ID_CURR"]
    submission_df = trainer_lgb.build_submission(models, pred_df, id_, task_type, categorical_feats)
    submission_df.columns = ['SK_ID_CURR', 'TARGET']
    submission_df.to_csv('submission.csv', index = False)

def main():
    args = parse_args()
    main_pipeline(args)

if __name__ == "__main__":
    main()
