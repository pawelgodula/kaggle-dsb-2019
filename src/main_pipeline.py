import gc
import json
import argparse
from models import TrainerLGBM
from data_processors import (
    MainData,
    BureauData,
    PreviousApplicationData,
    InstallmentsPaymentsData,
    POSCashBalanceData,
    CreditCardBalanceData,
    BureauBalanceData,
)
from utils import load_features_and_params

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.*")


def parse_args():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk Prediction")
    parser.add_argument(
        "--path_to_data",
        required=True,
        type=str,
        default="/kaggle/input/home-credit-default-risk/",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--path_to_opt_settings",
        required=True,
        type=str,
        default="/kaggle/working/kagglehomecredit/src/opt_settings",
        help="Path to the optimization settings directory",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.01,
        help="Sample rate for data processing",
    )
    parser.add_argument(
        "--num_parallel_processes",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--use_precomputed_optimal_settings",
        type=bool,
        default=True,
        help="Whether to use precomputed optimal settings",
    )
    parser.add_argument(
        "--n_fold", type=int, default=5, help="Number of folds for cross-validation"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--metric", type=str, default="auc", help="Evaluation metric")
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        help="classification or regression",
    )
    args = parser.parse_args()
    return args


def feature_engineering(path_to_data, num_parallel_processes, sample_rate):
    main_data_processor = MainData(path_to_data, sampling=0.1)
    df, target_col, y, categorical_feats = main_data_processor.process()
    del main_data_processor
    gc.collect()

    processors = [
        BureauData(path_to_data, num_parallel_processes, sample_rate),
        PreviousApplicationData(path_to_data, num_parallel_processes, sample_rate),
        InstallmentsPaymentsData(path_to_data, num_parallel_processes, sample_rate),
        POSCashBalanceData(path_to_data, num_parallel_processes, sample_rate),
        CreditCardBalanceData(path_to_data, num_parallel_processes, sample_rate),
        BureauBalanceData(path_to_data, bureau_id_map, num_parallel_processes, sample_rate)
    ]

    for processor in processors:
        feature = processor.process()
        for feat_df in feature:
            df = df.merge(feat_df, on="SK_ID_CURR", how="left")
        del processor, feature
        gc.collect()

    return df, y, categorical_feats


def feature_selection_and_hyperparameter_optimization(df, y, categorical_feats, args):
    if args.use_precomputed_optimal_settings:
        unimportant_features, optimal_lgb_params = load_features_and_params(args.path_to_opt_settings)
    else:
        full_df = df.iloc[: y.shape[0], :].copy()
        full_df["TARGET"] = y
        trainer_lgb = TrainerLGBM(seed=args.seed)
        unimportant_features, optimal_lgb_params = trainer_lgb.optimize_features_params(
            full_df,
            args.task_type,
            None,
            full_df.columns.drop("TARGET"),
            "TARGET",
            args.metric,
            2,
            1,
            3600,
            categoricals=categorical_feats,
        )
    df.drop(columns=unimportant_features, inplace=True)
    return df, optimal_lgb_params


def train_model(df, y, categorical_feats, args, optimal_lgb_params):
    full_df = df.iloc[: y.shape[0], :].copy()
    full_df["TARGET"] = y
    trainer_lgb = TrainerLGBM(seed=args.seed)
    models, _, val_metric = trainer_lgb.fit_kfold(
        full_df,
        args.task_type,
        optimal_lgb_params,
        full_df.columns.drop("TARGET"),
        "TARGET",
        args.metric,
        args.n_fold,
        False,
        categorical_feats,
    )
    print(f"CV {args.metric}: {val_metric}")
    return models


def build_submission(df, y, models, args, categorical_feats):
    pred_df = df.iloc[y.shape[0]:, :][df.columns.drop("TARGET")]
    id_ = df.iloc[y.shape[0]:, :]["SK_ID_CURR"]
    submission_df = trainer_lgb.build_submission(models, pred_df, id_, args.task_type, categorical_feats)
    submission_df.columns = ["SK_ID_CURR", "TARGET"]
    return submission_df


def main_pipeline(args):
    df, y, categorical_feats = feature_engineering(args.path_to_data, args.num_parallel_processes, args.sample_rate)
    df, optimal_lgb_params = feature_selection_and_hyperparameter_optimization(df, y, categorical_feats, args)
    models = train_model(df, y, categorical_feats, args, optimal_lgb_params)
    submission_df = build_submission(df, y, models, args, categorical_feats)
    submission_df.to_csv("submission.csv", index=False)


def main():
    args = parse_args()
    main_pipeline(args)


if __name__ == "__main__":
    main()
