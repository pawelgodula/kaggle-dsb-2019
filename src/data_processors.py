from multiprocessing import Pool, cpu_count
import math
import concurrent.futures
import pandas as pd
import numpy as np
import psutil
import gc
import time
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


#system settings
path_to_data = '/kaggle/input/home-credit-default-risk/'
num_parallel_processes = cpu_count()
sampling = {'main': 0.05,
            'bureau': 0.01,
            'pr_app': 0.01,
            'inst_pmt': 0.01,
            'pos_bal': 0.01,
           'cc_bal': 0.01,
           'buro_bal': 0.01}

#business settings
common_sense_interest_threshold = 0.085
days_in_month = 365/12

def dispersion(x):
    """
    returns the difference between max and min value ignoring nans
    """
    if np.all(np.isnan(x)) or len(x) == 0:
        return np.nan
    return np.nanmax(x) - np.nanmin(x)

def share_na(x):
    return(sum(x.isnull()) / len(x))

def light_divide(numerator, denominator):
    return np.divide(numerator, denominator).astype(np.float32)

def reduce_column_names(multi_level_df, prefix):
    """
    Reduces the column names in a multi-level pandas DataFrame by concatenating all levels with a prefix.
    
    """
    if not isinstance(multi_level_df, pd.DataFrame):
        raise ValueError("The first argument must be a pandas DataFrame.")

    if not isinstance(multi_level_df.columns, pd.MultiIndex):
        raise ValueError("The DataFrame does not have multi-level columns.")

    new_columns = [
        f"{prefix}_{'_'.join(map(str, levels))}" for levels in multi_level_df.columns
    ]

    return new_columns

aggregation_recipes = {
  
    'bureau' :
            {
               'AMT_CREDIT_SUM_DEBT': ['sum','mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_SUM': ['sum','mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_SUM_LIMIT': ['sum','mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_MAX_OVERDUE': ['sum', 'mean', "max", 'min', dispersion, share_na],           
               'CREDIT_DAY_OVERDUE': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'AMT_ANNUITY': ['sum', 'mean', "max", 'min', dispersion, share_na],  
               'CREDIT_CURRENCY': ['nunique', share_na],
               'SK_ID_BUREAU': ['count','sum','mean', "max", 'min', dispersion, share_na],
               'DAYS_CREDIT': ['sum','mean', "max", 'min', dispersion, share_na],
               'DAYS_CREDIT_UPDATE': ['sum','mean', "max", 'min', dispersion, share_na],
               'DAYS_CREDIT_ENDDATE': ['sum','mean', "max", 'min', dispersion, share_na],
               'DAYS_ENDDATE_FACT': ['sum','mean', "max", 'min', dispersion, share_na],
               'CNT_CREDIT_PROLONG': ['sum','mean', "max", 'min', dispersion, share_na],
               'credit_duration': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'credit_advance': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_SUM_DEBT_to_credit': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'AMT_CREDIT_MAX_OVERDUE_to_credit': ['sum', 'mean', "max", 'min', dispersion, share_na],
               'missing_info': ['sum', 'mean', "max", 'min', dispersion, share_na],
               "AMT_CREDIT_SUM_OVERDUE_to_credit" : ['sum', 'mean', "max", 'min', dispersion, share_na],
               "AMT_CREDIT_SUM_LIMIT_to_credit" : ['sum', 'mean', "max", 'min', dispersion, share_na],
               "repaid_to_credit" : ['sum', 'mean', "max", 'min', dispersion, share_na],
               "interest" : ['sum', 'mean', "max", 'min', dispersion, share_na],
               "princ_to_repay_per_month": ['sum'],
               'amt_repaid': ["sum"]
            },
    'previous_app' : 
            {
            'AMT_ANNUITY': ['sum','mean', "max", 'min', dispersion, share_na],
                       'AMT_APPLICATION': ['sum','mean', "max", 'min', dispersion, share_na],
           'AMT_CREDIT': ['sum','mean', "max", 'min', dispersion, share_na],
           'AMT_DOWN_PAYMENT': ['sum','mean', "max", 'min', dispersion, share_na],           
           'AMT_GOODS_PRICE': ['sum','mean', "max", 'min', dispersion, share_na],
           'HOUR_APPR_PROCESS_START': ['sum', 'mean', "max", 'min', dispersion, share_na],
           'DAYS_DECISION': ['sum', 'mean', "max", 'min', dispersion, share_na],
           'CNT_PAYMENT': ['sum', 'mean', "max", 'min', dispersion, share_na],
           'NFLAG_INSURED_ON_APPROVAL': ['sum', 'mean', "max", 'min', dispersion, share_na],
           
           "DAYS_FIRST_DRAWING" : ['sum', 'mean', "max", 'min', dispersion, share_na],
           "DAYS_FIRST_DUE": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "DAYS_LAST_DUE_1ST_VERSION": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "DAYS_LAST_DUE": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "DAYS_TERMINATION": ['sum', 'mean', "max", 'min', dispersion, share_na],
           
           'RATE_DOWN_PAYMENT': ['sum', 'mean', "max", 'min', dispersion, share_na],
           'RATE_INTEREST_PRIMARY': ['sum', 'mean', "max", 'min', dispersion, share_na],
           'RATE_INTEREST_PRIVILEGED': ['sum', 'mean', "max", 'min', dispersion, share_na],
                    
            'active': ['sum','mean'],
            'credit_to_app': ['sum','mean', "max", 'min', dispersion, share_na],
            'credit_to_good': ['sum','mean', "max", 'min', dispersion, share_na],
            'annuity_to_good': ['sum','mean', "max", 'min', dispersion, share_na],
            'annuity_to_credit': ['sum','mean', "max", 'min', dispersion, share_na],
            'down_to_good': ['sum','mean', "max", 'min', dispersion, share_na],
           'missing_info': ['sum', 'mean', "max", 'min', dispersion, share_na],
           "days_diff_last_first": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "credit_duration": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "credit_duration2": ['sum', 'mean', "max", 'min', dispersion, share_na],
           "days_diff_last_last" : ['sum', 'mean', "max", 'min', dispersion, share_na]           },
    
    'installments_payments' : {'delay': ['count','sum','mean', "max", 'min', dispersion],
           'lacking_money': ['sum','mean', "max", 'min', dispersion],
                      'surplus_money': ['sum','mean', "max", 'min', dispersion],
          "delay_money": ['sum','mean', "max", 'min', dispersion],
          "advance_money": ['sum','mean', "max", 'min', dispersion],           
          'lacking_money_ratio': ['sum','mean', "max", 'min', dispersion],
           "surplus_money_ratio": ['sum','mean', "max", 'min', dispersion]},
    
    'pos_bal': {'MONTHS_BALANCE': ['count','sum','mean', "max", 'min', dispersion],
           'CNT_INSTALMENT': ['sum','mean', "max", 'min', dispersion],
          "CNT_INSTALMENT_FUTURE": ['sum','mean', "max", 'min', dispersion],
          'SK_DPD': ['sum','mean', "max", 'min', dispersion], 
          'SK_DPD_DEF': ['sum','mean', "max", 'min', dispersion],
           "no_inst" : ['sum','mean', "max", 'min', dispersion]
          },
    
    'cc_bal' : {'count_missing': ['count','sum','mean', "max", 'min', dispersion, share_na],
           'bal_to_limit': ['sum','mean', "max", 'min', dispersion, share_na],
          "draw_atm_to_limit": ['sum','mean', "max", 'min', dispersion, share_na],
          'draw_pos_to_limit': ['sum','mean', "max", 'min', dispersion, share_na], 
          'draw_other_to_limit': ['sum','mean', "max", 'min', dispersion, share_na],
           "draw_atm_to_min_inst" : ['sum','mean', "max", 'min', dispersion, share_na],
           "draw_pos_to_min_inst" : ['sum','mean', "max", 'min', dispersion, share_na],
           "draw_other_to_min_inst" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_CREDIT_LIMIT_ACTUAL" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_DRAWINGS_ATM_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_DRAWINGS_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_DRAWINGS_OTHER_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_DRAWINGS_POS_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_INST_MIN_REGULARITY" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_PAYMENT_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_PAYMENT_TOTAL_CURRENT" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_RECEIVABLE_PRINCIPAL" : ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_RECIVABLE": ['sum','mean', "max", 'min', dispersion, share_na],
           "AMT_TOTAL_RECEIVABLE": ['sum','mean', "max", 'min', dispersion, share_na],
           "CNT_DRAWINGS_ATM_CURRENT": ['sum','mean', "max", 'min', dispersion, share_na],
           "CNT_DRAWINGS_CURRENT": ['sum','mean', "max", 'min', dispersion, share_na],
           "CNT_DRAWINGS_OTHER_CURRENT": ['sum','mean', "max", 'min', dispersion, share_na],
           "CNT_DRAWINGS_POS_CURRENT": ['sum','mean', "max", 'min', dispersion, share_na],
           "CNT_INSTALMENT_MATURE_CUM": ['sum','mean', "max", 'min', dispersion, share_na],
           "SK_DPD": ['sum','mean', "max", 'min', dispersion, share_na],
           "SK_DPD_DEF": ['sum','mean', "max", 'min', dispersion, share_na]
          },
    'buro_bal': {'0_col': ['count','sum','mean', "max", 'min'],
           '1_col': ['sum','mean', "max", 'min'],
          "2_col": ['sum','mean', "max", 'min'],
          '3_col': ['sum','mean', "max", 'min'], 
          '4_col': ['sum','mean', "max", 'min'],
           "5_col" : ['sum','mean', "max", 'min'],
           "C_col" : ['sum','mean', "max", 'min'],
           "X_col" : ['sum','mean', "max", 'min']           
          }
}

class MainData:
   def __init__(self, path_to_data, sampling = 1):
        self.path_to_data = path_to_data
        self.train_df = None
        self.test_df = None
        self.full_df = None
        self.target_col = ["TARGET"]
        self.y = None
        self.categorical_variables = []
        self.numerical_variables = []
        self.sampling = sampling
                
    def load_main_data(self):
        self.train_df = pd.read_csv(self.path_to_data + 'application_train.csv')
        if self.sampling < 1:
            self.train_df = self.train_df.sample(frac=self.sampling)
        
        self.test_df = pd.read_csv(self.path_to_data + 'application_test.csv')
        self.y = np.array(self.train_df.loc[:, self.target_col]).reshape((self.train_df.shape[0],))
        self.train_df.drop(self.target_col, axis=1, inplace=True)
        self.full_df = pd.concat([self.train_df, self.test_df], axis=0)
        return self
    
    def set_variable_types(self):
        self.categorical_variables = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                                      'NAME_HOUSING_TYPE',  'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','OCCUPATION_TYPE', 
                                      'WEEKDAY_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                                      'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'FLAG_DOCUMENT_2', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                                      'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 
                                      'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                                      'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        self.numerical_variables = ['SK_ID_CURR','AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','DAYS_BIRTH', 'CNT_FAM_MEMBERS','CNT_CHILDREN', 'REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED', 
                                    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 
                                    'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',"ENTRANCES_AVG", 
                                    'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 
                                    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 
                                    'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
                                    'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                                    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'HOUR_APPR_PROCESS_START',
                                    'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR',
                                    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
        return self
    
    def fe_main(self):
        lbl = preprocessing.LabelEncoder()
        for col in self.categorical_variables:
            self.full_df.loc[:, col] = lbl.fit_transform(self.full_df[col].astype(str))

        self.full_df['loan_to_income'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['AMT_INCOME_TOTAL'])
        self.full_df['loan_to_disp_income'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['AMT_INCOME_TOTAL'] / self.full_df['CNT_FAM_MEMBERS'])
        self.full_df['loan_to_car'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['OWN_CAR_AGE'])
        self.full_df['loan_to_good'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['AMT_GOODS_PRICE'])
        self.full_df['loan_to_age'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['DAYS_BIRTH'])
        self.full_df['loan_to_score'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['EXT_SOURCE_1'])
        self.full_df['ann_to_income'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['AMT_INCOME_TOTAL'])
        self.full_df['ann_to_disp_income'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['AMT_INCOME_TOTAL'] / self.full_df['CNT_FAM_MEMBERS'])
        self.full_df['ann_to_car'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['OWN_CAR_AGE'])
        self.full_df['ann_to_good'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['AMT_GOODS_PRICE'])
        self.full_df['ann_to_age'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['DAYS_BIRTH'])
        self.full_df['ann_to_score'] = light_divide(self.full_df['AMT_ANNUITY'], self.full_df['EXT_SOURCE_1'])
        self.full_df['score1_2'] = light_divide(self.full_df['EXT_SOURCE_1'], self.full_df['EXT_SOURCE_2'])
        self.full_df['score1_3'] = light_divide(self.full_df['EXT_SOURCE_1'], self.full_df['EXT_SOURCE_3'])
        self.full_df['score2_3'] = light_divide(self.full_df['EXT_SOURCE_2'], self.full_df['EXT_SOURCE_3'])
        self.full_df['loan_to_ann'] = light_divide(self.full_df['AMT_CREDIT'], self.full_df['AMT_ANNUITY'])
        self.full_df['app_completeness'] = self.full_df.isnull().sum(axis=1)
        
        return self
        
    def generate_interest(self, credit_amt, annuity_amt):
        interest_table = []
        for i in [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]:
            current_interest = ((annuity_amt * i) / credit_amt) ** (1 / (i / 12)) - 1
            if math.isnan(current_interest) or current_interest == -np.inf:
                current_interest = -1
            interest_table.append(current_interest)
        return interest_table

    def generate_interest_stats(self, row):
        row_filtered = [x for x in row if x > common_sense_interest_threshold]
        row_filtered = np.array(row_filtered)

        if row_filtered.size > 0:
            min_interest = np.min(row_filtered)
            max_interest = np.max(row_filtered)
            mean_interest = np.mean(row_filtered)
            median_interest = np.median(row_filtered)
            disp_interest = max_interest - min_interest
            num_interest = row_filtered.size
        else:
            min_interest = max_interest = mean_interest = median_interest = common_sense_interest_threshold
            disp_interest = num_interest = 0
        return [min_interest, max_interest, mean_interest, median_interest, disp_interest, num_interest]

    def generate_cnt_pmt_stats(self, row):
        common_sense_interest_threshold = 0  # Define a sensible threshold
        row_filtered = [(i + 1) * 6 for i, x in enumerate(row) if x > common_sense_interest_threshold]
        row_filtered = np.array(row_filtered)

        if row_filtered.size > 0:
            min_cnt = np.min(row_filtered)
            max_cnt = np.max(row_filtered)
            mean_cnt = np.mean(row_filtered)
            median_cnt = np.median(row_filtered)
            disp_cnt = max_cnt - min_cnt
        else:
            min_cnt = max_cnt = mean_cnt = median_cnt = disp_cnt = np.nan
        return [min_cnt, max_cnt, mean_cnt, median_cnt, disp_cnt]

    def append_interest_features(self):
        interest_table = self.full_df[['AMT_CREDIT', 'AMT_ANNUITY']].apply(lambda x: self.generate_interest(x['AMT_CREDIT'], x['AMT_ANNUITY']), axis=1)
        interest_table_pd = pd.DataFrame(list(interest_table), columns=['ir_' + f for f in ['6', '12', '18', '24', '30', '36', '42', '48', '54', '60']])
        ir_stats = interest_table_pd.apply(self.generate_interest_stats, axis=1)
        ir_stats_pd = pd.DataFrame(list(ir_stats), columns=['min_int', 'max_int', 'mean_int', 'med_int', 'disp_int', 'num_int'])
        cnt_stats = interest_table_pd.apply(self.generate_cnt_pmt_stats, axis=1)
        cnt_stats_pd = pd.DataFrame(list(cnt_stats), columns=["min_cnt", "max_cnt", "mean_cnt", "median_cnt", "disp_cnt"])
        
        self.full_df.reset_index(inplace = True, drop = True)
        self.full_df = pd.concat([self.full_df, ir_stats_pd, interest_table_pd, cnt_stats_pd], axis=1)
        self.full_df['int_to_time_min'] = self.full_df['min_int'] / self.full_df['min_cnt']
        self.full_df['int_to_time_max'] = self.full_df['max_int'] / self.full_df['min_cnt']
        self.full_df['int_to_time_disp'] = self.full_df['disp_int'] / self.full_df['min_cnt']
        
        return self
    
    def add_categorical_counts(self):
        num_train = self.train_df.shape[0]

        for current_feature in self.categorical_variables:
            try:
                v_counts_train = self.full_df.iloc[:num_train][current_feature].value_counts().reset_index()
                v_counts_train.columns = [current_feature, current_feature + '_count_train']

                v_counts_test = self.full_df.iloc[num_train:][current_feature].value_counts().reset_index()
                v_counts_test.columns = [current_feature, current_feature + '_count_test']

                v_counts_total = self.full_df[current_feature].value_counts().reset_index()
                v_counts_total.columns = [current_feature, current_feature + '_count_total']

                self.full_df = self.full_df.merge(v_counts_train, on=current_feature, how='left')
                self.full_df = self.full_df.merge(v_counts_test, on=current_feature, how='left')
                self.full_df = self.full_df.merge(v_counts_total, on=current_feature, how='left')
            except Exception as e:
                print(f"Failed to add counts for {current_feature}: {e}")

        return self

    def process(self):
        self.load_main_data().set_variable_types().fe_main().append_interest_features().add_categorical_counts() 
        gc.collect()
        return self.full_df, self.target_col, self.y, 

class BureauData:
    def __init__(self, path_to_data, sampling):
        self.bureau_df = pd.read_csv(path_to_data + 'bureau.csv')
        self.dataset_name = 'bureau'
        self.agg_map = aggregation_recipes[self.dataset_name]
        self.feature_dfs_to_merge_with_main_df = []
        self.credit_types = ['all', 'Consumer credit', 'Credit card', 'Car loan', 'Mortgage', 'Microloan']
        self.credit_statuses = ['Active', 'Closed']
        self.sampling = sampling

    def preprocess_data(self):
        if self.sampling < 1:
            self.bureau_df = self.bureau_df.sample(frac=self.sampling)
        
        self.bureau_df["missing_info"] = self.bureau_df.isnull().sum(axis=1).astype(np.float32)
        self.bureau_df['credit_duration'] = np.floor_divide(self.bureau_df['DAYS_CREDIT_ENDDATE'] - self.bureau_df['DAYS_CREDIT'], 30).astype(np.float32)
        self.bureau_df['credit_advance'] = light_divide(self.bureau_df['DAYS_CREDIT_ENDDATE'] - self.bureau_df['DAYS_ENDDATE_FACT'], 30)

        for feature in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM_LIMIT']:
            self.bureau_df[f'{feature}_to_credit'] = light_divide(self.bureau_df[feature], self.bureau_df['AMT_CREDIT_SUM'])
        self.bureau_df['repaid_to_credit'] = light_divide(self.bureau_df['AMT_ANNUITY'] * self.bureau_df['credit_duration'], self.bureau_df['AMT_CREDIT_SUM'])
        self.bureau_df['interest'] = (np.power(self.bureau_df['repaid_to_credit'] + 1, 12 / self.bureau_df['credit_duration']) - 1).astype(np.float32)
        guessed_principal = (self.bureau_df["AMT_CREDIT_SUM"] - (self.bureau_df["AMT_CREDIT_SUM_DEBT"].fillna(0) + self.bureau_df["AMT_CREDIT_SUM_LIMIT"].fillna(0))).astype(np.float32)
        self.bureau_df['guessed_annuity_princ'] = light_divide(guessed_principal, -self.bureau_df['DAYS_CREDIT'] / 30)
        self.bureau_df['guessed_annuity_princ_to_act_annuity'] = light_divide(self.bureau_df['guessed_annuity_princ'], self.bureau_df['AMT_ANNUITY'])
        self.bureau_df['princ_to_repay_per_month'] = light_divide(self.bureau_df["AMT_CREDIT_SUM_DEBT"], self.bureau_df["DAYS_CREDIT_ENDDATE"] / 30)
        self.bureau_df['amt_repaid'] = (self.bureau_df['AMT_CREDIT_SUM'] - self.bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(0)).astype(np.float32)

        return self

    def process_segment_of_credit_data(self, credit_type, credit_status, name_prefix):
        filtered = self.bureau_df if credit_type == 'all' else self.bureau_df[self.bureau_df['CREDIT_TYPE'] == credit_type]
        filtered = filtered[filtered["CREDIT_ACTIVE"] == credit_status]
        stats = filtered.groupby("SK_ID_CURR").agg(self.agg_map).astype(np.float32)
        stats.columns = reduce_column_names(stats, name_prefix)
        return stats.reset_index()

    def compute_features_concurrently(self):
        futures=[]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for credit_status in self.credit_statuses: 
                for credit_type in self.credit_types:
                    future = executor.submit(
                        self.process_segment_of_credit_data, credit_status, credit_type, f"{self.dataset_name}_{credit_status}_{credit_type}"
                    )
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                self.feature_dfs_to_merge_with_main_df.append(future.result())
                
        return self
    
    def get_id_mapping(self):        
        return self.bureau_df[['SK_ID_BUREAU', 'SK_ID_CURR']]
    
    def process(self):
        self.preprocess_data().compute_features_concurrently()
        gc.collect()
        return self.feature_dfs_to_merge_with_main_df


class PreviousApplicationData:
    def __init__(self, path_to_data, sampling = 1):
        self.pr_app = pd.read_csv(path_to_data + 'previous_application.csv')
        self.dataset_name = 'previous_app'
        self.agg_map = aggregation_recipes[self.dataset_name]

        self.feature_dfs_to_merge_with_main_df = []
        self.categorical_variables = ["WEEKDAY_APPR_PROCESS_START", "NAME_CASH_LOAN_PURPOSE", "NAME_PAYMENT_TYPE","CODE_REJECT_REASON", "NAME_TYPE_SUITE", 
                                  "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY","NAME_PRODUCT_TYPE",  "CHANNEL_TYPE", "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP", 
                                  "PRODUCT_COMBINATION" ]
        self.sampling = sampling

    def preprocess_data(self):
        if self.sampling < 1:
            self.pr_app = self.pr_app.sample(frac=self.sampling)
        
        self.pr_app['is_x_sell'] = self.pr_app['PRODUCT_COMBINATION'].str.contains('X-Sell').fillna(0)
        self.pr_app['missing_info'] = self.pr_app.isnull().sum(axis=1)
        self.pr_app['active'] = self.pr_app['DAYS_TERMINATION'] > 0

        self.pr_app["credit_to_app"] = light_divide(self.pr_app['AMT_CREDIT'], self.pr_app['AMT_APPLICATION'])
        self.pr_app["credit_to_good"] = light_divide(self.pr_app['AMT_CREDIT'], self.pr_app['AMT_GOODS_PRICE'])
        self.pr_app["annuity_to_good"] = light_divide(self.pr_app['AMT_ANNUITY'], self.pr_app['AMT_GOODS_PRICE'])
        self.pr_app["annuity_to_credit"] = light_divide(self.pr_app['AMT_ANNUITY'], self.pr_app['AMT_CREDIT'])
        self.pr_app["down_to_good"] = light_divide(self.pr_app["AMT_DOWN_PAYMENT"], self.pr_app['AMT_GOODS_PRICE'])

        self.pr_app["days_diff_last_first"] = self.pr_app["DAYS_LAST_DUE"] - self.pr_app['DAYS_FIRST_DUE']
        self.pr_app["days_diff_last_last"] = self.pr_app["DAYS_LAST_DUE"] - self.pr_app['DAYS_LAST_DUE_1ST_VERSION']
        self.pr_app["credit_duration"] = self.pr_app["DAYS_TERMINATION"] - self.pr_app['DAYS_FIRST_DUE']
        self.pr_app["credit_duration2"] = self.pr_app['DAYS_FIRST_DUE'] - self.pr_app["DAYS_FIRST_DRAWING"]

        return self

    def encode_categoricals(self):        
        lbl = preprocessing.LabelEncoder()
        for col in self.categorical_variables:
            self.pr_app[col] = lbl.fit_transform(self.pr_app[col].astype(str))
        enc = preprocessing.OneHotEncoder()
        one_hot_pr_app = enc.fit_transform(self.pr_app.loc[:,self.categorical_variables])
        dense = pd.DataFrame(one_hot_pr_app.todense())
        dense['SK_ID_CURR'] = self.pr_app['SK_ID_CURR']
        dense = dense.groupby('SK_ID_CURR').sum().reset_index()        
        self.feature_dfs_to_merge_with_main_df.append(dense)

        return self

    def compute_xsell_features(self):
        agg_map = {'is_x_sell': ['sum']}
        for lookback_window in [-30, -360, -np.inf]:
            cur_term_stats = self.pr_app[self.pr_app['DAYS_DECISION'] > lookback_window].groupby("SK_ID_CURR").agg(agg_map).astype(np.float32)
            name_prefix = f'{self.dataset_name}_xsell_in_{str(lookback_window)}'
            cur_term_stats.columns = reduce_column_names(cur_term_stats, name_prefix)
            self.feature_dfs_to_merge_with_main_df.append(cur_term_stats.reset_index())

        return self

    def compute_active_closed_features(self, active_flag, name_prefix):
        filtered = self.pr_app[self.pr_app["active"] == active_flag]
        grouped = filtered.groupby("SK_ID_CURR").agg(self.agg_map).astype(np.float32)
        grouped.columns = reduce_column_names(grouped, name_prefix)
        return grouped.reset_index()
    
    def compute_active_closed_features_parallel(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_processes) as executor:
            futures = {
                executor.submit(self.compute_active_closed_features, True, f'{self.dataset_name}_active'),
                executor.submit(self.compute_active_closed_features, False, f'{self.dataset_name}_closed')
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.feature_dfs_to_merge_with_main_df.append(result)
                
        return self
        
    def compute_status_features(self, status):
        agg_map = aggregation_recipes['previous_app']
        status_stats = self.pr_app[self.pr_app['NAME_CONTRACT_STATUS'] == status].groupby("SK_ID_CURR").agg(agg_map).astype(np.float32)
        name_prefix = f'{self.dataset_name}_status_{status}'
        status_stats.columns = reduce_column_names(status_stats, name_prefix)
        
        return status_stats.reset_index()

    def compute_status_features_parallel(self):
        statuses = ["Approved", "Refused"]
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(statuses)) as executor:
            futures = {executor.submit(self.compute_status_features, status) for status in statuses}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.feature_dfs_to_merge_with_main_df.append(result)
        
        return self
    
    def process(self):
        self.preprocess_data().encode_categoricals().compute_xsell_features().compute_status_features_parallel().compute_active_closed_features_parallel()
        gc.collect()
        
        return self.feature_dfs_to_merge_with_main_df


class InstallmentsPaymentsData:
    def __init__(self, path_to_data, sampling = 1):
        self.ip = pd.read_csv(path_to_data + 'installments_payments.csv').sort_values(['SK_ID_PREV', "DAYS_INSTALMENT"])
        self.feature_dfs_to_merge_with_main_df = []
        self.days_in_month = days_in_month  
        self.dataset_name = 'installments_payments'
        self.agg_map = aggregation_recipes[self.dataset_name]  
        self.lb_window_prefix_map = {-np.inf: 'all', -720: '720', -360: '360', -90: '90', -30: '30'}  
        self.version_filters = {None:'all', '0': '0' , '1' : '1', '>=2': '2andmore', '<3': 'first_2', '<6': 'first_5'} 
        self.sampling = sampling
                
    def preprocess_data(self):
        if self.sampling < 1:
            self.ip = self.ip.sample(frac=self.sampling)
        self.ip['delay'] = -(self.ip["DAYS_INSTALMENT"] - self.ip['DAYS_ENTRY_PAYMENT'])
        self.ip['lacking_money'] = np.maximum(self.ip['AMT_INSTALMENT'] - self.ip["AMT_PAYMENT"], 0)
        self.ip['surplus_money'] = np.maximum(self.ip["AMT_PAYMENT"] - self.ip['AMT_INSTALMENT'], 0)

        self.ip['lacking_money_ratio'] = (self.ip['lacking_money'] / self.ip['AMT_INSTALMENT']).replace([np.inf, -np.inf], 0)
        self.ip['surplus_money_ratio'] = np.minimum((self.ip['surplus_money'] / self.ip['AMT_INSTALMENT']).replace([np.inf, -np.inf], 0), 100)

        self.ip["delay_money"] = (self.ip['lacking_money'] * self.ip['delay']) / self.days_in_month
        self.ip["advance_money"] = (self.ip['surplus_money'] * self.ip['delay']) / self.days_in_month

        return self

    def compute_features_for_group(self, group_df, name_prefix):
        stats = group_df.groupby("SK_ID_CURR").agg(self.agg_map).astype(np.float32)
        stats.columns = reduce_column_names(stats, name_prefix)
        return stats.reset_index()

    def compute_features_for_lbwindow_and_version(self, lookback_window, version_filter, prefix):
        # Filter based on lookback window
        filtered = self.ip if lookback_window == -np.inf else self.ip[self.ip.DAYS_INSTALMENT > lookback_window]

        # Apply version filters
        if version_filter is not None:
            if isinstance(version_filter, list):
                filtered = filtered[filtered['NUM_INSTALMENT_VERSION'].isin(version_filter)]
            elif version_filter.startswith('>='):
                version = int(version_filter[2:])
                filtered = filtered[filtered['NUM_INSTALMENT_VERSION'] >= version]
            elif version_filter.startswith('<'):
                version = int(version_filter[1:])
                filtered = filtered[filtered['NUM_INSTALMENT_VERSION'] < version]
            else:
                version = int(version_filter)
                filtered = filtered[filtered['NUM_INSTALMENT_VERSION'] == version]

        # Compute features
        return self.compute_features_for_group(filtered, prefix)

    def compute_features_concurrently(self):
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for lookback_window, lookback_window_prefix in self.lb_window_prefix_map.items():
                for version_filter, version_prefix in self.version_filters.items():
                    future = executor.submit(
                        self.compute_features_for_lbwindow_and_version,
                        lookback_window,
                        version_filter,
                        f"{self.dataset_name}_{lookback_window_prefix}_{version_prefix}",
                    )
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                self.feature_dfs_to_merge_with_main_df.append(future.result())

        return self


    def process(self):
        self.preprocess_data().compute_features_concurrently()
        gc.collect()

        return self.feature_dfs_to_merge_with_main_df


class POSCashBalanceData:
    def __init__(self, path_to_data, sampling = 1):
        self.pos_bal = pd.read_csv(path_to_data + 'POS_CASH_balance.csv')
        self.feature_dfs_to_merge_with_main_df = []
        self.dataset_name = "pos_bal"
        self.agg_map = aggregation_recipes[self.dataset_name]
        self.filter_conditions = {'all',
                                  'first_1', 'first_5', 'first_12', 
                                  'recent_1', 'recent_6', 'recent_12'}
        self.sampling = sampling
                
    def preprocess_data(self):
        if self.sampling < 1:
            self.pos_bal = self.pos_bal.sample(frac=self.sampling)

        self.pos_bal['no_inst'] = self.pos_bal['CNT_INSTALMENT'] - self.pos_bal["CNT_INSTALMENT_FUTURE"]
        return self

    def compute_features_for_group(self, group_df, prefix):
        stats = group_df.groupby("SK_ID_CURR").agg(self.agg_map)
        stats.columns = reduce_column_names(stats, self.dataset_name + prefix)
        return stats.reset_index()

    def compute_features_concurrently(self, filter_conditions):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for condition in self.filter_conditions:
                if condition == 'all':
                    filtered = self.pos_bal
                elif 'recent' in condition:
                    condition_val = -int(condition.replace('recent_', ''))
                    filtered = self.pos_bal[self.pos_bal['MONTHS_BALANCE'] > condition_val]
                else:
                    condition_val = int(condition.replace('first_', ''))
                    filtered = self.pos_bal[self.pos_bal['no_inst'] < condition_val]
                
                future = executor.submit(self.compute_features_for_group, filtered, f'{self.dataset_name}_{condition}')
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                self.feature_dfs_to_merge_with_main_df.append(future.result())

        return self

    def merge_features(self, main_df):
        for feat_df in self.feature_dfs_to_merge_with_main_df:
            main_df = main_df.merge(feat_df, on="SK_ID_CURR", how='left')
        return main_df

    def process(self):
        self.preprocess_data()
        self.compute_features_concurrently(self.filter_conditions)
        gc.collect()

        return self.feature_dfs_to_merge_with_main_df



class CreditCardBalanceData:

    def __init__(self, path_to_data, sampling = 1):
        self.cc_bal = pd.read_csv(path_to_data + 'credit_card_balance.csv')
        self.feature_dfs_to_merge_with_main_df = []
        self.dataset_name = "cc_bal"
        self.agg_map = aggregation_recipes[self.dataset_name]  
        self.filter_conditions = {'all', 'recent_1', 'recent_6', 'recent_12'}
        self.sampling = sampling
                
    def preprocess_data(self):
        if self.sampling < 1:
            self.cc_bal = self.cc_bal.sample(frac=self.sampling)

        self.cc_bal['count_missing'] = self.cc_bal.isnull().sum(axis=1)
        self.cc_bal['bal_to_limit'] = light_divide(self.cc_bal['AMT_BALANCE'] , self.cc_bal['AMT_CREDIT_LIMIT_ACTUAL'])
        self.cc_bal['draw_atm_to_limit'] = light_divide(self.cc_bal["AMT_DRAWINGS_ATM_CURRENT"], self.cc_bal['AMT_CREDIT_LIMIT_ACTUAL'])
        self.cc_bal['draw_pos_to_limit'] = light_divide(self.cc_bal["AMT_DRAWINGS_POS_CURRENT"], self.cc_bal['AMT_CREDIT_LIMIT_ACTUAL'])
        self.cc_bal['draw_other_to_limit'] = light_divide(self.cc_bal["AMT_DRAWINGS_OTHER_CURRENT"], self.cc_bal['AMT_CREDIT_LIMIT_ACTUAL'])

        self.cc_bal['draw_atm_to_min_inst'] = light_divide(self.cc_bal["AMT_DRAWINGS_ATM_CURRENT"], self.cc_bal['AMT_INST_MIN_REGULARITY'])
        self.cc_bal['draw_pos_to_min_inst'] = light_divide(self.cc_bal["AMT_DRAWINGS_POS_CURRENT"], self.cc_bal['AMT_INST_MIN_REGULARITY'])
        self.cc_bal['draw_other_to_min_inst'] = light_divide(self.cc_bal["AMT_DRAWINGS_OTHER_CURRENT"], self.cc_bal['AMT_INST_MIN_REGULARITY'])

        return self

    def compute_features_for_group(self, group_df, prefix):
        stats = group_df.groupby("SK_ID_CURR").agg(self.agg_map).astype(np.float32)
        stats.columns = reduce_column_names(stats, f"{self.dataset_name}_{prefix}")
        return stats.reset_index()

    def compute_features_concurrently(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for condition in self.filter_conditions:
                filtered = self.cc_bal
                if 'recent' in condition:
                    months = int(condition.split('_')[1])
                    filtered = self.cc_bal[self.cc_bal['MONTHS_BALANCE'] > -months]
                
                future = executor.submit(self.compute_features_for_group, filtered, condition)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                self.feature_dfs_to_merge_with_main_df.append(future.result())

        return self

    def merge_features(self, main_df):
        for feat_df in self.feature_dfs_to_merge_with_main_df:
            main_df = main_df.merge(feat_df, on="SK_ID_CURR", how='left')
        return main_df

    def process(self):
        self.preprocess_data().compute_features_concurrently()
        gc.collect()
        return self.feature_dfs_to_merge_with_main_df


class BureauBalanceData:
    def __init__(self, path_to_data, bureau_id_map, sampling = 1):
        self.buro_balance = pd.read_csv(path_to_data + 'bureau_balance.csv')
        self.bureau_id_map = bureau_id_map
        self.dataset_name = 'buro_bal'
        self.agg_map = aggregation_recipes[self.dataset_name] 
        self.feature_dfs_to_merge_with_main_df = []
        self.time_windows = {None: 'all', -1: 'first1', -7: 'first6', -13: 'first12'}
        self.sampling = 1
    
    def preprocess_data(self):
        if self.sampling < 1:
            self.buro_balance = self.buro_balance.sample(frac=self.sampling)
        self.buro_balance = self.buro_balance.merge(self.bureau_id_map, on='SK_ID_BUREAU', how='left')        
        self.buro_balance['SK_ID_CURR'] = self.buro_balance['SK_ID_CURR'].fillna(0).astype(int)
        one_hot = pd.get_dummies(self.buro_balance['STATUS'])
        one_hot.columns = [f + '_col' for f in one_hot.columns.tolist()]
        self.buro_balance = pd.concat([self.buro_balance, one_hot], axis=1)
        return self

    def compute_features_for_group(self, group_df, prefix):
        stats = group_df.groupby("SK_ID_CURR").agg(self.agg_map).astype(np.float32)
        stats.columns = reduce_column_names(stats, prefix)
        return stats.reset_index()

    def compute_features_concurrently(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for window, prefix in self.time_windows.items():
                filtered = self.buro_balance if window is None else self.buro_balance[self.buro_balance['MONTHS_BALANCE'] > window]                
                future = executor.submit(self.compute_features_for_group, filtered, f"{self.dataset_name}_{prefix}")
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                self.feature_dfs_to_merge_with_main_df.append(future.result())

        return self

    def process(self):
        self.preprocess_data().compute_features_concurrently()
        gc.collect()
        return self.feature_dfs_to_merge_with_main_df
