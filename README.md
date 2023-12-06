# kaggle homecredit - 5th place solution

This repository contains the main elements of my ML pipeline to achieve 5th place out of 7176 teams in the [Kaggle Homecredit Competition](https://www.kaggle.com/competitions/home-credit-default-risk).
I participated in this competition in a team with my dear friends [Alvor](https://www.kaggle.com/allvor) and [Mike](https://www.kaggle.com/meykds). Their contributions are not covered in this repo.

## Instructions

### 1. Download the repo

`git clone https://github.com/pawelgodula/kagglehomecredit`

### 2. Install the requirements

`pip install requirements.txt`

(no need if working on a Kaggle Notebook or a Kaggle Python docker image, as of Nov 2023)

### 3. Get the data

Make sure [the competition data](https://www.kaggle.com/competitions/home-credit-default-risk/data) is in the proper `path/to/data` folder

### 4. Run the main pipeline

`python main_pipeline.py python main_pipeline.py --path_to_data "/path/to/data/folder" --path_to_opt_settings "/path/to/opt/settings/folder"`

Example (from running on a Kaggle notebook):

`!python /kaggle/working/kagglehomecredit/src/main_pipeline.py --path_to_data "/kaggle/input/home-credit-default-risk/" --path_to_opt_settings "/kaggle/working/kagglehomecredit/src/opt_settings/"`

### 5. Requirements

By default, the sampling rate is set to 0.01, enabling the pipeline to run on a Kaggle Notebook (with 4 CPUs and 32GB RAM) in approximately 40 minutes. 
Without sampling, it requires 256GB of RAM and takes about 24 hours to complete.

## Solution Architecture

![Homecredit Architecture](https://github.com/pawelgodula/kaggle-homecredit/blob/main/images/homecredit_architecture.png)
Notes:
- The code in this repo covers steps colored in grey.
- In the comments section below I outline key elements of the architecture. 

## Data Structure

![data architecture](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

Main challenges:
- Nested data structure -> I built a custom data processing class for every data source to handle the specificity of the feature engineering
- Feature Engineering: heavy aggregations over multiple time windows -> I used parallelization with `concurrent.futures` to speed up the pipeline

## Comments on the architecture

There are a few main ideas that we think we have done differently vs. others, and which gave us the lead on public LB.

1. Using deep learning to extract interactions among different data sources
2. Using “nested models”, with (2.1) a different approach to credit card balance features
3. Using a model to predict the interest rate/duration of loans in current applications

Below is a detailed description of all the points above:

### 1. Using deep learning to extract interactions among different data sources

We wondered how we could capture the interactions between signals coming from different data sources. For example, what if 20 months ago someone was rejected in the external Bureau, had a late payment in installment payments, and applied for a loan at Homecredit? These types of interactions are very hard to capture by humans because of the number of possible options. So, we turned to deep learning and turned this problem into an image classification problem. 

How? Below is the sample “user image” that we fed to the neural network:
![user image for nn](https://github.com/pawelgodula/kaggle-homecredit/blob/main/images/homecredit-user-image-for-nn.png)

One can see in the image above that we created a single vector of user characteristics coming from different data sources for every month of user history, going as far as 96 months into the past (8 years was a cutoff in most data sources). Then we stacked those vectors and created a very sparse “user image”.

We used the following CONV NN architecture :

![NN architecture](https://github.com/pawelgodula/kaggle-homecredit/blob/main/images/homecredit_nn_architecture.png)

This model scored 0.72 AUC on CV, without any features from the current application. It trained rather quickly = around 30 mins on GTX 1080. We have put oof predictions from this model as a feature into our LGBM model, which gave us around 0.001 on CV (an improvement on an already very strong model with >3000 features) and 0.001 on LB. This means that the network was able to extract some information on top of >3000 hand-crafted features.

### 2. Using nested models

One of the things that bothered us throughout the competition was the somehow arbitrary nature of various group-bys that we performed on data. For example, we supposed that an overdue installment 5 years ago is less important than 1 month ago, but what is the exact relationship? The traditional way is to test different thresholds using a cv score, but there is also a way for the model to figure it out.

In order to explain our approach, I will use an example of installment_payments data source.
What we did:

Add a `“TARGET”` column to installment_payments, by merging on SK_ID_CURR with application_train
Run a LGBM model to predict which records have 0 or 1 target. This step allows a model to abstract “what type of behavior in installment payments generally leads to default in loan in current_application”. You can see in the example below, that the same “fraudulent” behavior (like `missing money = installment amount - payment amount`) a long time ago received a lower score compared to more recent similar behavior. We concluded then that the model learned to identify fraudulent behavior at the “behavior” level (without the need to aggregate or set time thresholds).

![Nested models predictions](https://github.com/pawelgodula/kaggle-homecredit/blob/main/images/homecredit-nested-image.png)

To the best of our knowledge as seasoned Kagglers, this is a unique approach to using LGBM to encode the temporal importance of behaviors. 

From step 2 we receive a bunch of OOF predictions for every SK_ID_CURR on row-level in installment payments. Then we can aggregate: `min, max, mean, median` etc., and attach them as features to the main model.
We ran the same procedure on all data sources (`previous application, credit card balance, pos cash balance, installment payments, bureau, bureau balance`). OOF aggregated features on all of them added value, apart from Bureau Balance, which actually decreased cv and we didn’t use it in the end.
We received very low auc on those “nested models”:

- previous application: 0.63
- credit card balance: 0.58
- pos cash balance: 0.54
- installment payments: 0.58
- bureau: 0.61
- bureau balance: 0.55

The low AUC scores for these models were hardly surprising, as they carry an enormous amount of noise. Even for default clients, the majority of their behaviors are OK. The point is to identify those few behaviors that are common across defaulters.

This gave us a 0.002 improvement on CV / 0.004 improvement on LB, on our strong model with >3000 features.

#### 2.1 Another approach to credit card balance features

We have observed that there is a significant difference in train and test sets in ‘loan types’.
The cash_loans to revolving_loans ratio in the train and test sets is respectively ~90/10 and ~99/1. That is why we have decided to use features generated on the credit card balance set only in one final submission.

Our observation was that in the case of the ‘nested feature’ generated on credit card balance set by one model was 0.58, in another case (using a different approach of Light GBM) the AUC score was ~0.64.
Despite the fact that only one-third of the train and test sets we labeled by this feature we have trained the model with only one additional feature and the improvement on the CV was shocking!
From 0.805 to 0.811! This corresponds with the ratio we have mentioned before. It appears to be confirmed in our final result - that it is an overfit caused by cash_loans / revolving_loans ratio.

### 3. Model to predict the interest rate (THE TRICK)

We have been wondering for a long time why the organizers did not share cnt_payments for current_application. We finally came up with the following reasoning - if we knew the amount of credit, amount of annuity, and number of payments, we can calculate the interest rate from the following formula:

`Annuity x CNT payments / Amount of Credit = (1+ir) ^ (CNT payment /12)`

This formula assumes annual payments, while in reality the payments were done in a monthly fashion, but it was good enough as a proxy.

The interest rate is the measure of risk that a current Homecredit model assigns to a customer (especially within a given duration of the loan). Hence, knowing interest rate means to some extent knowing the risk assessment from the current HomeCredit model. From that moment on we set out on a journey to guess the interest rate of loans in train and test sets.

#### 3.1 First iteration: prediction of interest rate based on business understanding of credit products

The key to predicting interest rates is to understand that credit duration (or CNT payment) is not a continuous variable - it belongs to a very specific set of values. When you look at cnt_payments from the previous application, you see that the majority of loans have a duration which is a multiple of 6. It makes sense - you don’t take a loan for 92.3 days, rather you take it for half a year, one year, 2 years, etc. So, at the very beginning, I assumed that duration can belong to the following set of values (in months): `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]`.

What can you do now? Let me explain it using a loan with a `35.000` credit value and `1.000` annuity (both of these values are known for applications in train and test). What you can do, is iterate over different durations and check if the interest rate makes sense.

Have a look at the below table. From this table you can see that only two interest rate values make sense: `10%` or `11%` (I assumed, based on previous applications, that around `8.5%` is the lowest interest rate that a loan may get at Homecredit). 

![it table](https://storage.googleapis.com/kaggle-forum-message-attachments/379104/10212/ir_table.jpg)

Which one is correct? We don’t know, so we can compute various characteristics of “possible interest rates”, like `min, max, median`, etc, and let LGBM find the best correlation with TARGET. It turned out, that a minimum value of such a “possible” set of interest rates was a very good predictor of the actual interest rate, until…

#### 3.2. .. we understood, that you can actually use previous applications to build a model to predict interest rates.

For the majority of previous applications, you can calculate interest rates, because you have `cnt_payment`, `credit amount`, and `annuity`. Then, you can treat the interest rate calculated in such a way as a Target, and use features common between the current application and previous application as features to explain it. But still, the most important features are those from the table above: `min, max, mean, and median` over a possible set of interest values. Features like `credit amount, annuity amount` etc. are only ‘helper’ features, letting the model choose properly between different possible values of interest rates.

We achieved 0.02 rmse on the model to predict interest rate and 0.50 rmse on the model to predict duration, both based on oof predictions tested on previous applications. These models used only features common between the previous application and the current application so that we could use them to predict interest rate and duration for loans in the current application.

Interest rate prediction gave us a 0.002 improvement on CV/ 0.004 improvement on LB, on our strong model with >3000 features.

### 4. Feature selection

We did a lot of hand-crafted features (around 8.000), out of which we selected around 3.000 using [this mechanism](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances). In this repo, it is substituted with a simple mechanism that also works well: if a feature has zero importance in at least one of the folds (on a model trained with `feature_fraction = 1.0`), it can be removed.

## Lessons learned

In retrospect, our key mistake was that we used a weighted average of 3 variations of a single model, but the differences between them were so small, that the weighted average scored the same as the single model. We put all of our effort into feature engineering. We consider it a major omission, which led to a drop from #1 on Public LB to #5 on Private LB. [The winning team](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821) used 3-level stacking with ~90 base models. 

## Final remarks
- For a full discussion of the solution please see [the Kaggle forum](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64625)

## About the author:
[Narsil - Kaggle Competitions Grandmaster](https://www.kaggle.com/narsil), the founder of https://jobs-in-data.com/ - a job search engine for data scientists.
