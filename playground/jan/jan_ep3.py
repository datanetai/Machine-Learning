#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

key = os.getenv('WANDB_KEY')
wandb.login(key=key)
#%% load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# drop id
train.drop('id', axis=1, inplace=True)
target = train['Attrition']
train.drop('Attrition', axis=1, inplace=True)
train.head()

# %% preprocessing
# test train split
# remove features with only one value
for col in train.columns:
    if len(train[col].unique()) == 1:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42,stratify=target)
cols_to_encode = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
for col in cols_to_encode:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        test[col] = le.transform(test[col])
# %% EDA
# check for missing values  
print(X_train.isnull().sum())
# pie chart
plt.figure(figsize=(10,10))
plt.pie(X_train['Attrition'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Attrition')
plt.show()

# %% model
# specify your configurations as a dict
params = {
    'n_estimators': 407,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 100,
    'max_depth': 9,
    'min_data_in_leaf': 46,
    'lambda_l1': 0.01,
    'lambda_l2': 0.6,
    'min_gain_to_split': 1.42,
    'bagging_fraction': 0.45,
    'unbalanced': True,
    'feature_fraction': 0.3}
print('Starting training...')
auc = []
acc = []
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
    
    # train
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[valid_idx], y_train.iloc[valid_idx])],
            eval_metric='auc',
            early_stopping_rounds=50)
    
                                                             
    y_pred = gbm.predict_proba(X_train.iloc[valid_idx], num_iteration=gbm.best_iteration_)[:, 1]
    auc.append(roc_auc_score(y_train.iloc[valid_idx], y_pred))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    acc.append(accuracy_score(y_train.iloc[valid_idx], y_pred))
print('Mean Accuracy:', np.mean(acc))
print('Mean AUC:', np.mean(auc))
print('AUC std:', np.std(auc))

# %% optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-8, 10.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.6),
        'unbalanced': True,
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.4)}
    auc = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=[(X_train.iloc[valid_idx], y_train.iloc[valid_idx])],
                eval_metric='auc',
                early_stopping_rounds=50)
        y_pred = gbm.predict_proba(X_train.iloc[valid_idx], num_iteration=gbm.best_iteration_)[:, 1]
        auc.append(roc_auc_score(y_train.iloc[valid_idx], y_pred))
    return np.mean(auc)

wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name="auc", wandb_kwargs=wandb_kwargs)
wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name="auc", wandb_kwargs=wandb_kwargs)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[wandbc])
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

best_params = {'n_estimators': 954, 'num_leaves': 27, 'max_depth': 9, 'min_data_in_leaf': 33, 'lambda_l1': 0.20442624257062314, 'lambda_l2': 3.4653959013952527, 'min_gain_to_split': 0.7956184960626376, 'bagging_fraction': 0.4545831376346249, 'feature_fraction': 0.36842535432472573}
best_params['boosting_type
lgb = LGBMClassifier(**best_params)
lgb.fit(X_train, y_train)
# test 
y_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration_)[:, 1]
print('AUC', roc_auc_score(y_test, y_pred))
y_pred = np.where(y_pred > 0.5, 1, 0)
print('Accuracy:', accuracy_score(y_test, y_pred))
# %% submit to kaggle
testX = test.drop('id', axis=1)
pred = gbm.predict_proba(testX, num_iteration=gbm.best_iteration_)[:, 1]
#pred = np.where(pred > 0.5, 1, 0)
submission = pd.DataFrame({'id': test['id'], 'Attrition': pred})
submission.to_csv('submission.csv', index=False)

name = 'playground-series-s3e3'
message = "first submission"
! kaggle competitions submit -c $name -f submission.csv -m "$message"
# %%
