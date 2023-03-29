# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier
# xgboost
import xgboost as xgb
from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
import optuna

import os
from sklearn.model_selection import KFold
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
key = os.environ.get('wandb_key')
print(key)
wandb.login(key=key)




# %%
# load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# drop id
train.drop('id', axis=1, inplace=True)
train.head()

# %%
# some eda
null_values = train.isnull().sum()
if null_values.any():
    print("There are null values in the dataset")
else:
    print("There are no null values in the dataset")

target = 'booking_status'
# distribution of target variable
fig, ax = plt.subplots(figsize=(6, 4))

# Plot countplot of target variable
sns.countplot(x=train[target], palette='Set3', edgecolor='black', ax=ax)

# Set x-axis label
ax.set_xlabel('Booking Status')

# Set y-axis label
ax.set_ylabel('Count')

# Show plot
plt.show()



# %%
# preprocess
def add_more_features(df):
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]

    # Create family_size feature
    df["family_size"] = df["no_of_adults"] + df["no_of_children"]
    
    df["is_repeated_guest"] = (df["repeated_guest"] == 1) & ((df["no_of_previous_cancellations"] > 0) | (df["no_of_previous_bookings_not_canceled"] > 0))
   

    df["total_cost"] = df["avg_price_per_room"] * df["total_nights"]
    # Create adr_per_person feature
    df["adr_per_person"] = df["avg_price_per_room"] / df["family_size"]
    return df

train = add_more_features(train)
test = add_more_features(test)
X_train, X_test, y_train, y_test = train_test_split(train.drop(target, axis=1), train[target], test_size=0.2, random_state=42)

# %%
#  features
features = train.columns.tolist()
print(features)
# %%
# model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc}")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# %%
# hyperparameter tuning
def objective(trial):
    xtrain,xval,ytrain,yval = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
     # define categorical features
    cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]

    # define hyperparameters to tune
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0),
        'random_state': 42,
    }

    # create datasets
    dtrain = lgb.Dataset(xtrain, label=ytrain)
    dvalid = lgb.Dataset(xval, label=yval)

    # train model
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dtrain,dvalid],
                    num_boost_round=10000,
                    early_stopping_rounds=100,
                    verbose_eval=False,
                    categorical_feature=cat_features)

    # get predictions
    preds = gbm.predict(xval)

    # calculate roc_auc score
    roc_auc = roc_auc_score(yval, preds)

    return roc_auc
wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name='auc',  wandb_kwargs=wandb_kwargs)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[wandbc])
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
# %%
# model with best params
params = {'n_estimators': 771, 'num_leaves': 47, 'learning_rate': 0.08164365275225877, 'min_child_samples': 61, 'subsample': 0.7100637335496885, 'colsample_bytree': 0.5873690606514762, 'reg_alpha': 7.973901642384134, 'reg_lambda': 9.027591920823808}
params['objective'] = 'binary'
params['metric'] = 'auc'
params['verbosity'] = -1
params['boosting_type'] = 'gbdt'
params['random_state'] = 42
trainX, valX, trainY, valY = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# define categorical features
cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]
# create datasets
dtrain = lgb.Dataset(trainX, label=trainY)
dvalid = lgb.Dataset(valX, label=valY)
# train model
gbm = lgb.train(params,
                dtrain,
                valid_sets=[dtrain,dvalid],
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=False,
                categorical_feature=cat_features)
# get predictions
preds = gbm.predict(valX)
# calculate roc_auc score
roc_auc = roc_auc_score(valY, preds)
print(f"ROC AUC: {roc_auc}")

#%% # feature importance
lgb.plot_importance(gbm, max_num_features=10, importance_type='gain', figsize=(10, 6))
plt.show()

# catboost
def objective(trial):
    xtrain,xval,ytrain,yval = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
    # define categorical features
    cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]

    # define hyperparameters to tune
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "boosting_type": "Plain",
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        "depth": trial.suggest_int("depth", 2, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0),
        "random_state": 42,
        # use all cpu cores
        "thread_count": -1,

    }

    # create datasets
    # import pool
    dtrain = Pool(xtrain, label=ytrain, cat_features=cat_features)
    dvalid = Pool(xval, label=yval, cat_features=cat_features)
    


    # train model
    model = CatBoostClassifier(**params)
    model.fit(dtrain,
              eval_set=[dvalid],
              early_stopping_rounds=100,
              verbose=False)

    # get predictions
    preds = model.predict_proba(xval)[:, 1]

    # calculate roc_auc score
    roc_auc = roc_auc_score(yval, preds)

    return roc_auc

wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name='auc',  wandb_kwargs=wandb_kwargs)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[wandbc])
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# %%
# model with best params
params = {'n_estimators': 969, 'depth': 5, 'learning_rate': 0.245190956053446, 'l2_leaf_reg': 7.6011550622469315}
params['loss_function'] = 'Logloss'
params['eval_metric'] = 'AUC'
params['verbose'] = False
params['boosting_type'] = 'Plain'
params['random_state'] = 42
params['thread_count'] = -1
trainX, valX, trainY, valY = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# define categorical features
cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]
# create datasets
dtrain = Pool(trainX, label=trainY)
dvalid = Pool(valX, label=valY)
# train model
model = CatBoostClassifier(**params)
model.fit(dtrain,
            eval_set=[dvalid],
            early_stopping_rounds=100,
            verbose=False)
# get predictions
preds = model.predict_proba(valX)[:, 1]
# calculate roc_auc score
roc_auc = roc_auc_score(valY, preds)
print(f"ROC AUC: {roc_auc}")

#%% xgboost
def objective(trial):
    xtrain,xval,ytrain,yval = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
    # define categorical features
    cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]

    # define hyperparameters to tune
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "booster": "gbtree",
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0),
        "random_state": 42,
        # use all cpu cores
        "n_jobs": -1,
    }

    # create datasets
    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dvalid = xgb.DMatrix(xval, label=yval)

    # train model
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=10000,
                      early_stopping_rounds=100,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      verbose_eval=False)

    # get predictions
    preds = model.predict(dvalid)

    # calculate roc_auc score
    roc_auc = roc_auc_score(yval, preds)

    return roc_auc

wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name='auc',  wandb_kwargs=wandb_kwargs)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[wandbc])
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# %%
# model with best params
params = {'n_estimators': 1000, 'max_depth': 2, 'learning_rate': 0.001, 'reg_lambda': 0.0001}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['verbosity'] = 0
params['booster'] = 'gbtree'
params['random_state'] = 42
trainX, valX, trainY, valY = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# define categorical features
cat_features = ["type_of_meal_plan", "room_type_reserved", "arrival_month", "market_segment_type"]
# create datasets
dtrain = xgb.DMatrix(trainX, label=trainY)
dvalid = xgb.DMatrix(valX, label=valY)
# train model
model = xgb.train(params,
                    dtrain,
                    num_boost_round=10000,
                    early_stopping_rounds=100,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    verbose_eval=False)
# get predictions
preds = model.predict(dvalid)
# calculate roc_auc score
roc_auc = roc_auc_score(valY, preds)
print(f"ROC AUC: {roc_auc}")
# test prediction
test_pred = model.predict(xgb.DMatrix(test.drop('id', axis=1)))
# %%
# predict on test data and submission
# test_pred = gbm.predict(test.drop('id', axis=1))
test_pred = model.predict_proba(test.drop('id', axis=1))[:, 1]
submission = pd.read_csv('data/sample_submission.csv')
submission['booking_status'] = test_pred
submission.to_csv('submission.csv', index=False)
comp_name = 'playground-series-s3e7'
os.system(f'kaggle competitions submit -c {comp_name} -f submission.csv -m "submission"')
# %%
