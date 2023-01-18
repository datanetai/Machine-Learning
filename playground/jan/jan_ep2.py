# %%
import pandas as pd
import numpy as np
# seabor
import seaborn as sns
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# lightgbm classifier
from sklearn.model_selection import StratifiedKFold
import optuna
from lightgbm import LGBMClassifier
import wandb
import os
from optuna.integration.wandb import WeightsAndBiasesCallback

key = os.getenv('WANDB_KEY')
!wandb login $key


# %%
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
data = data.drop('id', axis=1)
data.head()


# %%
def preprocess(df):
    # label encoding
    categorical_columns = df.select_dtypes(include='object').columns
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    return df

data = preprocess(data)
test = preprocess(test)



# %%
# target piechart
target = 'stroke'
plt.figure(figsize=(10, 10))
plt.pie(data[target].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Target distribution')
plt.show()

# distribution of other values
numerical_columns = data.select_dtypes(include='float').columns
category_columns = data.select_dtypes(include='int').columns
plt.figure(figsize=(20, 20))
for i, col in enumerate(numerical_columns):
    plt.subplot(4, 3, i + 1)
    sns.histplot(data[col])
plt.show()
plt.figure(figsize=(20, 20))
for i, col in enumerate(category_columns):
    plt.subplot(4, 3, i + 1)
    sns.countplot(data[col])
plt.show()

# %%

plt.figure(figsize=(20, 20))
for i, col in enumerate(numerical_columns):
    plt.subplot(4, 3, i + 1)
    sns.histplot(data[col])

plt.show()
plt.figure(figsize=(20, 20))
for i, col in enumerate(category_columns):
    plt.subplot(4, 3, i + 1)
    sns.countplot(data[col])
plt.show()
    

# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2, random_state=42,stratify=data[target])
# smote oversampling
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# %%
def objective(trial):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 10, 10000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 127),
        'max_depth': trial.suggest_int('max_depth', -1, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 20),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10.0),
        'is_unbalance': True,
        'n_jobs': -1,
        'random_state': 42,
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_index, val_index in kf.split(X_train, y_train):
        X_train1,y_train1,X_val, y_val = X_train.iloc[train_index],y_train.iloc[train_index],X_train.iloc[val_index], y_train.iloc[val_index]
        model = LGBMClassifier(**params)
        model.fit(X_train1, y_train1)
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        auc_scores.append(score)
    return np.mean(auc_scores)
wandb_kwargs = {"project": "kaggle s3e2"}

wandbc = WeightsAndBiasesCallback(metric_name="auc", wandb_kwargs=wandb_kwargs)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100,callbacks=[wandbc])

# %%
wandb.finish()

# %%
best_params = {'n_estimators': 4750, 'learning_rate': 0.006992287893629308, 'num_leaves': 2, 'max_depth': 79, 'min_child_samples': 37, 'subsample': 0.5371657157566819, 'subsample_freq': 2, 'colsample_bytree': 0.5968999253770698, 'reg_alpha': 0.40371115328583657}

best_params['objective'] = 'binary'
best_params['boosting_type'] = 'gbdt'
best_params['metric'] = 'auc'
best_params['is_unbalance'] = True
best_params['n_jobs'] = -1
best_params['random_state'] = 42
model = LGBMClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred)
print(score)

# %%
# pie chart for ypred
pred = model.predict(X_test)
plt.figure(figsize=(10, 10))
plt.pie(pd.Series(pred).value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title('Target distribution')


# %%
# model
# ratio of positive to negative
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(ratio)
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'is_unbalance': True,
    'n_jobs': -1,
    'random_state': 42,
    }
model = LGBMClassifier(**params)
model.fit(X_train, y_train)


# %%
# evaluation
y_pred = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)

# %%
# submit to kaggl
testX = test.drop('id', axis=1)
y_pred = model.predict_proba(testX)[:, 1]
# submission
submission = pd.DataFrame({'id': test['id'], 'stroke': y_pred})
submission.to_csv('submission.csv', index=False)
name_of_competition = 'playground-series-s3e2'
message = 'rf model'
!kaggle competitions submit -c $name_of_competition -f submission.csv -m "$message"

# %%



