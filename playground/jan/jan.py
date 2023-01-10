# %%
# Import libraries
import pandas as pd
import optuna
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns

# lgbm
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
# load data
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

data.head()

# %% simple EDA
numerical_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=[np.object])

num_plots = len(numerical_features.columns)

# calculate the number of rows and columns needed
num_rows = int(np.ceil(np.sqrt(num_plots)))
num_cols = int(np.ceil(num_plots / num_rows))

# draw distribution of numerical features
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
for i, feature in enumerate(numerical_features.columns):
    sns.distplot(numerical_features[feature], ax=axes[i // num_cols, i % num_cols])
plt.show()

# calculate the number of rows and columns needed for the categorical features
if not categorical_features.empty:
    # calculate the number of rows and columns needed for the categorical features
    cat_plots = len(categorical_features.columns)
    cat_rows = int(np.ceil(np.sqrt(cat_plots)))
    cat_cols = int(np.ceil(cat_plots / cat_rows))

    # draw distribution of categorical features
    fig, axes = plt.subplots(cat_rows, cat_cols, figsize=(20, 20))
    for i, feature in enumerate(categorical_features.columns):
        sns.countplot(categorical_features[feature], ax=axes[i // cat_cols, i % cat_cols])
    plt.show()

# draw correlation matrix
corr = numerical_features.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()




# %%
# Data preprocessing
# Drop ID
data.drop('id', axis=1, inplace=True)
# todo: more preprocessing
# Split data
target = 'MedHouseVal'

# Add income_per_person feature
def preprocess_data(data):
   
    return data
data = preprocess_data(data)
test = preprocess_data(test)

X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# find best parameters
def objective(trial):
    # Create an empty dictionary to hold the hyperparameters
    params = {'random_state': 42, 'n_jobs': -1, 'boosting_type': 'gbdt','objective': 'regression','metric': 'rmse'}
    # Extract the current hyperparameters from the trial and add them to the dictionary
    params['max_depth'] = trial.suggest_int('max_depth', 2, 10)
    params['num_leaves'] = trial.suggest_int('num_leaves', 2, 100)
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.5)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.0)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0)

    # Create a LightGBM regressor using the hyperparameters in the dictionary
    model = lgb.LGBMRegressor(**params)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Train the model using the training data
    scores = []
    for train_index, val_index in kfold.split(X_train, y_train):
        model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        y_pred = model.predict(X_train.iloc[val_index])
        rmse = sqrt(mean_squared_error(y_train.iloc[val_index], y_pred))
        scores.append(rmse)
    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_params = study.best_trial.params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['boosting_type'] = 'gbdt'
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'

# %%
params = best_params
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
# %%
# Plot feature importance
lgb.plot_importance(model, max_num_features=10)
plt.show()
# %% save and load model
# save model
model.booster_.save_model('model.txt')
# load model
model = lgb.Booster(model_file='model.txt')

#%% catboost
def objective(trial):
    # Create an empty dictionary to hold the hyperparameters
    params = {'random_state': 42, 'loss_function': 'RMSE','verbose': False}
    # Extract the current hyperparameters from the trial and add them to the dictionary
    params['depth'] = trial.suggest_int('depth', 2, 10)
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.5)
    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0)

    # Create a LightGBM regressor using the hyperparameters in the dictionary
    model = CatBoostRegressor(**params)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Train the model using the training data
    scores = []
    for train_index, val_index in kfold.split(X_train, y_train):
        model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        y_pred = model.predict(X_train.iloc[val_index])
        rmse = sqrt(mean_squared_error(y_train.iloc[val_index], y_pred))
        scores.append(rmse)
    return np.mean(scores)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_params = study.best_trial.params
best_params['random_state'] = 42
best_params['loss_function'] = 'RMSE'
# %%
# train model
best_params = study.best_trial.params
best_params['random_state'] = 42
best_params['loss_function'] = 'RMSE'

params = best_params
model = CatBoostRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
# save
model.save_model('catboost_model')
# %% load
model = CatBoostRegressor()
model.load_model('catboost_model')
model2 = lgb.Booster(model_file='model.txt')
def ensemble_prediction(X):
    prediction1 = model.predict(X)
    prediction2 = model2.predict(X)
    return (prediction1 + prediction2) / 2
# y_pred = ensemble_prediction(X_test)
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE: %f" % (rmse))
ypred2 = model2.predict(X_test)
ypred3 = model.predict(X_test)
print("RMSE2: %f" % (sqrt(mean_squared_error(y_test, ypred2))))
print("RMSE3: %f" % (sqrt(mean_squared_error(y_test, ypred3))))
testX = test.drop('id', axis=1)
y_pred = ensemble_prediction(testX)
# ensembling using blending
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
prediction1 = model.predict(X_val)
prediction2 = model2.predict(X_val)
# Train the blending model on the predictions of the base models and the true labels of the validation set
blending_model = LinearRegression()
blending_model.fit(np.column_stack((prediction1, prediction2)), y_val)
# Make predictions on the test set using the base models
prediction1 = model.predict(X_test)
prediction2 = model2.predict(X_test)
# Make predictions on the test set using the blending model
blending_predictions = blending_model.predict(np.column_stack((prediction1, prediction2)))
# Calculate the RMSE of the predictions made by the blending model
rmse = sqrt(mean_squared_error(y_test, blending_predictions))
print("RMSE: %f" % (rmse))
y_pred = blending_model.predict(np.column_stack((model.predict(testX), model2.predict(testX))))
# %%
# submit to kaggle

submission = pd.DataFrame({'id': test['id'], 'MedHouseVal': y_pred})
submission.to_csv('submission.csv', index=False)
# show submission
# search for kaggle competitions 
name = 'playground-series-s3e1'

# # submit to kaggle
! kaggle competitions submit -c {name} -f submission.csv -m "lgbm"
submission.head()

# %%
