# %%
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna
import datetime

# %%
# loading data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.head()

# %%
# EDA
# No correlation
# year 10000 is outlier
# no missing values
train.shape #(22730, 18)
# missing values
missing_val = train.isnull().sum()
if missing_val.sum() == 0:
    print('No missing values')
else:
    print('Missing values')
    print(missing_val)

# boxplot price and made
plt.figure(figsize=(10, 10))
sns.boxplot(x='made', y='price', data=train)
plt.xticks(rotation=90)
plt.title('Price and make')
plt.show()
# year (made) == 10000 is outlier

train[train['made'] == 10000]
# remove
train = train[train['made'] != 10000]
# %%
# more eda
def iqr_outliers(df):
    out = []
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df:
        if i > Upper_tail or i < Lower_tail:
            out.append(i)
    return out

cols =['squareMeters','floors','cityCode','basement']


# %%

for col in cols:
    out  = iqr_outliers(train[col])
    if len(out) > 0:
        # remove outliers
        train = train[~train[col].isin(out)]
        print('Removed outliers from {}'.format(col))
      



# %%
# preprocess

def engineer_features(df):
    # Total area
    df['totalArea'] = df['squareMeters'] * df['floors']
    
    # Room size
    df['avgRoomSize'] = df['squareMeters'] / df['numberOfRooms']
    
    # Outdoor space
    df['hasOutdoorSpace'] = ((df['hasYard'] == 1) | (df['hasPool'] == 1)).astype(int)
    
    # Building age
    current_year = datetime.datetime.now().year
    df['buildingAge'] = current_year - df['made']
    
    # # Location
    # df['city'] = df['cityCode'].apply(lambda x: 'Paris' if x == 75000 else 'Outside Paris')
    # # label
    # df['city'] = df['city'].astype('category').cat.codes
    # df['cityPartBin'] = pd.cut(df['cityPartRange'], bins=[-1, 1, 3, 5, float('inf')], labels=['Close', 'Near', 'Far', 'Very Far'])
    
    # # Ownership history
    # df['multiOwner'] = (df['numPrevOwners'] > 1).astype(int)
    
    # # Parking space
    # df['hasParkingSpace'] = ((df['garage'] == 1) | (df['hasStorageRoom'] == 1) | (df['hasGuestRoom'] == 1)).astype(int)
    
    # # Roof type
    # df['hasRoofTop'] = ((df['attic'] == 1) & (df['floors'] == 1)).astype(int)
    
    return df

def preprocess(data):
    # drop id
    data = data.drop('id', axis=1)
    # drop cityCode
    
 
    return data




train_data = preprocess(train)
X = train_data.drop('price', axis=1)
y = train_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# model
params = {'n_estimators': 805, 'max_depth': 10, 'learning_rate': 0.09018520409355811, 'num_leaves': 100, 'min_child_samples': 5, 'reg_alpha': 0.5477926395359636, 'reg_lambda': 0.10705678919841011}
params['random_state'] = 42
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# %%
# hyperparameter tuning
def objective(trial):
    # parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1),
   
        'random_state': 42
    }
    # model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

# %%
# submit
test_data = preprocess(test)
y_pred = model.predict(test_data)
submission = pd.DataFrame({'id': test['id'], 'price': y_pred})
submission.to_csv('submission.csv', index=False)
comp_name = 'playground-series-s3e6'
!kaggle competitions submit -c $comp_name -f submission.csv -m "Message"
# %%
