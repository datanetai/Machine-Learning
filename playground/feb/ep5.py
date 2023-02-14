# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lightgbm import LGBMClassifier
import lightgbm as lgb
# regression
from lightgbm import LGBMRegressor
from ordinal_classifier.ordinal_classifier import OrdinalClassifier
from ordinal_classifier.ordinal_classifier import OrdinalClassifierLGBM
from sklearn.metrics import cohen_kappa_score
# random forest
from sklearn.ensemble import RandomForestClassifier
# svm
from sklearn.svm import SVC
from copy import deepcopy
import os
import wandb
import optuna

from optuna.integration.wandb import WeightsAndBiasesCallback

key ='69fa8af555860b6f5f7d7e64161d1117754e4e0d'
wandb.login(key=key)


# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.drop('Id', axis=1, inplace=True)
sample_submission = pd.read_csv('data/sample_submission.csv')
train.head()

# %%
# EDA
# count chart for quality
target = 'quality'
train[target] -= 3
sns.countplot(y=target, data=train,hue=target)
# unique values for quality
train[target].unique()
# check for missing values
missing = train.isnull().sum()
if missing[missing>0].empty:
    print('No missing values')
else:
    print('Missing values')
# distribution of features
train.hist(figsize=(20,20))
# correlation matrix
corr = train.corr()
plt.figure(figsize=(20,20))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')

# %%
# utils
# quadratic weighted kappa, which measures the agreement between two outcomes
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
 

# %%
# modeling
# split data
X = train.drop(target, axis=1)
y = train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
# remove correlated features
def remove_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X = X.drop(X[to_drop], axis=1)
    print('Dropped columns: ', to_drop)
    return X

def add_features(df):
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383685
    df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
    df['free_sulfur/total_sulfur'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    df['sugar/alcohol'] = df['residual sugar'] / df['alcohol']
    df['alcohol/density'] = df['alcohol'] / df['density']
    df['total_acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['sulphates/chlorides'] = df['sulphates'] / df['chlorides']
    df['bound_sulfur'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
    df['alcohol/pH'] = df['alcohol'] / df['pH']
    df['alcohol/acidity'] = df['alcohol'] / df['total_acid']
    df['alkalinity'] = df['pH'] + df['alcohol']
    df['mineral'] = df['chlorides'] + df['sulphates'] + df['residual sugar']
    df['density/pH'] = df['density'] / df['pH']
    df['total_alcohol'] = df['alcohol'] + df['residual sugar']
    
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698
    df['acid/density'] = df['total_acid']  / df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']
    df['sulphates/acid'] = df['sulphates'] / df['volatile acidity']
    df['sulphates*alcohol'] = df['sulphates'] * df['alcohol']
    
    return df

# X_train = remove_correlated_features(X_train)
# X_test = remove_correlated_features(X_test)
def preprocess(X, is_test=False):
    X = add_features(X)
    # if is_test:
    #     X = scaler.transform(X)
    # else:
    #     X = scaler.fit_transform(X)
    return X
X_train = preprocess(X_train)
X_test = preprocess(X_test, is_test=True)




# %%
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Logistic regression accuracy: ', accuracy_score(y_test, y_pred))
print('Logistic regression kappa: ', quadratic_weighted_kappa(y_test, y_pred))



# %%
# ordinal regression
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
ordinal_classifier = OrdinalClassifier(logreg, y.unique())
ordinal_classifier.fit(X_train, y_train)
y_pred = ordinal_classifier.predict(X_test)
print('Ordinal regression accuracy: ', accuracy_score(y_test, y_pred))
print('Ordinal regression kappa: ', ordinal_classifier.score(X_test, y_test))


# %%
# lightgbm
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

print('LightGBM accuracy: ', accuracy_score(y_test, y_pred))
print('LightGBM kappa: ', quadratic_weighted_kappa(y_test, y_pred))


# %%
# ordinal lgb
params = {'num_leaves': 236, 'min_child_samples': 18, 'max_depth': 2, 'learning_rate': 0.026207746905531938, 'subsample': 0.5125867483475446, 'colsample_bytree': 0.46194742279214507, 'reg_alpha': 0.5119842980049739, 'reg_lambda': 0.4216915554849349}
lgb = LGBMClassifier(**params)
ordinal_classifier = OrdinalClassifier(lgb, y.unique())
ordinal_classifier.fit(X_train, y_train)
y_pred = ordinal_classifier.predict(X_test)
print('Ordinal lgb accuracy: ', accuracy_score(y_test, y_pred))
print('Ordinal lgb kappa: ', ordinal_classifier.score(X_test, y_test))

# %%


# %%
# ordinal lightgbm tuning
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.1, 1.0),
        'random_state': 42,
    }
    lgb = LGBMClassifier(**params)
    ordinal_classifier = OrdinalClassifier(lgb, y.unique())
    ordinal_classifier.fit(X_train, y_train)
    y_pred = ordinal_classifier.predict(X_test)
    return ordinal_classifier.score(X_test, y_test)

wandb_kwargs = {"project": "kaggle s3e2"}
wandbc = WeightsAndBiasesCallback(metric_name='kappa',  wandb_kwargs=wandb_kwargs)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, callbacks=[wandbc])
print('Best score: ', study.best_value)

# %%
# submit
X_test = test.drop('Id', axis=1)
X_test = preprocess(X_test)
y_pred = ordinal_classifier.predict(X_test)
y_pred += 3

competition_name = 'playground-series-s3e5'
submission = pd.DataFrame({'Id': test['Id'], 'quality': y_pred})
submission.to_csv(f'{competition_name}.csv', index=False)
message = 'LightGBM'
os.system(f'kaggle competitions submit -c {competition_name} -f {competition_name}.csv -m "{message}"')


# %%



