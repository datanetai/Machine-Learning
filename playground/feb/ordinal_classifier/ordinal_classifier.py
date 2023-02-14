# A Simple Approach to Ordinal Classification
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
class AbstractOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.classes_ = None
        self.estimators_ = None
        
    def fit(self, X, y):
        pass
        
    def predict_proba(self, X):
        unique_classes = np.unique(self.classes_)
        n_classes = len(unique_classes)
        proba = np.zeros((X.shape[0], n_classes))
        for i,y in enumerate(unique_classes):
            if i == 0:
                proba[:,i] = 1 - self.estimators_[i].predict_proba(X)[:,1]
                

            elif i == n_classes - 1:
                proba[:,i] = self.estimators_[i-1].predict_proba(X)[:,1]
            else:
                proba[:,i] = self.estimators_[i-1].predict_proba(X)[:,1] - self.estimators_[i].predict_proba(X)[:,1]
        return proba
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    def score(self, X, y):
        return cohen_kappa_score(y, self.predict(X), weights='quadratic')
    
class OrdinalClassifier(AbstractOrdinalClassifier):
    def __init__(self, base_estimator, n_classes):
        self.base_estimator = base_estimator
        super().__init__(n_classes)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.estimators_ = []
        for i in range(self.n_classes - 1):
            y_ = np.array(y > self.classes_[i], dtype=int)
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y_)
            self.estimators_.append(estimator)
        return self

class OrdinalClassifierLGBM(AbstractOrdinalClassifier):
    def __init__(self, n_classes,params):
        self.params = params
        super().__init__(n_classes)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.estimators_ = []
        for i in range(self.n_classes - 1):
            y_ = np.array(y > self.classes_[i], dtype=int)
            estimator = lgb.LGBMClassifier(**self.params)
            estimator.fit(X, y_)
            
            print("shape of estimator#{}:{}".format(i+1,estimator.predict_proba(X).shape))
            self.estimators_.append(estimator)

            
        return self