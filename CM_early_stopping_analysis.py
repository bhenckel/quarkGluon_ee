#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:32:34 2018

@author: michelsen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from copy import copy

from xgboost import XGBClassifier
import xgboost


from CM_extra_funcs import CV_EarlyStoppingTrigger

#%%

# metrics_xgb = ['auc', 'error', 'logloss']

num_boost_round = 10000
early_stopping_rounds = 1000

n_estimators = 1000

n_fold = 10
n_data = 1000

cpu_n_jobs = 7

verbose = True
verbose_eval = early_stopping_rounds // 2 if verbose else False

#%%


X, y = make_classification(n_samples=n_data,
                           n_features=10, 
                           n_redundant=2, 
                           n_informative=8,
                           random_state=42
                           )

# y = y.reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42,
                                                    )



#%%

clf = XGBClassifier(n_estimators = n_estimators,
                    learning_rate = 0.1,
                    objective = 'binary:logistic',
                    eval_metric = 'auc',
                    n_jobs = cpu_n_jobs,
                    random_state = 42,
                    silent=True
                    )

params = clf.get_xgb_params()


cv_early_stopping = CV_EarlyStoppingTrigger(stopping_rounds = early_stopping_rounds, 
                                            maximize_score = True,
                                            method = 'xgb')  


Dmatrix_train = xgboost.DMatrix(X_train, label=y_train)


#%%

# run k-fold CV with XGB
cvres = xgboost.cv(params, 
                   Dmatrix_train, 
                   num_boost_round = num_boost_round, 
                   nfold = n_fold, 
                   # metrics = metrics_xgb,
                   metrics = 'auc',
                   early_stopping_rounds = early_stopping_rounds, 
                   # stratified = True, 
                   seed = 42, 
                   shuffle = False,
                   verbose_eval = verbose_eval,
                   callbacks = [cv_early_stopping],
                   )

# get best result:
N_cv_best = cvres['test-auc-mean'].values.argmax()
AUC_cv_best_mean, AUC_cv_best_std = cvres.loc[cvres.index[N_cv_best], 
                                    ['test-auc-mean', 'test-auc-std']]

print(f"\nBest number of estimators based on cross validation: {N_cv_best}")
print(f"ROC for XGB, {n_fold}-fold CV set \t\t " + 
           f"{AUC_cv_best_mean:.5f} +/- {AUC_cv_best_std:.5f} \n")



index = cvres.index
mean = cvres['test-auc-mean']
std = cvres['test-auc-std']
best = np.argmax(mean.values)


plt.figure(figsize=(14, 8))
plt.plot(index, mean, color='r')
plt.fill_between(index, mean+std, mean-std,
                        color='r', interpolate=True, alpha=0.1)
plt.plot(best, mean.iloc[best], 'or')


# plt.figure(figsize=(14, 8))
# plt.plot(index, mean / std, color='r')
# plt.plot(best, mean.iloc[best] / std.iloc[best], 'or')


# clf = copy(clf_org)

#%%

clf.fit(X_train, y_train, verbose=True)
y_pred = clf.predict(X_test)


AUC = np.zeros(n_estimators-1)
for i in range(1, n_estimators):
    y_score = clf.predict_proba(X_test, ntree_limit=i)[:, 1]
    AUC[i-1] = roc_auc_score(y_test, y_score)


N_best = AUC.argmax()
AUC_best_mean = AUC.max()

print(f"\nBest number of estimators based on cross validation: {N_best}")
print(f"ROC for XGB, {n_fold}-fold CV set \t\t {AUC_best_mean:.5f} \n")

plt.figure(figsize=(14, 8))
plt.plot(AUC)
