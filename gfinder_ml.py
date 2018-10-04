#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:57:29 2018

@author: benjamin
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from CM_extra_funcs_benja import (show_metrics)
import matplotlib.pyplot as plt
plt.ioff()
import joblib


if joblib.cpu_count() >=30:
    cpu_n_jobs = 15
else:
    cpu_n_jobs = joblib.cpu_count()-1
    
lgb_data = pd.read_hdf('lgb_events_btag.h5')
sklrf_data = pd.read_hdf('sklrf_events_btag.h5')

X = [lgb_data.values[:,:-1], sklrf_data.values[:,:-1]]
y = [lgb_data['label'].values, sklrf_data['label'].values]

for i in range(2):
    X_dev, X_eval, y_dev, y_eval = train_test_split(X[i],y[i], test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33, random_state=492)
    
    print('Start training...')
    # train
    gbm = lgb.LGBMClassifier(n_estimators=500, nthread=1, n_jobs=1, verbose=-1, random_state=42)
    gbm.fit(X_train, y_train)
    
    RF = RandomForestClassifier(n_estimators = 100, n_jobs=cpu_n_jobs)
    RF.fit(X_train, y_train)
    
    print('Start predicting...')
    # predict
    gbm._Booster.reset_parameter({'nthread':cpu_n_jobs})
    y_pred = gbm.predict(X_test)
    p_test = gbm.predict_proba(X_test)[:,1]
    show_metrics(y_test, y_pred, p_test, 'LGBM', 'b')
    
    y_pred_rf = RF.predict(X_test)
    p_test_rf = RF.predict_proba(X_test)[:,1]
    show_metrics(y_test, y_pred_rf, p_test_rf, 'RF', 'c')
    
    plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC LGBM Jet Tagging (3 jet systems)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()