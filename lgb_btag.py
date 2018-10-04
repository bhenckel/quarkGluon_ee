#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:44:48 2018

@author: benjamin
"""
from time import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from CM_extra_funcs_benja import (LGBM_Nest_opt,
                                  check_time,
                                  show_metrics,
                                  plot_btags,
                                  btag_df)
import matplotlib.pyplot as plt
import joblib
plt.ioff()

if joblib.cpu_count() >=30:
    cpu_n_jobs = 7
else:
    cpu_n_jobs = joblib.cpu_count()-1
    
t_start = time()

opt_estimators = False
opt_hyperparam = False
opt_estimators_re = False
saveplots = True

df = pd.read_hdf('/home/benjamin/hep/work/btag/scripts/MC_btag_sample.h5')
training_labels = ['ejet', 'costheta', 'projet','sphjet','pt2jet','muljet', 'bqvjet','ptljet']
# Initial settings - these will be pretty narrow for the "simple" RF.
sig_class = 'b'
N_sample = 1458683 # 0.2 * length of datasample.
# Creating seperate DataFrames for 2 and 3 jet events.
df_2jet = df[df['njet'] == 2.0] # Select data with njets = 2
df_3jet = df[df['njet'] == 3.0] # Select data with njets = 3

df_2jet = df_2jet.assign(y=np.where(df_2jet['flevt'] == 5,  1, 0)) # Add ones for signal events

# Creating data and targets.
X = df_2jet[training_labels].values
X_3jet = df_3jet[training_labels].values
y = df_2jet['y'].values

# Select the first N_sample events for training and the next N_sample events for testing.
if(N_sample > 0):
    X_train = X[:N_sample,:]
    y_train = y[:N_sample]
    X_test = X[N_sample:(2*N_sample),:]
    y_test = y[N_sample:(2*N_sample)]
Dataset = lgb.Dataset(X_train, label=y_train)

checkpoint = check_time(t_start, 'import data and arrange into datasets')

if(opt_estimators):
    Model = lgb.LGBMClassifier(nthreads=cpu_n_jobs, n_estimators = 2000)
    cv_results = LGBM_Nest_opt(Model, Dataset, 2000, 500) #Model, Dataset, num_boost_rounds, early_stopping
    n_opt_estimators = len(cv_results)
    print(f'First optimization of N estimators yields the result: {n_opt_estimators}')
    checkpoint = check_time(checkpoint, 'initial optimization of N estimators')
    fig, ax = plt.subplots()
    ax.plot(cv_results.index,
            cv_results['auc-mean'].values,
            label = 'Number of Estimators')
    ax.fill_between(cv_results.index,
                    cv_results['auc-mean'].values-cv_results['auc-stdv'].values,
                    cv_results['auc-mean'].values+cv_results['auc-stdv'].values,)
    plt.savefig("LGBM_n_estimators.png")
    plt.show()   
else:
    n_opt_estimators = 441 #Last Optimized value.
    print(f'Selected to not optimize N estimators, using last optimized value: {n_opt_estimators}')

if(opt_hyperparam):
    Model = lgb.LGBMClassifier(nthread=1, n_jobs=1, n_estimators=n_opt_estimators, verbose=-1)
    param_grid = {'num_leaves':[15, 31, 63, 127],
                  #'max_depth':[4, 5, 8, -1],
                  'learning_rate':[0.25, 0.2, 0.15, 0.1, 0.5],
                  'subsample':[0.6, 0.7, 0.8, 1.0],
                  'colsample_bytree':[0.6, 0.7, 0.8, 1.0],
                  #'min_split_gain':[],
                  'min_child_weight':[1e-2, 1e-3, 1e-4],
                  'min_child_samples':[10, 20, 30, 40]
                  }
    print(f'Doing Hyperparameter optimization using RandomizedSearch. The param grid is printed below.')
    print(param_grid)    
    RSCV_opt = RandomizedSearchCV(Model, 
                                  param_grid, 
                                  n_iter=100,
                                  n_jobs=cpu_n_jobs,
                                  cv=5,
                                  verbose=10,
                                  random_state=42)
    
    RSCV_opt.fit(X_train, y_train)
    print(f'Best estimator is {RSCV_opt.best_estimator_}')
    lgbm_opt = RSCV_opt.best_estimator_
    joblib.dump(RSCV_opt.best_estimator_, 'lgb_btag.pkl')
    checkpoint = check_time(checkpoint, 'optimize hyperparameters with Randomized Search')
else:
    lgbm_opt = joblib.load('lgb_btag.pkl')
    print(f'Selected to not do hyperparameter optimization, using last optimized model')
    print(lgbm_opt)

if(opt_estimators_re):
    print(f'Selected to re-optimize N estimators, current model is printed below')
    print(lgbm_opt)
    lgbm_opt.set_params(n_estimators = 2000)
    cv_results2 = LGBM_Nest_opt(lgbm_opt, Dataset, 2000, 500)
    
    n_re_opt_estimators = len(cv_results2)
    lgbm_opt.set_params(n_estimators = n_re_opt_estimators)
    joblib.dump(lgbm_opt, 'lgb_btag.pkl')
    print(f'N estimators re-optimized, result for N estimators is: {n_re_opt_estimators}')
    checkpoint = check_time(checkpoint, 'reoptimize N estimators')
    fig, ax = plt.subplots()
    ax.plot(cv_results2.index,
            cv_results2['auc-mean'].values,
            label = 'Number of Estimators')
    ax.fill_between(cv_results2.index,
                    cv_results2['auc-mean'].values-cv_results2['auc-stdv'].values,
                    cv_results2['auc-mean'].values+cv_results2['auc-stdv'].values,)
    plt.savefig("LGBM_n_estimators.png")
    plt.show()
print(f'Using model to predict on the test set.')
lgbm_opt._Booster.reset_parameter({'nthread':cpu_n_jobs})
p_test = lgbm_opt.predict_proba(X_test)[:,1]
y_predicted = lgbm_opt.predict(X_test)
           
show_metrics(y_test, y_predicted, p_test, 'LGBM', 'b')

plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC LGBM Jet Tagging (2 jet systems)')
plt.legend(loc="lower right")
plt.grid()
if(saveplots): plt.savefig('lgb_btag_ROC.png')
plt.show()

print(f'Using model to predict on 3jet events')
p_3jet = lgbm_opt.predict_proba(X_3jet)[:,1]
probs_3jet = btag_df(p_3jet, df_3jet)
probs_3jet.to_hdf('lgb_btags_3jet.h5', key='df', mode='w')
plot_btags(probs_3jet, 'LGBM', saveplots, 'lgb_btag_3jet.png')

current_time = time()-t_start
print(f'Jobs Done! Time taken in total: {current_time:.2f} seconds')