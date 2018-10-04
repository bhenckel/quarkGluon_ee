#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:45:53 2018

@author: benjamin
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
from time import time
import joblib
plt.ioff()
from CM_extra_funcs_benja import (SKLRF_Nest_opt,
                                  check_time,
                                  show_metrics,
                                  plot_btags,
                                  btag_df)

if joblib.cpu_count() >=30:
    cpu_n_jobs = 7
else:
    cpu_n_jobs = joblib.cpu_count()-1
# RUN SETTINGS :

opt_Nest  = False
opt_hyp   = False
saveplots = True

t_start = time()

# Importing data and determining training variables.
df = pd.read_hdf('MC_btag_sample.h5')
training_labels = ['ejet', 'costheta', 'projet','sphjet','pt2jet','muljet', 'bqvjet','ptljet']
# Initial settings - these will be pretty narrow for the "simple" RF.
sig_class = 'b'
opt = False
N_sample = 1458683 # 0.2 * length of datasample.
# Creating seperate DataFrames for 2 and 3 jet events.
df_2jet = df[df['njet'] == 2.0] # Select data with njets = 2
df_3jet = df[df['njet'] == 3.0] # Select data with njets = 3

df_2jet = df_2jet.assign(y=np.where(df_2jet['flevt'] == 5,  1, 0)) # Add ones for signal events
# Creating data and targets.
X = df_2jet[training_labels].values
y = df_2jet['y'].values
X_3jet = df_3jet[training_labels].values

# Select the first N_sample events for training and the next N_sample events for testing.
if(N_sample > 0):
    X_train = X[:N_sample,:]
    y_train = y[:N_sample]
    X_test = X[N_sample:(2*N_sample),:]
    y_test = y[N_sample:(2*N_sample)]

# Scale the both training and test sets according to the test set.    
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

checkpoint = check_time(t_start, 'import data and arrange into datasets')
# Opitimize number of trees needed
if(opt_Nest):
    RF = RandomForestClassifier(n_jobs=1)
    df_results, n_opt_est = SKLRF_Nest_opt(RF, X_train, y_train, 10, 500, 5, cpu_n_jobs)

    plt.clf()
    plt.plot(df_results['Trees'], df_results['scores'])
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    if(saveplots): plt.savefig('sklRF_trees.png')
    plt.show()
else:
    n_opt_est = 200 #Somewhat optimized value.

if(opt_hyp):
    param_dist = {"n_estimators": [n_opt_est],
                  "max_depth": [10, 20, 30, None],
                  "max_features": sp_randint(1, 7),
                  "min_samples_split": sp_randint(2,11),
                  "min_samples_leaf": sp_randint(1,11),
                  "bootstrap":[True, False],
                  "criterion":["gini","entropy"]}
    
    rf = RandomForestClassifier(n_jobs=1)
    opt_rf = RandomizedSearchCV(rf, 
                                param_distributions=param_dist, 
                                n_iter=100, 
                                n_jobs=cpu_n_jobs, 
                                cv=5, verbose=5, 
                                random_state=42)
    
    opt_rf.fit(X_train, y_train)
    print(f'Best estimator is {opt_rf.best_estimator_}')
    sklrf_opt = opt_rf.best_estimator_
    joblib.dump(opt_rf.best_estimator_, 'sklrf_btag.pkl')
    checkpoint = check_time(checkpoint, 'optimize hyperparameters based on randomized search')
else:
    sklrf_opt = joblib.load('sklrf_btag.pkl')
    print(f'Selected to not do hyperparameter optimization, using last optimized model')
    print(sklrf_opt)

print(f'Using model to predict on the test set.')

p_test = sklrf_opt.predict_proba(X_test)[:,1]
y_predicted = sklrf_opt.predict(X_test)

#probs_2jet = pd.DataFrame(data={'Btag_prob': p_test ,'flevt': df_2jet['flevt'] })
#probs_2jet[probs_2jet['flevt'] == 5]['Btag_prob'].hist(bins=100)
#plt.show()

show_metrics(y_test, y_predicted, p_test, 'sklRF', 'b')

plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC sklRF Jet Tagging (2 jet events)')
plt.legend(loc="lower right")
plt.grid()
if(saveplots): plt.savefig('sklRF_btag_ROC.png')
plt.show()

print(f'Using model to predict on 3jet events')
X_3jet = scaler.transform(X_3jet)
probs_3jet = btag_df(p_test, df_3jet)
probs_3jet.to_hdf('sklRF_btags_3jet.h5', key='df', mode='w')
plot_btags(probs_3jet, 'sklRF', saveplots, 'sklRF_btag_3jet.png')
current_time = time()-t_start
print(f'Jobs Done! Time taken in total: {current_time:.2f} seconds')