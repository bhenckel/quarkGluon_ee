#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:30:50 2018

@author: benjamin
"""
from time import time
import joblib
import pandas as pd
import numpy as np
from CM_extra_funcs_benja import (check_time,
                                  show_metrics,
                                  plot_btags)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.ioff()

if joblib.cpu_count() >=30:
    cpu_n_jobs = 15
else:
    cpu_n_jobs = joblib.cpu_count()-1
    
t_start = time()

saveplots = True

df = pd.read_hdf('MC_btag_sample.h5')
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
y = df_2jet['y'].values
X_3jet = df_3jet[training_labels].values

# Select the first N_sample events for training and the next N_sample events for testing.
if(N_sample > 0):
    X_train = X[:N_sample,:]
    X_test = X[N_sample:(2*N_sample),:]
    y_test = y[N_sample:(2*N_sample)]

# Scaling data for sklRF (it depends greatly on scaling)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sklrf = scaler.transform(X_train)
X_test_sklrf = scaler.transform(X_test)
X_3jet_sklrf = scaler.transform(X_3jet)

checkpoint = check_time(t_start, 'import data and arrange into train/test')

#Importing sklearn RF model and lgbm model.

lgbm_opt = joblib.load('lgb_btag.pkl')
sklrf_opt = joblib.load('sklrf_btag.pkl')

print(f'Imported the two models, they are each printed below.')
print(lgbm_opt)
print(sklrf_opt)

checkpoint = check_time(checkpoint, 'import and print models')

# Use models to predict on the test sets to produce ROC curves (2 jets).
# lgbm
print(f'Predicting labels and probs for LGBM (2 jets)')
lgbm_opt._Booster.reset_parameter({'nthread':cpu_n_jobs})
p_test_lgbm      = lgbm_opt.predict_proba(X_test)[:,1]
y_predicted_lgbm = lgbm_opt.predict(X_test)
check_point = check_time(checkpoint, 'predict labels and probabilities for lgbm (2 jets)')
# sklrf
print(f'Predicting labels and probs for sklRF (2 jets)')
p_test_sklrf      = sklrf_opt.predict_proba(X_test_sklrf)[:,1]
y_predicted_sklrf = sklrf_opt.predict(X_test_sklrf)
check_point = check_time(checkpoint, 'predict labels and probabilities for sklrf (2 jets)')
# Showing Classification report and plotting ROC curves.
show_metrics(y_test, y_predicted_lgbm, p_test_lgbm, 'LGBM', 'b') # labels, labels predicted, probabilities predicted, 'model'
show_metrics(y_test, y_predicted_sklrf, p_test_sklrf, 'sklRF', 'c')
plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC LGBM and sklRF Jet Tagging (2 jet events)')
plt.legend(loc="lower right")
plt.grid()
if(saveplots): plt.savefig('compare_ROC.png')
plt.show()

# Use models to predict on the test sets to produce ROC curves (3 jets).
checkpoint = time()
# lgbm
print(f'Predicting labels and probs for LGBM (3 jets)')
p_lgbm_3jet = lgbm_opt.predict_proba(X_3jet)[:,1]
probs_lgbm_3jet = pd.DataFrame(data={'Btag_prob': p_lgbm_3jet ,'flevt': df_3jet['flevt'] })
check_point = check_time(checkpoint, 'predict labels and probabilities for lgbm (3 jets)')
plot_btags(probs_lgbm_3jet, 'LGBM', saveplots, 'lgm_btag_3jets.png')
# sklrf
print(f'Predicting labels and probs for sklRF (3 jets)')
p_sklrf_3jet = sklrf_opt.predict_proba(X_3jet_sklrf)[:,1]
probs_sklrf_3jet = pd.DataFrame(data={'Btag_prob': p_sklrf_3jet ,'flevt': df_3jet['flevt'] })
check_point = check_time(checkpoint, 'predict labels and probabilities for sklrf (3 jets)')
plot_btags(probs_sklrf_3jet, 'sklRF', saveplots, 'sklrf_btag_3jets.png')

current_time = time()-t_start
print(f'Jobs Done! Time taken in total: {current_time:.2f} seconds')