#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:52:35 2018

@author: benjamin
"""
from time import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

#%%

def error_prop(sigma, cov):
    sigma_z = np.sqrt(sigma[0]**2 + sigma[1]**2 + 2*cov.iloc[0,1])
    return sigma_z

def error_prop_adv(N_bins, cov):
    name = ['eff_b', 'eff_c', 'eff_l']
    vector = np.array([-1 for x in range(N_bins-1)])
    sigma = [0, 0, 0]
    for i in range(len(name)):
        col = [col for col in cov.columns if name[i] in col]
        cov_sliced = cov.loc[col, col]
        sigma[i] = vector @ cov_sliced @ vector.T
    return sigma

def create_MC_matrix(N_bins, f_fit_MC_2j_sym, s_f_fit_MC_2j_sym, f_fit_MC_2j_asym, s_f_fit_MC_2j_asym, chi2_values):
    results = []
    results.append(N_bins)
    results.append(f_fit_MC_2j_sym[0])
    results.append(s_f_fit_MC_2j_sym[0])
    results.append(f_fit_MC_2j_sym[1])
    results.append(s_f_fit_MC_2j_sym[1])
    if(len(s_f_fit_MC_2j_sym) == 3):
        results.append(f_fit_MC_2j_sym[2])
        results.append(s_f_fit_MC_2j_sym[2])
    else:
        results.append(0)
        results.append(0)
    results.append(chi2_values[0])
    results.append(f_fit_MC_2j_asym[0])
    results.append(s_f_fit_MC_2j_asym[0])
    results.append(f_fit_MC_2j_asym[1])
    results.append(s_f_fit_MC_2j_asym[1])
    if(len(s_f_fit_MC_2j_asym) == 3):
        results.append(f_fit_MC_2j_asym[2])
        results.append(s_f_fit_MC_2j_asym[2])
    else:
        results.append(0)
        results.append(0)
    results.append(chi2_values[1])
    return results

def create_data_matrix(N_bins, f_fit_data_2j_sym, s_f_fit_data_2j_sym, f_fit_data_2j_asym, s_f_fit_data_2j_asym, chi2_values):
    results = []
    results.append(N_bins)
    results.append(f_fit_data_2j_sym[0])
    results.append(s_f_fit_data_2j_sym[0])
    results.append(f_fit_data_2j_sym[1])
    results.append(s_f_fit_data_2j_sym[1])
    if(len(s_f_fit_data_2j_sym) == 3):
        results.append(f_fit_data_2j_sym[2])
        results.append(s_f_fit_data_2j_sym[2])
    else:
        results.append(0)
        results.append(0)
    results.append(chi2_values[2])
    results.append(f_fit_data_2j_asym[0])
    results.append(s_f_fit_data_2j_asym[0])
    results.append(f_fit_data_2j_asym[1])
    results.append(s_f_fit_data_2j_asym[1])
    if(len(s_f_fit_data_2j_asym) == 3):
        results.append(f_fit_data_2j_asym[2])
        results.append(s_f_fit_data_2j_asym[2])
    else:
        results.append(0)
        results.append(0)
    results.append(chi2_values[3])
    return results

def even_events_binning(N_bins, data) :
    btag = np.sort(data['nnbjet'].values)
    events_per_bin = int(len(btag)/float(N_bins))
    bin_edges = [0.0] + [btag[x*events_per_bin] for x in range(1,N_bins)] + [1.0]
    return bin_edges

def LGBM_Nest_opt(Model, Dataset, n_boost, early_stop):
        
    cv = lgb.cv(Model.get_params(), Dataset, 
                num_boost_round = n_boost, 
                nfold=5, 
                metrics='auc',
                early_stopping_rounds= early_stop,
                stratified = True,
                verbose_eval = True)
        
    cv_results = pd.DataFrame(cv)
    return(cv_results)

def SKLRF_Nest_opt(RF, X, y, start, end, nstep, cpu_n_jobs):
    
    cv = GridSearchCV(RF,
                      {'n_estimators': np.linspace(start, end, int(((start-end)/nstep)+1), dtype='int')},
                      'accuracy',
                      n_jobs = 1,
                      refit=False,
                      cv=5,
                      verbose=5)
    cv.set_params(n_jobs=cpu_n_jobs)
    cv.fit(X,y)
    
    scores = cv.cv_results_['mean_test_score']
    trees = np.asarray([cv.cv_results_['params'][i]['n_estimators'] for i in range(len(cv.cv_results_['params']))])
    o = np.argsort(trees)
    scores = scores[o]
    trees = trees[o]
    df_results = pd.DataFrame({'Scores': scores,'Trees': trees})
    df_results.to_hdf('ntrees_results.h5', key='df', mode='w')
    n_opt_est = df_results.loc[df_results['Scores'] == max(df_results['Scores']), 'Trees'].iloc[0]
    return(df_results, n_opt_est)
    
def check_time(checkpoint, note):
    current_time = time()-checkpoint
    checkpoint = time()
    print(f'Time taken to {note}: {current_time:.2f} seconds')
    return checkpoint

def show_metrics(y, y_predicted, probs, model, color):
    # Print the Classification report.
    print(classification_report(y, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y,probs)))
    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curves.
    plt.plot(fpr, tpr, lw=1, label=f'{model} ROC (area = {roc_auc:.2f})', color=color)

def plot_btags(df, model, saveplots, filename):
    df[df['flevt'] == 5]['Btag_prob'].hist(bins=100,alpha=1, color = 'b')
    plt.hist([df[df['flevt'] <= 3]['Btag_prob'], df[df['flevt'] == 4]['Btag_prob']], bins=100, range=(0,1), stacked=True, color = ['r', 'm'], alpha=0.7)
    plt.xlabel('Btag Probability')
    plt.ylabel('Count')
    plt.title(f'{model} Btag Probabilities For 3 Jet Events')
    if(saveplots): plt.savefig(filename)
    plt.show()

def btag_df(probs, df):
    df = pd.DataFrame(data={'Btag_prob': probs ,'flevt': df['flevt'], 'qmatch': df['qmatch'] })
    return df