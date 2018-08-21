#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:54:58 2018

@author: christian michelsen 
"""

# to ignore deprecation warnings:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

#imports 

import numpy as np
import matplotlib.pyplot as plt
#import uproot
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from copy import copy



from CM_extra_funcs import (
                                              color_dict,
        
                                              load_data_file,
                                              
                                              # train_test_split_non_random,
                                              train_test_index,
                                              
                                              check_event_numbers_are_matching,
                                              check_qmatch_are_correct,
                                              
                                              CV_EarlyStoppingTrigger,
                                              plot_cv_res,
                                              
                                              degrees_of_freedom,
                                              
                                              get_dict_split,
                                              get_dict_flavor,
                                              # SeriesWrapper,
                                              # DataFrameWrapper,
                                              PandasContainer,
                                              )


#%% ===========================================================================
#  Initial parameters
# =============================================================================

down_sample = 0.01
verbose = False
cpu_n_jobs = 7
create_plots = True
save_plots = False
create_pairgrid_plot = False
close_figure_after_saving = False


branches_to_import = None 
filename = './data/outputfile_Ref12.root'
treename_data = 'ntuple_DataBtag'
treename_MC = 'ntuple_McBtag'

# use these columns as variables
MC_cols = ['bqvjet', 'muljet', 'projet', 'pt2jet', 'ptljet', 'sphjet', 'phijet']


plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.figsize': (14,8)})



#%% ===========================================================================
#  Load data and split into 2- and 3-jet events 
# =============================================================================

# load real data
df_data_all = load_data_file(filename, treename_data, branches_to_import, 
                         verbose, save_type='hdf5')
# load MC
df_MC_all = load_data_file(filename, treename_MC, branches_to_import, 
                       verbose, save_type='hdf5')

#keep all original data
df_data = df_data_all.copy()
# TODO notice here that we do not split on 'datatype'
# df_MC = df_MC_all[df_MC_all.datatype == 1].copy() # normal MC 
# df_MCb = df_MC_all[df_MC_all.datatype == 2].copy() # MC b, i.e. extra b-events
df_MC = df_MC_all.copy() # normal MC 


# downsample data if needed to 'down_sample'
if down_sample is not None:
    print(f'\n\n Downsampling by {down_sample*100:.2f}%\n\n')
    df_MC = df_MC.iloc[:int((len(df_MC)*down_sample)//2)*2]

# get 2-jet events and 3-jet
df_MC_2j = df_MC[df_MC.njet == 2]
df_MC_3j = df_MC[df_MC.njet == 3]


# check that the two jets share same event number:
check_event_numbers_are_matching(df_MC_2j)
check_event_numbers_are_matching(df_MC_3j)

# TODO qmatch, check_qmatch_are_correct


#%% ===========================================================================
#  Original DF to X, y and dicts using DataFrameContainer dfc 
# =============================================================================

# use the MC columns as variables:
X_MC_2j = df_MC_2j.loc[:, MC_cols]
X_MC_3j = df_MC_3j.loc[:, MC_cols]

# and use flevt as y-value
flevt_MC_2j = df_MC_2j.loc[:, 'flevt']
flevt_MC_3j = df_MC_3j.loc[:, 'flevt']

# set all values where flevt is 5 to 1 (ie. bb), 4 (ie. cc) to 2 and else 0
y_MC_2j_3f = flevt_MC_2j.where(4 <= flevt_MC_2j, 0).replace(4, 2).replace(5, 1)
y_MC_3j_3f = flevt_MC_3j.where(4 <= flevt_MC_3j, 0).replace(4, 2).replace(5, 1)

# take 3-flavor events and make into 2-flavor by replacing 2's with 0's
y_MC_2j_2f    = y_MC_2j_3f.replace(2, 0)
y_MC_3j_2f    = y_MC_3j_3f.replace(2, 0)

# get training and test indices for MC
index_MC_2j_train, index_MC_2j_test = train_test_index(X_MC_2j, test_size=0.20)
index_MC_3j_train, index_MC_3j_test = train_test_index(X_MC_3j, test_size=0.20)

# get dictionaries of indices for flavours (b, c, l, cl, all) 
# and splits (train, test, all)
dict_flavor_MC_2j = get_dict_flavor(y_MC_2j_3f)
dict_split_MC_2j = get_dict_split(y_MC_2j_3f.loc[index_MC_2j_train], 
                                  y_MC_2j_3f.loc[index_MC_2j_test])

dict_flavor_MC_3j = get_dict_flavor(y_MC_3j_3f)
dict_split_MC_3j = get_dict_split(y_MC_3j_3f.loc[index_MC_3j_train], 
                                  y_MC_3j_3f.loc[index_MC_3j_test])


# make dataframecontainer using own class PandasContainer. 
# Takes dataframe/series as input and the dictionaries of flavors and splits
dfc_X_MC_2j = PandasContainer(X_MC_2j, dict_flavor_MC_2j, dict_split_MC_2j, 
                              max_rows=5, max_cols=7)
dfc_y_MC_2j_3f = PandasContainer(y_MC_2j_3f, dict_flavor_MC_2j, dict_split_MC_2j) 
dfc_y_MC_2j_2f = PandasContainer(y_MC_2j_2f, dict_flavor_MC_2j, dict_split_MC_2j) 

dfc_X_MC_3j = PandasContainer(X_MC_3j, dict_flavor_MC_3j, dict_split_MC_3j)
dfc_y_MC_3j_3f = PandasContainer(y_MC_3j_3f, dict_flavor_MC_3j, dict_split_MC_3j) 
dfc_y_MC_3j_2f = PandasContainer(y_MC_3j_2f, dict_flavor_MC_3j, dict_split_MC_3j) 


# get nnbjet for 2- and 3-jet events
nnbjet_2j       = df_MC_2j.loc[:, 'nnbjet']
dfc_nnbjet_2j = PandasContainer(nnbjet_2j, dict_flavor_MC_2j, dict_split_MC_2j) 
nnbjet_3j       = df_MC_3j.loc[:, 'nnbjet']
dfc_nnbjet_3j = PandasContainer(nnbjet_3j, dict_flavor_MC_3j, dict_split_MC_3j) 


#%% ===========================================================================
#  Create initial overview plots 
# =============================================================================


print("ROC for org b-tag, training set \t", roc_auc_score(dfc_y_MC_2j_2f['train'], 
                                                          dfc_nnbjet_2j['train']))
print("ROC for org b-tag, test set \t\t", roc_auc_score(dfc_y_MC_2j_2f['test'], 
                                                        dfc_nnbjet_2j['test']))

# histogram of nnbjet value for b,c,l,cl flavors 
if create_plots:
    fig_nnbjet, ax_nnbjet = plt.subplots(figsize=(10, 10))
    
    for flavor, flavor_df, color in dfc_nnbjet_2j.iterflavors(include_cl=True,
                                                              include_color=True):
        ax_nnbjet.hist(flavor_df, 100, range=(0, 1), 
                       histtype='step', 
                       color=color_dict[color],
                       label=flavor)
    
    ax_nnbjet.set(xlabel='b-tag', ylabel='Counts', title='Histogram of b-tags')
    ax_nnbjet.legend(loc='upper left')
    
    if save_plots:
        fig_nnbjet.savefig('./figures/nnbjet_histogram.pdf', dpi=300)
        if close_figure_after_saving:
            plt.close('all')
    


# pair grid plots of the input variables and b-tags. Quite slow to run.
if create_pairgrid_plot and create_plots:
    # pair grid plots of the input variables: kde-plots, scatter_plots and kde-histograms
    g_inputvars = sns.PairGrid(dfc_X_MC_2j['train'].sample(10_000, random_state=42), 
                     diag_sharey=False)
    g_inputvars.map_lower(sns.kdeplot, cmap='Blues_d', n_levels=6) 
    g_inputvars.map_upper(plt.scatter, s=0.2, alpha=0.2)
    g_inputvars.map_diag(sns.kdeplot, lw=2)
    if save_plots:
        g_inputvars.savefig('./figures/pairgrid_input_vars.pdf', dpi=300)
        if close_figure_after_saving:
            plt.close('all')
    
    # plot of b-tags as 2d plot with marginal distributions on the axis
    g_btags = sns.JointGrid(x=nnbjet_2j[::2], y=nnbjet_2j[1::2], dropna=False,
                       xlim=(0, 1), ylim=(0, 1), space=0, size=10)
    g_btags = g_btags.plot_joint(plt.scatter, s=1)
    # g_btags = g_btags.plot_joint(sns.kdeplot, cmap="Blues_d", n_levels=3)
    g_btags = g_btags.plot_marginals(sns.distplot, bins=100, kde=False)
    g_btags.set_axis_labels("b-tag", "b-tag")
    if save_plots:
        g_btags.savefig('./figures/pairgrid_btags.pdf', dpi=300)
        g_btags.savefig('./figures/pairgrid_btags.png', dpi=100)
        if close_figure_after_saving:
            plt.close('all')



# overview of input variables as histograms
if create_plots:

    fig_overview, ax_overview = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
    ax_overview = ax_overview.flatten()
    for flavor, df_flavor in dfc_X_MC_2j.iterflavors():
        for i, (name, col) in enumerate(df_flavor.iteritems()):
            ax_overview[i].hist(col, 50, label=flavor, histtype='step', log=False)
            ax_overview[i].set(xlabel=name)
    ax_overview[-1].set_visible(False)
    ax_overview[0].legend(loc='upper right')
    
    fig_overview.tight_layout()
    if save_plots:
        fig_overview.savefig('./figures/variables_overview.pdf', dpi=300)
        if close_figure_after_saving:
            plt.close('all')


#%% ===========================================================================
#  Initiate XGBoost and LightGBM 
# =============================================================================


num_boost_round = 10000
early_stopping_rounds = 1000

n_fold = 5
n_sigma = 1


from xgboost import XGBClassifier
import xgboost

from lightgbm import LGBMClassifier
import lightgbm



clf_org_xgb = XGBClassifier(  n_estimators = 1000,
                          learning_rate = 0.1,
                          objective = 'binary:logistic',
                          eval_metric = 'auc',
                          # base_score = proportion_2j,
                          n_jobs = cpu_n_jobs,
                          random_state = 42,
                          silent=True
                          )

clf_org_lgb = LGBMClassifier(  n_estimators = 1000,
                          learning_rate = 0.1,
                          objective = 'binary',
                          n_jobs = cpu_n_jobs,
                          random_state = 42,
                          silent=True
                          )

xgb_params = clf_org_xgb.get_xgb_params()

lgb_params = clf_org_lgb.get_params()
lgb_params.pop('n_estimators')
lgb_params.pop('silent')


xgb_cv_early_stopping = CV_EarlyStoppingTrigger(
                                        stopping_rounds = early_stopping_rounds, 
                                        maximize_score = True,
                                        method = 'xgb')  

lgb_cv_early_stopping = CV_EarlyStoppingTrigger(
                                        stopping_rounds = early_stopping_rounds, 
                                        maximize_score = True,
                                        method = 'lgb')  


#%% ===========================================================================
#  XGboost, 2-jet 2-flavour events 
# =============================================================================

print("\n")

# define training matrix and verbosity level 
xgb_trainDmatrix_2j_2f = xgboost.DMatrix(dfc_X_MC_2j['train'], 
                                         label=dfc_y_MC_2j_2f['train'])
verbose_eval = early_stopping_rounds // 2 if verbose else False

# run k-fold CV with XGB
cvres_xgb_2j_2f = xgboost.cv(  xgb_params, 
                   xgb_trainDmatrix_2j_2f, 
                   num_boost_round = num_boost_round, 
                   nfold = n_fold, 
                   metrics = ['auc'],
                   early_stopping_rounds = early_stopping_rounds, 
                   stratified = True, 
                   seed = 42, 
                   shuffle = False,
                   verbose_eval = verbose_eval,
                   callbacks = [xgb_cv_early_stopping],
                   )

# get best result:
N_est_cv_best_xgb_2j_2f = cvres_xgb_2j_2f['test-auc-mean'].values.argmax()
AUC_best_mean_xgb_2j_2f, AUC_best_std_xgb_2j_2f = cvres_xgb_2j_2f.loc[
                                    cvres_xgb_2j_2f.index[N_est_cv_best_xgb_2j_2f], 
                                    ['test-auc-mean', 'test-auc-std']]

print(f"Best number of estimators based on cross validation: {N_est_cv_best_xgb_2j_2f}")
str_xgb_2j_2f = (f"ROC for XGB-tag, {n_fold}-fold CV set \t\t " + 
           f"{AUC_best_mean_xgb_2j_2f:.5f} +/- {AUC_best_std_xgb_2j_2f:.5f}")
print(str_xgb_2j_2f)


# fit normal xgb classfier using the best number of estimators from CV
clf_xgb_2j_2f = copy(clf_org_xgb)
clf_xgb_2j_2f.set_params(n_estimators=N_est_cv_best_xgb_2j_2f)

clf_xgb_2j_2f.fit(dfc_X_MC_2j['train'], 
                  dfc_y_MC_2j_2f['train'], 
                  eval_metric = ['error', 'auc'], 
                  verbose = False,
                  eval_set = [((dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])), 
                               (dfc_X_MC_2j['test'], dfc_y_MC_2j_2f['test'])])

# predict b-tags and scores
y_pred_xgb_2j_2f_test = clf_xgb_2j_2f.predict(dfc_X_MC_2j['test'])
y_scores_xgb_2j_2f_test = clf_xgb_2j_2f.predict_proba(dfc_X_MC_2j['test'])[:, 1]
y_scores_xgb_2j_2f_train = clf_xgb_2j_2f.predict_proba(dfc_X_MC_2j['train'])[:, 1]

print("ROC for XGB-tag, test set \t\t", roc_auc_score(dfc_y_MC_2j_2f['test'], y_scores_xgb_2j_2f_test))


y_scores_xgb_2j_2f_test = pd.Series(y_scores_xgb_2j_2f_test, index=dfc_X_MC_2j['test'].index)
y_scores_xgb_2j_2f_train = pd.Series(y_scores_xgb_2j_2f_train, index=dfc_X_MC_2j['train'].index)
y_scores_xgb_2j_2f = y_scores_xgb_2j_2f_train.append(y_scores_xgb_2j_2f_test)

dfc_y_scores_xgb_2j = PandasContainer(y_scores_xgb_2j_2f, 
                                      dict_flavor_MC_2j, dict_split_MC_2j)


# histogram of b-tags for nnbjet (org) and for XGBoost 
if create_plots:
    
    fig_btag_hist, ax_btag_hist = plt.subplots(figsize=(12, 8))
    
    ax_btag_hist.hist(dfc_nnbjet_2j['b', 'test'], 100, range=(0, 1), histtype='step',
            density=True, label='nnbtag b_test', color=color_dict['blue'])
    ax_btag_hist.hist(dfc_nnbjet_2j['b', 'train'], 100, range=(0, 1), histtype='step',
            density=True, label='nnbtag b_train', color=color_dict['blue'], linestyle='dashed')
        
    ax_btag_hist.hist(dfc_nnbjet_2j['cl', 'test'], 100, range=(0, 1), histtype='step',
            density=True, label='nnbtag cl_test', color=color_dict['orange'])
    ax_btag_hist.hist(dfc_nnbjet_2j['cl', 'train'], 100, range=(0, 1), histtype='step',
            density=True, label='nnbtag cl_train', color=color_dict['orange'], linestyle='dashed')
    
    ax_btag_hist.hist(dfc_y_scores_xgb_2j['b', 'test'], 100, range=(0, 1), histtype='step',
            density=True, label='xgb b_test', color=color_dict['red'])
    ax_btag_hist.hist(dfc_y_scores_xgb_2j['b', 'train'], 100, range=(0, 1), histtype='step',
            density=True, label='xgb b_train', color=color_dict['red'], linestyle='dashed')
    
    ax_btag_hist.hist(dfc_y_scores_xgb_2j['cl', 'test'], 100, range=(0, 1), histtype='step',
            density=True, label='xgb cl_test', color=color_dict['green'])
    ax_btag_hist.hist(dfc_y_scores_xgb_2j['cl', 'train'], 100, range=(0, 1), histtype='step',
            density=True, label='xgb cl_train', color=color_dict['green'], linestyle='dashed')
    
    ax_btag_hist.set(xlabel='b-tag', ylabel='Normalized Counts', 
           title='Histogram of b-tags for NN and XGB')
    
    ax_btag_hist.legend(loc='upper center')
    
    if close_figure_after_saving:
            plt.close('all')



# roc curve for xgboost compared to nnbjet (org)
if create_plots:
    
    fig_roc_curve, ax_roc_curve = plt.subplots(figsize=(12, 8))
        
    for name, y_btag, color in zip(['nnbtag', 'xgb'], 
                                   [dfc_nnbjet_2j['test'], y_scores_xgb_2j_2f_test],
                                   ['blue', 'red']):
            
        fpr, tpr, thresholds = roc_curve(dfc_y_MC_2j_2f['test'], y_btag, pos_label=1)
        
        signal_eff = tpr
        background_rej = 1 - fpr
        # background_acc = 1 - background_rej
        
        roc_auc = auc(fpr, tpr)
        
        ax_roc_curve.plot(signal_eff, background_rej, color=color_dict[color], 
                label=f'{name}, AUC = {roc_auc:0.4f})')
        
    ax_roc_curve.plot([0, 1], [0, 1], color=color_dict['k'], lw=1, linestyle='--')
    
    ax_roc_curve.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], 
           xlabel='Signal Efficiency', ylabel='Background Rejection',
           title='ROC-curve')
    
    ax_roc_curve.legend(loc="lower right")
    
    
    if save_plots:
        fig_roc_curve.savefig('./figures/ROC_curve.pdf', dpi=300)
        if close_figure_after_saving:
                plt.close('all')




# learning curves for CV run and normal run for XGBoost
if create_plots:
    
    fig_xgb_2j_2f_cv, ax_xgb_2j_2f_cv = plt.subplots(figsize=(12, 6))
    plot_cv_res(cvres_xgb_2j_2f, ax_xgb_2j_2f_cv, method='xgb', n_sigma=n_sigma)
    ax_xgb_2j_2f_cv.set(title='XGB: 2-jet, 2 flavours, CV run')
    if save_plots:
        fig_xgb_2j_2f_cv.savefig('./figures/XGB_2jet_2flavor_CV_run.pdf', dpi=300)
    
    
    xgb_eval_2j_2f = clf_xgb_2j_2f.evals_result()
    eval_steps_xgb_2j_2f = range(len(xgb_eval_2j_2f['validation_0']['auc']))

    fig_xgb_2j_2f_normal, ax_xgb_2j_2f_normal = plt.subplots(1, 1, sharex=True, 
                                                     figsize=(8, 6))
    
    ax_xgb_2j_2f_normal.plot(eval_steps_xgb_2j_2f, 
                             xgb_eval_2j_2f['validation_0']['auc'], 
                             color = color_dict['blue'],
                             label = 'Train')
    ax_xgb_2j_2f_normal.plot(eval_steps_xgb_2j_2f, 
                             xgb_eval_2j_2f['validation_1']['auc'], 
                             color = color_dict['red'],
                             label='Test')
    
    ax_xgb_2j_2f_normal.legend()
    ax_xgb_2j_2f_normal.set(xlabel='Number of iterations', ylabel='AUC', 
           title='XGB: 2-jet, 2 flavours, normal run')

    if save_plots:
        fig_xgb_2j_2f_normal.savefig('./figures/XGB_2jet_2flavor_normal_run.pdf', 
                                     dpi=300)

        if close_figure_after_saving:
            plt.close('all')


#%% ===========================================================================
#  LightGBM, 2-jet 2-flavour events 
# =============================================================================

print("\n")

# define training matrix
dftrainLGB_2j_2f = lightgbm.Dataset(data = dfc_X_MC_2j['train'].values, 
                                    label = dfc_y_MC_2j_2f['train'].values, 
                         feature_name = dfc_X_MC_2j['train'].columns.tolist())



# run k-fold CV with LGB
cvres_lgb_2j_2f = pd.DataFrame(lightgbm.cv(
                    lgb_params,
                    dftrainLGB_2j_2f,
                    num_boost_round = num_boost_round,
                    nfold = n_fold,
                    stratified = True,
                    shuffle = False,
                    metrics = 'auc',
                    early_stopping_rounds = early_stopping_rounds,
                    seed = 42,
                    callbacks = [lgb_cv_early_stopping],
                    ))

# get best result:
N_est_cv_best_lgb_2j_2f = cvres_lgb_2j_2f['auc-mean'].values.argmax()
AUC_best_mean_lgb_2j_2f, AUC_best_std_lgb_2j_2f = cvres_lgb_2j_2f.loc[
                                    cvres_lgb_2j_2f.index[N_est_cv_best_lgb_2j_2f], 
                                    ['auc-mean', 'auc-stdv']]

print(f"Best number of estimators based on cross validation: {N_est_cv_best_lgb_2j_2f}")
str_lgb_2j_2f = (f"ROC for LGB-tag, {n_fold}-fold CV set \t\t " + 
           f"{AUC_best_mean_lgb_2j_2f:.5f} +/- {AUC_best_std_lgb_2j_2f:.5f}")
print(str_lgb_2j_2f)


# fit normal lgb classfier using the best number of estimators from CV
clf_lgb_2j_2f = copy(clf_org_lgb)
clf_lgb_2j_2f.set_params(n_estimators=N_est_cv_best_lgb_2j_2f)

clf_lgb_2j_2f.fit(dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'], 
                  eval_metric=['logloss', 'auc'], 
                  verbose = False,
                  eval_set = [((dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])), 
                            (dfc_X_MC_2j['test'], dfc_y_MC_2j_2f['test'])],
                  eval_names = ['train', 'test'],
                )

# predict b-tags and scores
y_pred_lgb_2j_2f = clf_lgb_2j_2f.predict(dfc_X_MC_2j['test'])
y_scores_lgb_2j_2f = clf_lgb_2j_2f.predict_proba(dfc_X_MC_2j['test'])[:, 1]

print("ROC for LGB-tag, test set \t\t", roc_auc_score(dfc_y_MC_2j_2f['test'],
                                                      y_scores_lgb_2j_2f))



# learning curves for CV run and normal run for XGBoost
if create_plots:

    fig_lgb_2j_2f_cv, ax_lgb_2j_2f_cv = plt.subplots(figsize=(12, 6))
    plot_cv_res(cvres_lgb_2j_2f, ax_lgb_2j_2f_cv, method='lgb', n_sigma = n_sigma)
    ax_lgb_2j_2f_cv.set(title='LGB: 2-jet, 2 flavours, CV run')

    if save_plots:
        fig_lgb_2j_2f_cv.savefig('./figures/LGB_2jet_2flavor_CV_run.pdf', dpi=300)
        if close_figure_after_saving:
            plt.close('all')

    lgb_eval_2j_2f = clf_lgb_2j_2f.evals_result_
    eval_steps_lgb_2j_2f = range(len(lgb_eval_2j_2f['train']['auc']))

    fig_lgb_2j_2f_normal, ax_lgb_2j_2f_normal = plt.subplots(1, 1, sharex=True, 
                                                             figsize=(8, 6))
    ax_lgb_2j_2f_normal.plot(eval_steps_lgb_2j_2f, 
                             lgb_eval_2j_2f['train']['auc'], 
                             label='Train', 
                             color = color_dict['blue'])
    
    ax_lgb_2j_2f_normal.plot(eval_steps_lgb_2j_2f, 
                             lgb_eval_2j_2f['test']['auc'], 
                             label='Test', 
                             color = color_dict['red'])
    
    ax_lgb_2j_2f_normal.legend()
    ax_lgb_2j_2f_normal.set(xlabel='Number of iterations', ylabel='AUC', 
           title='LGB: 2-jet, 2 flavours, normal run')


    if save_plots:
        fig_lgb_2j_2f_normal.savefig('./figures/LGB_2jet_2flavor_normal_run.pdf', 
                                     dpi=300)
        if close_figure_after_saving:
            plt.close('all')
        
        

#%% ===========================================================================
#  Test on 3-flavour events  
# =============================================================================


print("\n")

dfc_y_MC_3j_2f['test']

y_pred_xgb_3j_2f = clf_xgb_2j_2f.predict(dfc_X_MC_3j['test'])
y_scores_xgb_3j_2f = clf_xgb_2j_2f.predict_proba(dfc_X_MC_3j['test'])[:, 1]
print("ROC for XGB-tag, 3-jet test set \t", 
      roc_auc_score(dfc_y_MC_3j_2f['test'], y_scores_xgb_3j_2f))


y_pred_lgb_3j_2f = clf_lgb_2j_2f.predict(dfc_X_MC_3j['test'])
y_scores_lgb_3j_2f = clf_lgb_2j_2f.predict_proba(dfc_X_MC_3j['test'])[:, 1]
print("ROC for LGB-tag, 3-jet test set \t", 
      roc_auc_score(dfc_y_MC_3j_2f['test'], y_scores_lgb_3j_2f))





