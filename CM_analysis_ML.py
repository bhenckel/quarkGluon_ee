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
import pickle
import shap


from CM_extra_funcs import (
                                              color_dict,
        
                                              load_data_file,
                                              load_model,
                                              save_model,
                                              
                                              # train_test_split_non_random,
                                              train_test_index,
                                              
                                              check_event_numbers_are_matching,
                                              check_qmatch_are_correct,
                                              
                                              CV_EarlyStoppingTrigger,
                                              plot_cv_res,
                                              plot_cv_test_results,
                                              plot_random_search,
                                              
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

down_sample = 1
verbose = False
cpu_n_jobs = 7
create_plots = False
save_plots = False
create_pairgrid_plot = False
close_figure_after_saving = False

load_models = True


num_boost_round = 10000
early_stopping_rounds = 1000

n_fold = 5
n_sigma = 1
n_hyper = 100


branches_to_import = None 
filename = './data/outputfile_Ref12.root'
treename_data = 'ntuple_DataBtag'
treename_MC = 'ntuple_McBtag'

# use these columns as variables
MC_cols = ['bqvjet', 'muljet', 'projet', 'pt2jet', 'ptljet', 'sphjet', 'phijet']


plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.figsize': (14,8)})

sns.set()

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
    print(f'\n\nDownsampling to {down_sample*100:.2f}%\n\n')
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

verbose_eval = early_stopping_rounds // 2 if verbose else False
metrics_xgb = ['auc', 'error', 'logloss']
metrics_lgb = ['auc', 'binary_error', 'binary_logloss']


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



from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold
import scipy.stats as sp_stats
# from scipy.stats import norm as sp_normal


# A parameter grid for XGBoost
params_random_search = {
        'min_child_weight': sp_stats.randint(1, 10),
        'gamma': sp_stats.expon(0, 1),
        'subsample': sp_stats.uniform(0.5, 1-0.5),
        'colsample_bytree': sp_stats.uniform(0.5, 1-0.5),
        'max_depth': sp_stats.randint(1, 10)
        }

cols_to_use = ['rank_test_score', 'params', 'mean_test_score','std_test_score', 
                   'mean_train_score', 'std_train_score']



#%% ===========================================================================
#  XGboost, 2-jet 2-flavour events  - k-fold Cross Validation; # of trees
# =============================================================================


print("\n")

# define training matrix and verbosity level 
xgb_trainDmatrix_2j_2f = xgboost.DMatrix(dfc_X_MC_2j['train'], 
                                         label=dfc_y_MC_2j_2f['train'])


if load_models:

    print('Loading XGB CV res')
    
    #load saved model
    cvres_xgb_2j_2f = load_model('cvres_xgb_2j_2f_10_percent')

else:
    
    # run k-fold CV with XGB
    cvres_xgb_2j_2f = xgboost.cv(  xgb_params, 
                       xgb_trainDmatrix_2j_2f, 
                       num_boost_round = num_boost_round, 
                       early_stopping_rounds = early_stopping_rounds,
                       
                       nfold = n_fold, 
                       metrics = metrics_xgb,
                       stratified = True, 
                       seed = 42, 
                       shuffle = True,
                       verbose_eval = verbose_eval,
                       callbacks = [xgb_cv_early_stopping],
                       )
    #save model
    save_model(cvres_xgb_2j_2f, 'cvres_xgb_2j_2f_10_percent')


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


#%% XGBoost Random Search



if load_models:

    #load saved model
    rs_xgb_2j_2f = load_model('rs_xgb_2j_2f_10_percent')
    rs_xgb_2j_2f_res = load_model('rs_xgb_2j_2f_res_10_percent')
    clf_xgb_2j_2f = load_model('clf_xgb_2j_2f_10_percent_tmp_cv')
    


else:
    
    # run randomized grid search
    rs_xgb_2j_2f = RandomizedSearchCV(clf_xgb_2j_2f, 
                                       param_distributions = params_random_search, 
                                       n_iter = n_hyper, 
                                       scoring = 'roc_auc', 
                                       n_jobs = cpu_n_jobs, 
                                       cv = n_fold,
                                       verbose = 2, 
                                       random_state = 42,
                                       return_train_score = True,
                                       refit = True)
    
    rs_xgb_2j_2f.fit(dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])
    
    rs_xgb_2j_2f_res = pd.DataFrame(rs_xgb_2j_2f.cv_results_)
    rs_xgb_2j_2f_res = rs_xgb_2j_2f_res.loc[:, cols_to_use].sort_values(by='rank_test_score')

    clf_xgb_2j_2f = rs_xgb_2j_2f.best_estimator_

    #save model
    save_model(rs_xgb_2j_2f, 'rs_xgb_2j_2f_10_percent')
    save_model(rs_xgb_2j_2f_res, 'rs_xgb_2j_2f_res_10_percent')
    save_model(clf_xgb_2j_2f, 'clf_xgb_2j_2f_10_percent_tmp_cv')


print("\n")
print(rs_xgb_2j_2f.best_params_)
print("\n")
print(rs_xgb_2j_2f_res.loc[:, ['mean_test_score', 'std_test_score']].iloc[0, :])
print("\n")


# XGB: Results of Random Search
plot_random_search(rs_xgb_2j_2f_res, n_fold, 'XGB 2j 2f, 10% data')



#%% XGB: Fitting final model to data


if load_models:

    #load saved model
    clf_xgb_2j_2f = load_model('clf_xgb_2j_2f_10_percent')

else:
    # fit to data
    clf_xgb_2j_2f.fit(dfc_X_MC_2j['train'], 
                  dfc_y_MC_2j_2f['train'], 
                  eval_metric = metrics_xgb, 
                  verbose = False,
                  eval_set = [((dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])), 
                               (dfc_X_MC_2j['test'], dfc_y_MC_2j_2f['test'])])
    
    #save model
    save_model(clf_xgb_2j_2f, 'clf_xgb_2j_2f_10_percent')
    
    
# predict b-tags and scores
y_pred_xgb_2j_2f_test = clf_xgb_2j_2f.predict(dfc_X_MC_2j['test'])
y_scores_xgb_2j_2f_test = clf_xgb_2j_2f.predict_proba(dfc_X_MC_2j['test'])[:, 1]
y_scores_xgb_2j_2f_train = clf_xgb_2j_2f.predict_proba(dfc_X_MC_2j['train'])[:, 1]

y_scores_xgb_2j_2f = y_scores_xgb_2j_2f_test

print("ROC for XGB-tag, test set \t\t", roc_auc_score(dfc_y_MC_2j_2f['test'], y_scores_xgb_2j_2f_test))


y_scores_xgb_2j_2f_test = pd.Series(y_scores_xgb_2j_2f_test, index=dfc_X_MC_2j['test'].index)
y_scores_xgb_2j_2f_train = pd.Series(y_scores_xgb_2j_2f_train, index=dfc_X_MC_2j['train'].index)
y_scores_xgb_2j_2f_all = y_scores_xgb_2j_2f_train.append(y_scores_xgb_2j_2f_test)

dfc_y_scores_xgb_2j = PandasContainer(y_scores_xgb_2j_2f_all, 
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
    plot_cv_res(cvres_xgb_2j_2f, ax_xgb_2j_2f_cv, metrics_xgb, method='xgb', n_sigma=n_sigma)
    ax_xgb_2j_2f_cv.set(title='XGB: 2-jet, 2 flavours, CV run')
    if save_plots:
        fig_xgb_2j_2f_cv.savefig('./figures/XGB_2jet_2flavor_CV_run.pdf', dpi=300)
    
    
    
    xgb_eval_2j_2f = clf_xgb_2j_2f.evals_result()
    fig_xgb_2j_2f_normal, ax_xgb_2j_2f_normal = plot_cv_test_results(
                                                xgb_eval_2j_2f, 
                                                method = 'xgb',
                                                metrics = metrics_xgb,
                                                title = 'XGB: 2-jet, 2 flavours, normal run')
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



if load_models:

    #load saved model
    cvres_lgb_2j_2f = load_model('cvres_lgb_2j_2f_10_percent')


else:
    
    # run k-fold CV with LGB
    cvres_lgb_2j_2f = pd.DataFrame(lightgbm.cv(
                    lgb_params,
                    dftrainLGB_2j_2f,
                    
                    num_boost_round = num_boost_round,
                    early_stopping_rounds = early_stopping_rounds,
                    
                    nfold = n_fold,
                    stratified = True,
                    shuffle = True,
                    metrics = metrics_lgb,
                    
                    seed = 42,
                    verbose_eval = verbose_eval,
                    callbacks = [lgb_cv_early_stopping],
                    ))

    #save model
    save_model(cvres_lgb_2j_2f, 'cvres_lgb_2j_2f_10_percent')



# get best result:
N_est_cv_best_lgb_2j_2f = cvres_lgb_2j_2f['auc-mean'].values.argmax()
AUC_best_mean_lgb_2j_2f, AUC_best_std_lgb_2j_2f = cvres_lgb_2j_2f.loc[
                                    cvres_lgb_2j_2f.index[N_est_cv_best_lgb_2j_2f], 
                                    ['auc-mean', 'auc-stdv']]

print(f"Best number of estimators based on cross validation: {N_est_cv_best_lgb_2j_2f}")
str_lgb_2j_2f = (f"ROC for LGB-tag, {n_fold}-fold CV set \t\t " + 
           f"{AUC_best_mean_lgb_2j_2f:.5f} +/- {AUC_best_std_lgb_2j_2f:.5f}\n")
print(str_lgb_2j_2f)


# fit normal lgb classfier using the best number of estimators from CV
clf_lgb_2j_2f = copy(clf_org_lgb)
clf_lgb_2j_2f.set_params(n_estimators=N_est_cv_best_lgb_2j_2f)








#%% LGB: Random Search

if load_models:
    
    #load saved model
    
    rs_lgb_2j_2f = load_model('rs_lgb_2j_2f_10_percent')
    rs_lgb_2j_2f_res = load_model('rs_lgb_2j_2f_res_10_percent')
    clf_lgb_2j_2f = load_model('clf_lgb_2j_2f_10_percent_tmp_cv')

else:
    
    # run randomized grid search
    rs_lgb_2j_2f = RandomizedSearchCV(clf_lgb_2j_2f, 
                                   param_distributions = params_random_search, 
                                   n_iter = n_hyper, 
                                   scoring = 'roc_auc', 
                                   n_jobs = cpu_n_jobs, 
                                   cv = n_fold,
                                   verbose = 2, 
                                   random_state = 42,
                                   return_train_score = True,
                                   refit = True)

    rs_lgb_2j_2f.fit(dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])
    
    rs_lgb_2j_2f_res = pd.DataFrame(rs_lgb_2j_2f.cv_results_)
    rs_lgb_2j_2f_res = rs_lgb_2j_2f_res.loc[:, cols_to_use].sort_values(by='rank_test_score')
    
    clf_lgb_2j_2f = rs_lgb_2j_2f.best_estimator_

    #save model
    save_model(rs_lgb_2j_2f_res, 'rs_lgb_2j_2f_res_10_percent')
    save_model(rs_lgb_2j_2f, 'rs_lgb_2j_2f_10_percent')
    save_model(clf_lgb_2j_2f, 'clf_lgb_2j_2f_10_percent_tmp_cv')


print("\n")
print(rs_lgb_2j_2f.best_params_)
print("\n")
print(rs_lgb_2j_2f_res.loc[:, ['mean_test_score', 'std_test_score']].iloc[0, :])
print("\n\n")


# LGB: Results of Random Search
plot_random_search(rs_lgb_2j_2f_res, n_fold, 'LGB 2j 2f, 10% data')


#%% LGB fitting final model

if load_models:
    #load saved model
    clf_lgb_2j_2f = load_model('clf_lgb_2j_2f_10_percent')


else:
    clf_lgb_2j_2f.fit(dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'], 
                  eval_metric=metrics_lgb, 
                  verbose = False,
                  eval_set = [((dfc_X_MC_2j['train'], dfc_y_MC_2j_2f['train'])), 
                            (dfc_X_MC_2j['test'], dfc_y_MC_2j_2f['test'])],
                  eval_names = ['train', 'test'],
                      )
    #save model
    save_model(clf_lgb_2j_2f, 'clf_lgb_2j_2f_10_percent')


# predict b-tags and scores
y_pred_lgb_2j_2f = clf_lgb_2j_2f.predict(dfc_X_MC_2j['test'])
y_scores_lgb_2j_2f = clf_lgb_2j_2f.predict_proba(dfc_X_MC_2j['test'])[:, 1]

print("ROC for LGB-tag, test set \t\t", roc_auc_score(dfc_y_MC_2j_2f['test'],
                                                      y_scores_lgb_2j_2f))
print("\n")


#%%

# learning curves for CV run and normal run for LightGBM
if create_plots:

    fig_lgb_2j_2f_cv, ax_lgb_2j_2f_cv = plt.subplots(figsize=(12, 6))
    plot_cv_res(cvres_lgb_2j_2f, ax_lgb_2j_2f_cv, metrics_lgb, method='lgb', n_sigma = n_sigma)
    ax_lgb_2j_2f_cv.set(title='LGB: 2-jet, 2 flavours, CV run')


    if save_plots:
        fig_lgb_2j_2f_cv.savefig('./figures/LGB_2jet_2flavor_CV_run.pdf', dpi=300)
        if close_figure_after_saving:
            plt.close('all')



    lgb_eval_2j_2f = clf_lgb_2j_2f.evals_result_
    fig_lgb_2j_2f_normal, ax_lgb_2j_2f_normal = plot_cv_test_results(
                                                lgb_eval_2j_2f, 
                                                method='lgb',
                                                metrics = metrics_lgb,
                                                title = 'LGB: 2-jet, 2 flavours, normal run')

    if save_plots:
        fig_lgb_2j_2f_normal.savefig('./figures/LGB_2jet_2flavor_normal_run.pdf', 
                                     dpi=300)
        if close_figure_after_saving:
            plt.close('all')



#%%
            
from scipy.stats import rankdata, pearsonr, spearmanr

fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(y_scores_xgb_2j_2f, y_scores_lgb_2j_2f, s=4)
ax.set(xlabel='XGB', ylabel='LGB', title='Proba')

corr_pearson = pearsonr(y_scores_xgb_2j_2f, y_scores_lgb_2j_2f)
corr_spearman = spearmanr(y_scores_xgb_2j_2f, y_scores_lgb_2j_2f)

print("corr_pearson", corr_pearson)
print("corr_spearman", corr_spearman)


# ranking: small values get low rank
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(rankdata(y_scores_xgb_2j_2f), rankdata(y_scores_lgb_2j_2f), s=4)
ax.set(xlabel='XGB', ylabel='LGB', title='Rank of proba')




#%% SHAP Values

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer_xgb_2j_2f = shap.TreeExplainer(clf_xgb_2j_2f)
explainer_lgb_2j_2f = shap.TreeExplainer(clf_lgb_2j_2f)

xxx=xxx

# load_models = False
# print("REMEMBER THIS TODO!!!!, load_models = False") # TODO


if load_models:
    shap_values_xgb_2j_2f = load_model('shap_values_xgb_2j_2f_100_percent')
    shap_values_lgb_2j_2f = load_model('shap_values_lgb_2j_2f_100_percent')

else:
    shap_values_xgb_2j_2f = explainer_xgb_2j_2f.shap_values(dfc_X_MC_2j['test'])
    shap_values_lgb_2j_2f = explainer_lgb_2j_2f.shap_values(dfc_X_MC_2j['test'])

    save_model(shap_values_xgb_2j_2f, 'shap_values_xgb_2j_2f_10_percent')
    save_model(shap_values_lgb_2j_2f, 'shap_values_lgb_2j_2f_10_percent')


for shap_values, name in zip([shap_values_xgb_2j_2f, shap_values_lgb_2j_2f], 
                             ['XGB', 'LGB']):

    
    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    plt.figure()
    shap.dependence_plot("projet", shap_values, dfc_X_MC_2j['test'])
    plt.title(name)
    
    # dependence plot with a specific interaction index: here itself.
    plt.figure()
    shap.dependence_plot("projet", shap_values, dfc_X_MC_2j['test'], 
                         show=True, interaction_index="projet")
    plt.title(name)
    
    # summarize the effects of all the features
    plt.figure()
    shap.summary_plot(shap_values, dfc_X_MC_2j['test'])
    plt.title(name)
    
    # one-dimensional summary of all feature importances
    plt.figure()
    shap.summary_plot(shap_values, dfc_X_MC_2j['test'], plot_type="bar")
    plt.title(name)
    
    # the numerical calculation of above plot is:
    shap_values_df = pd.DataFrame(shap_values, columns = dfc_X_MC_2j['test'].columns)
    shap_feature_importance = shap_values_df.abs().mean(0)
    




#%% Early stopping test



#%%
    
    
def ES_rules_early_stopping(cvres, metric='auc'):
    """ The classic early stopping rule """
    
    return cvres[f'{metric}-mean'].values.argmax()


def ES_rules_early_stopping_frac(cvres, frac, metric='auc'):
    """ Instead of stopping at N_ES (early stopping) stop at int(N_ES*frac) 
        i.e. at e.g. 90% of the early stopping iteration
    """
    
    # if 0<frac<1:
    #     pass
    # elif 1<frac<100:
    #     frac /= 100
    # else:
        # raise Exception('wrong value of "frac"')
    
    return int(cvres[f'{metric}-mean'].values.argmax() * frac)


def ES_rules_early_stopping_frac_best_before(cvres, frac, metric='auc'):
    """ Find the early stopping iteration that was the best before 
        N_ES*frac iterations 
    """
    N_ES_frac = ES_rules_early_stopping_frac(cvres, frac, metric=metric)
    return ES_rules_early_stopping(cvres.iloc[:N_ES_frac], metric=metric)


def ES_rules_n_sigma(cvres, n, metric='auc'):
    """ Go back to when mean+n*sigma = max(mean), i.e. when the mean plus 
        n sigmas is similar to the maximum of the means.    
    """
    
    mean = cvres[f'{metric}-mean'].values
    std = cvres[f'{metric}-stdv'].values
    
    return np.argmax(mean + n*std >= mean.max())


def ES_rules_combined(cvres, metric='auc'):
    
    d = {'normal': ES_rules_early_stopping(cvres, metric=metric),
         
         'frac_115': ES_rules_early_stopping_frac(cvres, 1.15, metric=metric),
         'frac_110': ES_rules_early_stopping_frac(cvres, 1.10, metric=metric),
         'frac_105': ES_rules_early_stopping_frac(cvres, 1.05, metric=metric),
         
          'frac_95': ES_rules_early_stopping_frac(cvres, 0.95, metric=metric),
          'frac_90': ES_rules_early_stopping_frac(cvres, 0.90, metric=metric),
          'frac_85': ES_rules_early_stopping_frac(cvres, 0.85, metric=metric),
          'frac_80': ES_rules_early_stopping_frac(cvres, 0.80, metric=metric),
         
         'frac_95_best': ES_rules_early_stopping_frac_best_before(cvres, 0.95, metric=metric),
         'frac_90_best': ES_rules_early_stopping_frac_best_before(cvres, 0.90, metric=metric),
         'frac_85_best': ES_rules_early_stopping_frac_best_before(cvres, 0.85, metric=metric),
         'frac_80_best': ES_rules_early_stopping_frac_best_before(cvres, 0.80, metric=metric),
         
         '0.05_sigma': ES_rules_n_sigma(cvres, 0.05, metric=metric),
         '0.10_sigma': ES_rules_n_sigma(cvres, 0.10, metric=metric),
         '0.25_sigma': ES_rules_n_sigma(cvres, 0.25, metric=metric),
         '0.50_sigma': ES_rules_n_sigma(cvres, 0.50, metric=metric),
         '0.75_sigma': ES_rules_n_sigma(cvres, 0.75, metric=metric),
         '1.00_sigma': ES_rules_n_sigma(cvres, 1.00, metric=metric),
         
         }
    
    return d





  #%%

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


clf_org_lgb = LGBMClassifier(  n_estimators = 1000,
                          learning_rate = 0.01,
                          objective = 'binary',
                          n_jobs = cpu_n_jobs,
                          random_state = 42,
                          silent=True
                          )

lgb_params = clf_org_lgb.get_params()
lgb_params.pop('n_estimators')
lgb_params.pop('silent')


num_boost_round = 10_000
early_stopping_rounds = 100
n_fold = 10


fraction = 2 / 100
n_splits = int(1/fraction)

X = dfc_X_MC_2j['train']
y = dfc_y_MC_2j_2f['train']



skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
index_splitted = []
for train_index, test_index in skf.split(X, y):
    index_splitted.append(test_index)


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,10))
ax_mean, ax_sdom, ax_z, ax_diff, ax_diff_normed, ax_z_diff, ax_bla1, ax_bla2 = axes.flatten()

index_splitted_iter = index_splitted
# index_splitted_iter = index_splitted[:10]


print("X.shape: ", X.shape)
print("len(index_splitted_iter[0])", len(index_splitted_iter[0]))

all_cv_res = []
all_N_ES = []
all_AUC_df = []


xxx=xxx




for i, index_train in tqdm(enumerate(index_splitted_iter)):
    
    
    early_stopping_lgb = CV_EarlyStoppingTrigger(
                                    stopping_rounds = early_stopping_rounds, 
                                    maximize_score = True,
                                    method = 'lgb')  
    
    
    
    dataset_lgb = lightgbm.Dataset(data = X.iloc[index_train].values, 
                                    label = y.iloc[index_train].values, 
                                    feature_name = X.columns.tolist())


    cvres_lgb = pd.DataFrame(lightgbm.cv(
                    lgb_params,
                    dataset_lgb,
                    num_boost_round = num_boost_round,
                    early_stopping_rounds = early_stopping_rounds,
                    
                    nfold = n_fold,
                    stratified = True,
                    shuffle = True,
                    # metrics = metrics_lgb,
                    metrics = 'auc',
                    
                    seed = 42,
                    verbose_eval = verbose_eval,
                    callbacks = [early_stopping_lgb],
                    ))


    all_cv_res.append(cvres_lgb)


    N_ES = ES_rules_combined(cvres_lgb)
    all_N_ES.append(N_ES)


    clf_lgb = copy(clf_org_lgb)
    clf_lgb.set_params(n_estimators=N_ES['normal'])
            
    clf_lgb.fit(X.iloc[index_train], y.iloc[index_train], 
                eval_metric=metrics_lgb, 
                verbose = False)


    if True:
        
        # initialize the AUC values: dict of empty lists
        AUCs = {}
        for key, val in N_ES.items():
            AUCs[key] = []
    
    
        # test the trained model on all the other splits
        for j, index_test in enumerate(index_splitted_iter):
            
            # print(f'j = {j}')
            # avoid testing on training data
            if i != j:
                
                # get AUC for each stopping rule option
                for key, val in N_ES.items():
                    
                    y_scores = clf_lgb.predict_proba(X.iloc[index_test], 
                                                     num_iteration=val)[:, 1]
                    auc = roc_auc_score(y.iloc[index_test], y_scores)
                    AUCs[key].append(auc)
    
        # make the AUC values into a dict of numpy arrays
        for key, val in AUCs.items():
            AUCs[key] = np.array(val)



    AUC = pd.DataFrame(AUCs)
    
    all_AUC_df.append(AUC)
    

    index_lgb = cvres_lgb.index
    mean_lgb = cvres_lgb['auc-mean']
    std_lgb = cvres_lgb['auc-stdv'] / np.sqrt(n_fold)
    best_lgb = np.argmax(mean_lgb.values)
    z_lgb = mean_lgb / std_lgb
    diff_lgb = mean_lgb.diff()
    diff_normed_lgb = mean_lgb.diff()/std_lgb

    ax_mean.plot(index_lgb, mean_lgb, color='r')
    ax_mean.fill_between(index_lgb, mean_lgb+std_lgb, mean_lgb-std_lgb,
                            color='r', interpolate=True, alpha=0.1)
    ax_mean.plot(best_lgb, mean_lgb.iloc[best_lgb], 'ok')
    ax_mean.set(ylabel='mean', xlabel='iteration #', title='AUC: mean')
    
    ax_sdom.plot(index_lgb, std_lgb, label=str(i))
    ax_sdom.plot(best_lgb, std_lgb.iloc[best_lgb], 'ok')
    ax_sdom.set(ylabel='sdom', xlabel='iteration #', title='AUC: sdom')
    
    ax_z.plot(index_lgb, z_lgb, label=str(i))
    ax_z.plot(best_lgb, z_lgb.iloc[best_lgb], 'ok')
    ax_z.set(ylabel='z', xlabel='iteration #', title='AUC: mean / sdom = z')
    
    ax_diff.plot(index_lgb[1:], diff_lgb.iloc[1:], label=str(i))
    ax_diff.plot(best_lgb, diff_lgb.iloc[best_lgb], 'ok')
    ax_diff.set(ylabel='diff', xlabel='iteration #', title='AUC: diff(mean) = diff')
    
    ax_diff_normed.plot(index_lgb[1:], diff_normed_lgb.iloc[1:], label=str(i))
    ax_diff_normed.plot(best_lgb, diff_normed_lgb.iloc[best_lgb], 'ok')
    ax_diff_normed.set(ylabel='diff_normed)', xlabel='iteration #', 
                   title='AUC: diff(mean) / sdom = diff_normed')
    
    ax_z_diff.plot(index_lgb[1:], z_lgb.diff().iloc[1:], label=str(i))
    ax_z_diff.plot(best_lgb+1, z_lgb.diff().iloc[best_lgb], 'ok')
    ax_z_diff.set(ylabel='diff(z)', xlabel='iteration #', 
                   title='AUC: diff(z)')
    
fig.tight_layout()  




save_model(all_cv_res, 'all_cv_res_100_percent')
save_model(all_N_ES, 'all_N_ES_100_percent')
save_model(all_AUC_df, 'all_AUC_df_100_percent')


if load_models:
    all_cv_res = load_model('all_cv_res_100_percent-tmp')
    all_N_ES = load_model('all_N_ES_100_percent_tmp')
    all_AUC_df = load_model('all_AUC_df_100_percent_tmp')

else:
    save_model(all_cv_res, 'all_cv_res_100_percent')
    save_model(all_N_ES, 'all_N_ES_100_percent')
    save_model(all_AUC_df, 'all_AUC_df_100_percent')



df_AUC_agg = pd.concat(all_AUC_df, ignore_index=True)
df_N_ES_agg = pd.DataFrame(all_N_ES) 
df_N_ES_agg = df_N_ES_agg.reindex(df_AUC_agg.columns, axis=1)


ax_AUC_ES = df_AUC_agg.plot.kde(bw_method='scott')
ax_AUC_ES.set(xlabel='AUC', title='LGB: KDE plot of AUC')

ax_N_ES = df_N_ES_agg.plot.kde(bw_method='scott')
ax_N_ES.set(xlabel='# of trees', title='LGB: KDE plot of # of trees')


            
def return_mean_and_sdom(df, axis=0):
    means = df.mean(axis=axis)
    stds = df.std(axis=axis)
    sdoms = stds / np.sqrt(df.shape[axis])
    return means, sdoms

N_ES_means, N_ES_sdoms = return_mean_and_sdom(df_N_ES_agg, axis=0)
AUC_means, AUC_sdoms = return_mean_and_sdom(df_AUC_agg, axis=0)



fig_N_AUC, ax_N_AUC = plt.subplots()

import itertools
iterator = zip(N_ES_means, N_ES_sdoms, 
               AUC_means, AUC_sdoms, 
               N_ES_means.index, itertools.cycle(color_dict.values()))

for (n_mean, n_sdom, auc_mean, auc_sdom, name, color) in iterator:
    
    ax_N_AUC.errorbar(x=n_mean, xerr=n_sdom, y=auc_mean, yerr=auc_sdom,
                fmt='.', 
                color=color, ecolor=color,
                capsize=0,
                label=name,
                )
    
    ax_N_AUC.text(x=n_mean*1.02, y=auc_mean*1.0001, s=name, color=color)



ax_N_AUC2 = ax_N_AUC.twinx()  # instantiate a second axes that shares the same x-axis


y_ticks = ax_N_AUC.get_yticks()
y_ticks2 = 1 - 1/y_ticks

ax_N_AUC2.set_ylabel('Relative to 1')  # we already handled the x-label with ax1
# ax_N_AUC2.tick_params(axis='y', )

ax_N_AUC2.set_yticks(y_ticks, minor=False)
ax_N_AUC2.set_yticklabels([f'{i:.2%}' for i in y_ticks2])

ax_N_AUC.set(xlabel='# of trees (iterations)', ylabel='AUC')
ax_N_AUC.legend()









#%%



index_splitted = np.split(df_AUC_agg.index, n_splits)

counts_best = {}
for col_name in df_AUC_agg.columns:
    counts_best[col_name] = 0
    
for index in index_splitted:
    col_name = df_AUC_agg.iloc[index, :].max(axis=0).idxmax()
    counts_best[col_name] += 1

df_counts_best = pd.Series(counts_best)

plt.figure()
df_counts_best.plot.bar()


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





