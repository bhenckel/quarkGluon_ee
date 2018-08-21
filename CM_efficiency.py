#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:13:06 2018

@author: michelsen
"""

# to ignore deprecation warnings:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


import numpy as np
import matplotlib.pyplot as plt
import uproot
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.metrics import roc_auc_score
from copy import copy
from iminuit import Minuit
from collections import OrderedDict



from CM_extra_funcs import (
                              color_dict,

                              load_data_file,
                              nnbjet_to_matrix,
                              
                              check_event_numbers_are_matching,
                              check_qmatch_are_correct,
                              plot_histogram_lin_log,
                              
                              degrees_of_freedom,
                              chi2_asym_no_fit,
                              print_chi2_asym_no_fit,
                              
                              initial_values_from_MC,

                              calc_f_eff_per_cos_theta_bin,
                              plot_cos_theta_analysis,
                              )





#%% ===========================================================================
#  Initial parameters
# =============================================================================

verbose = True

saveplots = False
branches_to_import = None 
filename = './data/outputfile_Ref12.root'
treename_data = 'ntuple_DataBtag'
treename_MC = 'ntuple_McBtag'


np.random.seed(42)

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
df_MC = df_MC_all[df_MC_all.datatype == 1].copy() # normal MC 
df_MCb = df_MC_all[df_MC_all.datatype == 2].copy() # MC b, i.e. extra b-events


# get 2-jet events and 3-jet
df_MC_2j = df_MC[df_MC.njet == 2]
df_MC_3j = df_MC[df_MC.njet == 3]



df_data_2j = df_data[df_data.njet == 2]
df_data_3j = df_data[df_data.njet == 3]


# check that the two jets share same event number:
check_event_numbers_are_matching(df_MC_2j)
check_event_numbers_are_matching(df_MC_3j)

check_event_numbers_are_matching(df_data_2j)
check_event_numbers_are_matching(df_data_3j)


# #%% cos(theta) histograms

# delta = np.abs(df_MC_2j.costheta) - df_MC_2j.costhr

# mask_qmatch0 = (df_MC_2j.qmatch == 0).values

# fig_hist_cos_theta, ax_hist_cos_theta = plt.subplots(nrows=3, figsize=(8, 10))

# plot_histogram_lin_log(delta, ax_hist_cos_theta[0], Nbins=100, 
#                        title=f'All 2 jet events, N = {len(delta)}',
#                        xlabel=None,
#                        xlim=(-0.8, 0.8),
#                        )

# plot_histogram_lin_log(delta.iloc[mask_qmatch0], ax_hist_cos_theta[1], Nbins=100, 
#                        title=f'2 jet events with qmatch = 0, N = {mask_qmatch0.sum()}',
#                        xlabel=None,
#                        xlim=(-0.8, 0.8),
#                        )

# plot_histogram_lin_log(delta.iloc[~mask_qmatch0], ax_hist_cos_theta[2], Nbins=100, 
#                        title=f'2 jet events with qmatch != 0, N = {(~mask_qmatch0).sum()}',
#                        xlabel=r'$|\cos(\theta)| - \cos($thr$)$',
#                        xlim=(-0.8, 0.8),
#                        )

# fig_hist_cos_theta.tight_layout()



# fig, ax = plt.subplots(figsize=(12, 8))
# ax.hist(delta[mask_qmatch0], 100, range=(-0.8, 0.8), histtype='step', 
#         label='qmatch = 0', log=True)
# ax.hist(delta[~mask_qmatch0], 100, range=(-0.8, 0.8), histtype='step', 
#         label='qmatch != 0', log=True)
# ax.legend()


#%%

# set bin edges
# bin_edges = np.linspace(0, 1, 15+1)
bin_edges = np.array([0.0, 0.15, 0.4, 0.6, 0.85, 0.96, 1.0])
N_bins = len(bin_edges) - 1
N_flavors = 3
assumption = 'sym'

print('\n'*3+'- '*5 + 'Number of bins, observables, variables and DOF: ' + '- '*5+'\n')
N_obs, N_vars, N_dof = degrees_of_freedom(assumption, N_bins, N_flavors, verbose=verbose)
print("\n\n")


#%%




# first extract data counts as they are easy

nnbjet_data_2j = df_data_2j.loc[:, 'nnbjet']
matrix_data_2j = nnbjet_to_matrix(nnbjet_data_2j, bin_edges, 'matrix_data_2j')

nnbjet_data_3j = df_data_3j.loc[:, 'nnbjet']
matrix_data_3j = nnbjet_to_matrix(nnbjet_data_3j, bin_edges, 'matrix_data_3j')



# extract the b-tags for {b, c, l}-quarks

mask_MC_2j_b = df_MC_2j['flevt'] == 5
mask_MC_2j_c = df_MC_2j['flevt'] == 4
mask_MC_2j_l = df_MC_2j['flevt'] < 4


nnbjet_MC_2j_b = df_MC_2j.loc[mask_MC_2j_b, 'nnbjet']
nnbjet_MC_2j_c = df_MC_2j.loc[mask_MC_2j_c, 'nnbjet']
nnbjet_MC_2j_l = df_MC_2j.loc[mask_MC_2j_l , 'nnbjet']
nnbjet_MC_2j_all = df_MC_2j.loc[:, 'nnbjet']

# calculate the count matrix for the different quarks
matrix_MC_2j_b = nnbjet_to_matrix(nnbjet_MC_2j_b, bin_edges, 'matrix_MC_2j_b')
matrix_MC_2j_c = nnbjet_to_matrix(nnbjet_MC_2j_c, bin_edges, 'matrix_MC_2j_c')
matrix_MC_2j_l = nnbjet_to_matrix(nnbjet_MC_2j_l, bin_edges, 'matrix_MC_2j_L')

# make a count-matrix for all of them compbined
matrix_MC_2j_all = copy(matrix_MC_2j_b) + matrix_MC_2j_c + matrix_MC_2j_l
matrix_MC_2j_all.columns.name = 'matrix_MC_2j_all'

# make a count matrix for c and l quarks combined
matrix_MC_2j_cl = copy(matrix_MC_2j_c) + matrix_MC_2j_l
matrix_MC_2j_cl.columns.name = 'matrix_MC_2j_cl'

if verbose:
    print('\n'*3+'- '*5 + 'Count matrix for MC 2j b: ' + '- '*5+'\n')
    print(matrix_MC_2j_b, "\n")

# make a list of the count matrices from the 3 quarks 
matrix_MC_2j = [matrix_MC_2j_b, matrix_MC_2j_c, matrix_MC_2j_l]


# print the chi2-asymmetry, number of degrees of freedom and chi2-probability
print('\n'*3+'- '*3 + 
      'Chi2-test for symmetri assumption for the count matrices: ' + 
      '- '*3+'\n')
print("flavour \t chi2-val,   N_dof,      P ")
print_chi2_asym_no_fit(matrix_MC_2j_b, 'b')
print_chi2_asym_no_fit(matrix_MC_2j_c, 'c')
print_chi2_asym_no_fit(matrix_MC_2j_l, 'l')
print_chi2_asym_no_fit(matrix_MC_2j_cl, 'cl')
print_chi2_asym_no_fit(matrix_MC_2j_all, 'all')



#%% get initial values for the fit

pars = initial_values_from_MC(matrix_MC_2j)
N_total_2j, f_MC_2j, eff_MC_2j, C_MC_2j, S_MC_2j = pars


# pars = initial_values_from_MC(matrix_MC_3j)
# N_total_3j, f_MC_3j, eff_MC_3j, C_MC_3j, S_MC_3j = pars


#%% cos(theta) analysis


cut_off_cos_theta = 0.8
N_bins_cos_theta = 8

bin_edges_cos_theta = np.linspace(0, cut_off_cos_theta, N_bins_cos_theta+1)
N_bins_cos_theta = len(bin_edges_cos_theta)-1

df_cos_theta_analysis = calc_f_eff_per_cos_theta_bin(df_MC_2j, bin_edges_cos_theta, bin_edges,
                                 mask_MC_2j_b, mask_MC_2j_c, mask_MC_2j_l)


df_cos_theta_analysis.plot(
                           # ax=ax, 
                           subplots=True,
                           layout = (4,5),
                           figsize = (12, 8),
                           title='fig1'
                           )

fig_cos_theta, ax_cos_theta = plt.subplots(figsize=(16, 8))
plot_cos_theta_analysis(ax_cos_theta, df_cos_theta_analysis)
fig_cos_theta.tight_layout()
# if saveplots:
    # fig_cos_theta.savefig('bla.pdf', dpi=600)


#%%

S0_2j_sym = np.zeros_like(S_MC_2j)
S0_2j_asym = copy(S_MC_2j)


# if assumption.lower() == 'sym':
#     S0_2j = S0_2j_sym
# else:
#     S0_2j = S0_2j_asym


from CM_extra_funcs import (
                            # from_1D_to_2D, 
                            # from_2D_to_1D, 
                            # calc_y, 
                            # calc_chi2_sym,
                            # make_initial_values_dict,
                            # list_of_pars_to_f_eff,
                            # set_limits_on_params,
                            # minuit_wrapper,
                            # chi2_to_P,
                            
                            fit_object,
                            )


# eff_1D_test = np.arange(1, (N_bins-1)*N_flavors+1)/100 #
# eff_2D_test = from_1D_to_2D(eff_1D_test, N_bins, N_flavors)
# f_test = np.array([0.33, 0.44])
# f0 = f_test
# eff0 = eff_2D_test


f0_2j = f_MC_2j[:-1]
eff0_2j = eff_MC_2j[:-1, :]

# below values from http://pdg.lbl.gov/2018/listings/rpp2018-list-z-boson.pdf

f_bb = 15.6 / 100
s_f_bb = 0.4 / 100
f_cc = 11.6 / 100
s_f_cc = 0.6 / 100
f_hadrons = 69.91 / 100
s_f_hadrons = 0.06 / 100

f0_2j_theoretical = np.array([f_bb , f_cc]) / f_hadrons
sf0_2j_theoretical = np.array([np.sqrt( s_f_bb**2 + f_bb**2/f_hadrons**2 * s_f_hadrons**2 ), 
                               np.sqrt( s_f_cc**2 + f_cc**2/f_hadrons**2 * s_f_hadrons**2 )
                               ]) / f_hadrons




print('\n'*3+'- '*10 + 'Symmetric Fitting, MC ' + '- '*10+'\n'*3)

fit_MC_2j_sym = fit_object(f0_2j, eff0_2j, S0_2j_sym, matrix_MC_2j_all, 
                     verbose, assumption, use_limits=False)
fit_MC_2j_sym.initial_chi2_check()
fit_MC_2j_sym.fit()
pars = fit_MC_2j_sym.get_fit_values()
(f_fit_MC_2j_sym, eff_fit_MC_2j_sym, 
 s_f_fit_MC_2j_sym, s_eff_fit_MC_2j_sym,
 cov_f_fit_MC_2j_sym, cov_eff_fit_MC_2j_sym) = pars


print('\n'*3+'- '*10 + 'Asymmetric Fitting, MC ' + '- '*10+'\n'*3)

fit_MC_2j_asym = fit_object(f0_2j, eff0_2j, S0_2j_asym, matrix_MC_2j_all, 
                     verbose, assumption, use_limits=False)
fit_MC_2j_asym.initial_chi2_check()
fit_MC_2j_asym.fit()
pars = fit_MC_2j_asym.get_fit_values()
(f_fit_MC_2j_asym, eff_fit_MC_2j_asym, 
 s_f_fit_MC_2j_asym, s_eff_fit_MC_2j_asym,
 cov_f_fit_MC_2j_asym, cov_eff_fit_MC_2j_asym) = pars



print('\n'*3+'- '*10 + 'Symmetric Fitting, data ' + '- '*10+'\n'*3)

fit_data_2j_sym = copy(fit_MC_2j_sym)

fit_data_2j_sym.data = matrix_data_2j
fit_data_2j_sym.f0 = f_fit_MC_2j_sym
fit_data_2j_sym.eff0 = eff_fit_MC_2j_sym

fit_data_2j_sym.initial_chi2_check()
# fit_data_2j_sym.f0 = f0_2j_theoretical
# fit_data_2j_sym.initial_chi2_check()


fit_data_2j_sym.fit()
pars = fit_data_2j_sym.get_fit_values()
(f_fit_data_2j_sym, eff_fit_data_2j_sym, 
 s_f_fit_data_2j_sym, s_eff_fit_data_2j_sym,
 cov_f_fit_data_2j_sym, cov_eff_fit_data_2j_sym) = pars




print('\n'*3+'- '*10 + 'Asymmetric Fitting, data ' + '- '*10+'\n'*3)

fit_data_2j_asym = copy(fit_MC_2j_asym)

fit_data_2j_asym.data = matrix_data_2j
fit_data_2j_asym.f0 = f_fit_MC_2j_asym
fit_data_2j_asym.eff0 = eff_fit_MC_2j_asym

fit_data_2j_asym.initial_chi2_check()
# fit_data_2j_sym.f0 = f0_2j_theoretical
# fit_data_2j_sym.initial_chi2_check()


fit_data_2j_asym.fit()
pars = fit_data_2j_asym.get_fit_values()
(f_fit_data_2j_asym, eff_fit_data_2j_asym, 
 s_f_fit_data_2j_asym, s_eff_fit_data_2j_asym,
 cov_f_fit_data_2j_asym, cov_eff_fit_data_2j_asym) = pars
 
