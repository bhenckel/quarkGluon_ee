#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:35:37 2018
|=============================================================================|
|                Optimization of matrix fit using minuit                      |
|=============================================================================|
@author: benjamin
"""
# Importing Packages
from pandas import HDFStore
import pandas as pd
import numpy as np
from copy import copy
from tabulate import tabulate
from CM_extra_funcs import ( 
                            nnbjet_to_matrix,
                            degrees_of_freedom,
                            print_chi2_asym_no_fit,
                            initial_values_from_MC,
                            fit_object,                            
                            )
from CM_extra_funcs_benja import (
                                  error_prop,
                                  error_prop_adv,
                                  create_MC_matrix,
                                  create_data_matrix,
                                  even_events_binning,
                                  )
verbose = True
# Importing Data and MC.
#branches_to_import = None 
#filename = 'outputfile_Ref12.root'
#treename_data = 'ntuple_DataBtag'
#treename_MC = 'ntuple_McBtag'
# load real data
#df_data = load_data_file(filename, treename_data, branches_to_import, 
#                         verbose, save_type='hdf5')
# load MC
#df_MC = load_data_file(filename, treename_MC, branches_to_import, 
#                       verbose, save_type='hdf5')

store = HDFStore('raw_data.h5')
df_data = store['df_data']
df_MC = store['df_MC']
df_MC = df_MC[df_MC['datatype'] == 1]

df_data_2j = df_data[df_data['njet'] == 2.0] # Select data with njets = 2
df_data_3j = df_data[df_data['njet'] == 3.0] # Select data with njets = 3

df_MC_2j = df_MC[df_MC['njet'] == 2.0] # Select data with njets = 2
df_MC_3j = df_MC[df_MC['njet'] == 3.0] # Select data with njets = 3

del df_data
del df_MC

bin_min = 5 # Number of edges, actual minimum number of bins will be bin_min -1
bin_max = 10 # Number of edges, actual maximum number of bins will be bin_max -1
analysis_matrix = np.zeros(((bin_max+1)-bin_min,9))
results_eq_MC = np.zeros(((bin_max+1)-bin_min,15))
results_even_MC = np.zeros(((bin_max+1)-bin_min,15))
results_data = np.zeros(((bin_max+1)-bin_min,15))
chi2_values = [0, 0, 0, 0]
spacing_run = [True, False] # True indicates linear spacing and False indicates even N spacing.
strings = ['effeq%d.tex' % x for x in range(bin_min,(bin_max+1))] + ['effeven%d.tex' % x for x in range(bin_min,(bin_max+1))]
# Setting bin edges
for j in range(2):
    spacing = spacing_run[j]
    for i in range((bin_max+1)-bin_min):
        # This makes the binning even spaced
        if(spacing):
            bin_edges = np.around(np.linspace(0.0, 1.0, num = bin_min+(1+i)),decimals=2) #Initial values np.array([0.0, 0.15, 0.4, 0.6, 0.85, 0.96, 1.0])
        else:
        # This makes the binning even in # events.
            bin_edges = even_events_binning(bin_min+i, df_MC_2j)
            print(bin_edges)
                
        N_bins = len(bin_edges) - 1     
        N_flavors = 3
        assumption = 'sym'
        print('\n'*3+'- '*5 + 'Fitting over a range of linear spaced bin edges' + '- '*5+'\n')
        print('Current number of bins is ' + repr(N_bins))
        print('Number of flavors is ' + repr(N_flavors))
        print('The current bin edges are: ' + repr(bin_edges))
        
        print('\n'*3+'- '*5 + 'Number of bins, observables, variables and DOF: ' + '- '*5+'\n')
        N_obs, N_vars, N_dof = degrees_of_freedom(assumption, N_bins, N_flavors, verbose=verbose)
        print("\n\n")
        
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
        
        pars = initial_values_from_MC(matrix_MC_2j)
        N_total_2j, f_MC_2j, eff_MC_2j, C_MC_2j, S_MC_2j = pars
        
        S0_2j_sym = np.zeros_like(S_MC_2j)
        S0_2j_asym = copy(S_MC_2j)
        
        f0_2j = f_MC_2j[:-1]
        eff0_2j = eff_MC_2j[:-1, :]
        
        # below values from http://pdg.lbl.gov/2018/listings/rpp2018-list-z-boson.pdf
        f0_2j_theoretical = np.array([0.156/69.911*100 , 0.116/69.911*100])
        
        print('\n'*3+'- '*10 + 'Symmetric Fitting, MC ' + '- '*10+'\n'*3)
        
        fit_MC_2j_sym = fit_object(f0_2j, eff0_2j, S0_2j_sym, matrix_MC_2j_all, 
                             verbose, assumption, use_limits=False)
        fit_MC_2j_sym.initial_chi2_check()
        fit_MC_2j_sym.fit()
        
        pars = fit_MC_2j_sym.get_fit_values()
        (f_fit_MC_2j_sym, eff_fit_MC_2j_sym, s_f_fit_MC_2j_sym, 
         s_eff_fit_MC_2j_sym, cov_f_fit_MC_2j_sym, cov_s_fit_MC_2j_sym) = pars
    
        
        print('\n'*3+'- '*10 + 'Asymmetric Fitting, MC ' + '- '*10+'\n'*3)
        
        fit_MC_2j_asym = fit_object(f0_2j, eff0_2j, S0_2j_asym, matrix_MC_2j_all, 
                             verbose, assumption, use_limits=False)
        fit_MC_2j_asym.initial_chi2_check()
        fit_MC_2j_asym.fit()
        pars = fit_MC_2j_asym.get_fit_values()
        (f_fit_MC_2j_asym, eff_fit_MC_2j_asym, s_f_fit_MC_2j_asym, 
         s_eff_fit_MC_2j_asym, cov_f_fit_MC_2j_asym, cov_s_fit_MC_2j_asym) = pars
        
        
        
        
        print('\n'*3+'- '*10 + 'Symmetric Fitting, data ' + '- '*10+'\n'*3)
        
        fit_data_2j_sym = copy(fit_MC_2j_sym)
        
        fit_data_2j_sym.data = matrix_data_2j
        fit_data_2j_sym.f0 = f_fit_MC_2j_sym
        fit_data_2j_sym.eff0 = eff_fit_MC_2j_sym
        
        fit_data_2j_sym.initial_chi2_check()
        #fit_data_2j_sym.f0 = f0_2j_theoretical
        #fit_data_2j_sym.initial_chi2_check()
        
        fit_data_2j_sym.fit()
        pars = fit_data_2j_sym.get_fit_values()
        (f_fit_data_2j_sym, eff_fit_data_2j_sym, s_f_fit_data_2j_sym, 
         s_eff_fit_data_2j_sym, cov_f_fit_data_2j_sym, cov_s_fit_data_2j_sym) = pars
        
        print('\n'*3+'- '*10 + 'Asymmetric Fitting, data ' + '- '*10+'\n'*3)
        
        fit_data_2j_asym = copy(fit_MC_2j_asym)
        
        fit_data_2j_asym.data = matrix_data_2j
        fit_data_2j_asym.f0 = f_fit_MC_2j_asym
        fit_data_2j_asym.eff0 = eff_fit_MC_2j_asym
        
        fit_data_2j_asym.fit()
        pars = fit_data_2j_asym.get_fit_values()
        (f_fit_data_2j_asym, eff_fit_data_2j_asym, s_f_fit_data_2j_asym, 
         s_eff_fit_data_2j_asym, cov_f_fit_data_2j_asym, cov_s_fit_data_2j_asym) = pars
        
        # Analysis of results
        
        # Calculating third fraction and uncertainty as well as last bin efficiency and sigma, not available for unconverged fits.
        if fit_MC_2j_sym.fmin['is_valid']:
            f_fit_MC_2j_sym = np.append(f_fit_MC_2j_sym, (1-sum(f_fit_MC_2j_sym)))
            s_f_fit_MC_2j_sym = np.append(s_f_fit_MC_2j_sym, error_prop(s_f_fit_MC_2j_sym, cov_f_fit_MC_2j_sym))
            eff_fit_MC_2j_sym = np.vstack([eff_fit_MC_2j_sym, [1,1,1]-sum(eff_fit_MC_2j_sym)])
            s_eff_fit_MC_2j_sym = np.vstack([s_eff_fit_MC_2j_sym, error_prop_adv(N_bins, cov_s_fit_MC_2j_sym)])
            
        if fit_MC_2j_asym.fmin['is_valid']:
            f_fit_MC_2j_asym = np.append(f_fit_MC_2j_asym, (1-sum(f_fit_MC_2j_asym)))
            s_f_fit_MC_2j_asym = np.append(s_f_fit_MC_2j_asym, error_prop(s_f_fit_MC_2j_asym, cov_f_fit_MC_2j_asym))
            eff_fit_MC_2j_asym = np.vstack([eff_fit_MC_2j_asym, [1,1,1]-sum(eff_fit_MC_2j_asym)])
            s_eff_fit_MC_2j_asym = np.vstack([s_eff_fit_MC_2j_asym, error_prop_adv(N_bins, cov_s_fit_MC_2j_asym)])
            
        if fit_data_2j_sym.fmin['is_valid']:
            f_fit_data_2j_sym = np.append(f_fit_data_2j_sym, (1-sum(f_fit_data_2j_sym)))       
            s_f_fit_data_2j_sym = np.append(s_f_fit_data_2j_sym, error_prop(s_f_fit_data_2j_sym, cov_f_fit_data_2j_sym))
            eff_fit_data_2j_sym = np.vstack([eff_fit_data_2j_sym, [1,1,1]-sum(eff_fit_data_2j_sym)])
        if fit_data_2j_asym.fmin['is_valid']:
            f_fit_data_2j_asym = np.append(f_fit_data_2j_asym, (1-sum(f_fit_data_2j_asym)))       
            s_f_fit_data_2j_asym = np.append(s_f_fit_data_2j_asym, error_prop(s_f_fit_data_2j_asym, cov_f_fit_data_2j_asym))        
            eff_fit_data_2j_asym = np.vstack([eff_fit_data_2j_asym, [1,1,1]-sum(eff_fit_data_2j_asym)])
    
        Z_sym_MC = (f0_2j_theoretical[0] - f_fit_MC_2j_sym[0]) / f_fit_MC_2j_sym[0]
        Z_sym_data = (f0_2j_theoretical[0] - f_fit_data_2j_sym[0]) / f_fit_MC_2j_sym[0]
        Z_asym_MC = (f0_2j_theoretical[0] - f_fit_MC_2j_asym[0]) / f_fit_MC_2j_asym[0]
        Z_asym_data = (f0_2j_theoretical[0] - f_fit_data_2j_asym[0]) / f_fit_MC_2j_asym[0]
        
        analysis_matrix[i,:] = [N_bins, Z_sym_MC, fit_MC_2j_sym.fmin['is_valid'],
                               Z_asym_MC, fit_MC_2j_asym.fmin['is_valid'],
                               Z_sym_data, fit_data_2j_sym.fmin['is_valid'],
                               Z_asym_data, fit_data_2j_asym.fmin['is_valid'],
                               ]
        df_analysis = pd.DataFrame(data=analysis_matrix, columns=['N bins', 'Z sym MC', 'is valid','Z asym MC', 'is valid','Z sym data', 'is valid', 'Z asym data', 'is valid', ])
        chi2_values = [fit_MC_2j_sym.fmin["fval"], fit_MC_2j_asym.fmin["fval"], fit_data_2j_sym.fmin["fval"], fit_data_2j_asym.fmin["fval"]]
        if(j == 0):
            results_eq_MC[i,:] = create_MC_matrix(N_bins, f_fit_MC_2j_sym, s_f_fit_MC_2j_sym, f_fit_MC_2j_asym, s_f_fit_MC_2j_asym, chi2_values)
        else:
            results_even_MC[i,:] = create_MC_matrix(N_bins, f_fit_MC_2j_sym, s_f_fit_MC_2j_sym, f_fit_MC_2j_asym, s_f_fit_MC_2j_asym, chi2_values)

        results_data[i,:] = create_data_matrix(N_bins, f_fit_data_2j_sym, s_f_fit_data_2j_sym, f_fit_data_2j_asym, s_f_fit_data_2j_asym, chi2_values)
        
        df_results_eq_MC = pd.DataFrame(data=results_eq_MC, columns=['N bins', 'f b sym', '+/-', 'f c sym', '+/-', 'f light sym', '+/-', 'chi2', 'f b asym', '+/-', 'f c asym', '+/-', 'f light asym', '+/-', 'chi2'])
        df_results_even_MC = pd.DataFrame(data=results_even_MC, columns=['N bins', 'f b sym', '+/-', 'f c sym', '+/-', 'f light sym', '+/-', 'chi2', 'f b asym', '+/-', 'f c asym', '+/-', 'f light asym', '+/-', 'chi2'])
        with open(strings[i+(j*6)],'w') as tf:
            tf.write(tabulate(eff_fit_MC_2j_sym, tablefmt="latex"))
            tf.write(tabulate(s_eff_fit_MC_2j_sym, tablefmt="latex"))
            tf.write(tabulate(eff_fit_MC_2j_asym, tablefmt="latex"))
            tf.write(tabulate(s_eff_fit_MC_2j_asym, tablefmt="latex"))


print('\n'*3+'- '*10 + 'Results ' + '- '*10+'\n'*3)
print(df_results_eq_MC)
print(df_results_even_MC)
print('\n'*3+'- '*10 + 'Analysis of Results ' + '- '*10+'\n'*3)
print(df_analysis)
with open('results_eq.tex','w') as tf:
    tf.write(df_results_eq_MC.to_latex())
with open('results_even.tex','w') as tf:
    tf.write(df_results_even_MC.to_latex())