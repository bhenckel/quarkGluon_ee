#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:40:21 2018

@author: michelsen
"""

# to ignore deprecation warnings:
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uproot
from os import path
import os
import h5py
from collections import OrderedDict




#%%

sns.set()
sns.set(palette='Set1')

current_palette = sns.color_palette("Set1", n_colors=9)
color_dict = {}
# colors = 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'lime', 'cyan'
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey']

for i, color in enumerate(colors):
    color_dict[color] = current_palette[i]
color_dict['black'] = color_dict['k'] = (0, 0, 0)

#%%

def mem_usage(pandas_obj):
        """
        prints memory usage of pandas object
        """
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else: # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)


def load_data_file(filename, treename, branches_to_import=None, 
                   verbose=False, save_type='pickle', compression=False):
    
    if any(save_type.lower() == string for string in ['hdf5', 'h5', 'h5py']):
        file_ext = ".gzh5" if compression else ".h5"
        filename_h5 = filename.replace('.root', f'{treename}{file_ext}')
        if path.isfile(filename_h5):
            df = load_df_from_hdf(filename_h5, compression=compression)
        else:
            df = load_root_file(filename, treename, branches_to_import, verbose)
            save_hdf(df, filename_h5, compression=compression)
        return df
    
    else: 
        if not save_type.lower() == 'pickle':
            print("save_type is not 'pickle', but still defaults to this")
        filename_pickle = filename.replace('.root', f'{treename}.pkl')
        if path.isfile(filename_pickle):
            df = pd.read_pickle(filename_pickle)
        else:
            df = load_root_file(filename, treename, branches_to_import, verbose)
            df.to_pickle(filename_pickle)
        return df
    
    

def save_hdf(df, filename, compression=False):
    """ Saves a pandas DataFrame as filename .h5."""
    
    if path.isfile(filename): 
        os.remove(filename)
    
    with h5py.File(filename) as h5:
        h5.create_dataset(filename, data=df.to_records(index=False), 
                          compression=('gzip' if compression else None))
    return None


def load_df_from_hdf(filename, compression=False):
    """ Load pandas DataFrame from .h5 storage."""

    with h5py.File(filename, 'r') as h5:
        df = pd.DataFrame(h5[filename][:])

    df.reset_index(drop=True, inplace=True)
    return df


def load_root_file(filename, treename, branches_to_import=None, verbose=False):
    root_object = uproot.open(filename)[treename]
    data_dict = root_object.arrays(branches_to_import)
    data_dict = { key.decode(): val for key, val in data_dict.items() } # decodes byte strings
    
    df = pd.DataFrame(data=data_dict)
    
    print("\nMemory-optimizes the dataframe, please wait.\n")
    
    # integer columns:
    int_cols = ['evtn', 'flevt', 'njet', 'qmatch', 'datatype']
    for col in int_cols:
        df.loc[:, col] = df.loc[:, col].astype('int')
    
    # integer dtypes
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='signed')
    
    # floats dtypes
    df_float = df.select_dtypes(include=['float32'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    
    # combined floats and ints
    optimized_df = df.copy()
    optimized_df[converted_int.columns] = converted_int
    optimized_df[converted_float.columns] = converted_float
    
    if verbose:
        
        # initial overview of dataframe and memory usage
        print("")
        print(data_dict.keys())
        print("\nDataframe loaded with dimension: ", df.shape)
        print(df.info(memory_usage='deep'))
        print("")
        
        
        print("\nintegers, before and after")
        print(mem_usage(df_int))
        print(mem_usage(converted_int))
        compare_ints = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)
        compare_ints.columns = ['before','after']
        print(compare_ints.apply(pd.Series.value_counts))
        print("")
        
        # floats dtypes     
        print("floats, before and after")
        print(mem_usage(df_float))
        print(mem_usage(converted_float))
        compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes],axis=1)
        compare_floats.columns = ['before','after']
        print(compare_floats.apply(pd.Series.value_counts))
        print("")
        
         # combined floats and ints
        print("optimized dataframe")
        print(mem_usage(df))
        print(mem_usage(optimized_df))
        print("")
        
    return optimized_df


#%%
    
def check_all_equal(a):
    """ Small function that checks if all elements of an array are equal """
    return np.min(a) == np.max(a)


def check_qmatch_1_2_0(a):
    """ Small function that checks if the array a contains one of 0, 1, and 2"""
    return all([sum(a==i) == 1 for i in range(3)]) 


def from_df_to_array(df, column):
    """ Takes df as input and then converts it into a 2D-array with 
        a width of njet based on the mode of the njet column in the df    
    """
    
    # automatically find the njet number by using the mode
    njet = sp.stats.mode(df.njet).mode[0]
    
    # take the 1D-array and split it into a 2D-array of width njet
    L = [df[column].values[i::njet] for i in range(njet)]
    array = np.vstack(L).T
    return njet, array


def check_event_numbers_are_matching(df):    
    """ check that the two or three jets share same event number """
    
    njet, array_evtn = from_df_to_array(df, 'evtn')
    
    all_truth = (np.diff(array_evtn, axis=1) == 0).all()

    if not all_truth:
        raise ValueError(f"Event number doesn't match in {njet}-jet events")
    
    # if njet == 2:
    #     a = df.evtn.values[0::2]
    #     b = df.evtn.values[1::2]
    #     N_ab = sum(a==b)
    #     if not (N_ab == len(a) and N_ab == len(b)):
    #         raise ValueError("Error, event number doesn't match in 2 jet events")


def check_qmatch_are_correct(df):    
    """ check that three jets r """
    
    njet, array_qmatch = from_df_to_array(df, 'qmatch')
    
    all_truth = np.apply_along_axis(check_qmatch_1_2_0, 1, array_qmatch).all()

    if not all_truth:
        raise ValueError(f"qmatch does not match for {njet}-jet events")
    



    
class IllegalArgumentError(ValueError):
    pass


def train_test_split_non_random(X, y, test_size=0.2, train_size=None):
    """ Splits X,y into training and test set by taking the first p fraction
        and define that as training and the last 1-p fraction as test set.
        I.e. not a randomly taken subset.
        Takes either test_size or train_size as input.
    """
    
    
    if (test_size is None and train_size is None) or (
        test_size is not None and train_size is not None):
        e_str = 'Either test_size or train_size must be set, exclusively'
        raise IllegalArgumentError(e_str)
    
    N = len(y)
    N_train = int(N*(1-test_size)) if test_size is not None else int(N*train_size)
    
    X_train = X.iloc[:N_train]
    X_test = X.iloc[N_train:]
    y_train = y.iloc[:N_train]
    y_test = y.iloc[N_train:]
    
    return X_train, X_test, y_train, y_test



def train_test_index(X, test_size=0.2, train_size=None):
    """ Splits X,y into training and test set by taking the first p fraction
        and define that as training and the last 1-p fraction as test set.
        I.e. not a randomly taken subset.
        Takes either test_size or train_size as input.
    """
    
    
    if (test_size is None and train_size is None) or (
        test_size is not None and train_size is not None):
        e_str = 'Either test_size or train_size must be set, exclusively'
        raise IllegalArgumentError(e_str)
    
    N = len(X)
    N_train = int(N*(1-test_size)) if test_size is not None else int(N*train_size)
    
    X_train = X.iloc[:N_train]
    X_test = X.iloc[N_train:]
    
    return X_train.index, X_test.index



#%%
    
def plot_histogram_lin_log(data, ax, Nbins, 
                           title=None, 
                           xlabel=None,
                           xlim=None
                           ):

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not title is None:
        ax.set_title(title)
    if not xlim is None:
        ax.set_xlim(xlim)
    
    ax.hist(data, Nbins, histtype='step', color=color_dict['blue']) # range=(None, None)
    ax.set_ylabel('Counts, lin-scale', color=color_dict['blue'])
    ax.tick_params('y', colors=color_dict['blue'])
    
    ax2 = ax.twinx()
    ax2.hist(data, Nbins, histtype='step', log=True, color=color_dict['red']) # range=(None, None)
    ax2.set_ylabel('Counts, log-scale', color=color_dict['red'])
    ax2.tick_params('y', colors=color_dict['red'])

    return ax2




#%%%
    

class CV_EarlyStoppingTrigger:
    """
        Applies early stopping to xgb and lgb cv module with 
        specified evaluation function. 
        Does not cut off last values as the usual early stopping version does.
    """
    

    def  __init__(self, stopping_rounds, string='auc', maximize_score=True, 
                  method='xgb'):
        """
        :int stopping_rounds: Number of rounds to use for early stopping.
        :str string: Name of the evaluation function to apply early stopping to.
        :bool maximize_score: If True, higher metric scores treated as better.
        """
        
        self.stopping_rounds = stopping_rounds
        self.string = string
        self.method = method
        self.maximize_score = maximize_score
        
        self.reset_class()
        
        
    def reset_class(self):
        # TODO: implement in HousingPrices
        
        self.best_score = None
        self.best_iteration = 0
        self.iteration = 0
        self.do_reset_class = False


    def __call__(self, callback_env):
        
        if self.do_reset_class:
            self.reset_class()
        
        evaluation_result_list = callback_env.evaluation_result_list
        # print(evaluation_result_list)
        
        if self.method.lower() == 'xgb':
            names = [self.string, 'test']
            # test score for specific name:
            score = [x[1] for x in evaluation_result_list if 
                      all(y.lower() in x[0].lower() for y in names)][0]
        elif self.method.lower() == 'lgb':
                names = [self.string]
                # test score for specific name:
                score = [x[2] for x in evaluation_result_list if 
                          all(y.lower() in x[1].lower() for y in names)][0]
        else:
            raise IllegalArgumentError("'Method' has to be either xgb or lgb")
                
        
        # if first run
        if self.best_score is None:
            self.best_score = score
        
        # if better score than previous, update score
        if (self.maximize_score and score > self.best_score) or \
            (not self.maximize_score and score < self.best_score):
                self.best_iteration = self.iteration
                self.best_score = score
        
        # trigger EarlyStoppingException from callbacks library
        elif self.iteration - self.best_iteration >= self.stopping_rounds:

            self.do_reset_class = True
            
            if self.method.lower() == 'xgb':
                from xgboost.callback import EarlyStopException
                raise EarlyStopException(self.iteration)
            
            elif self.method.lower() == 'lgb':
                from lightgbm.callback import EarlyStopException
                # print('Raising Early stopping exception for lGB')
                # print('')
                raise EarlyStopException(self.iteration, self.best_score)
            
            
        self.iteration += 1

        



#%%
        
        
def get_cv_res_string(cv_res, substrings):
    col_names = list(cv_res.columns)
    return cv_res[[name for name in col_names 
                                if all(x in name for x in substrings)][0]]
        

def plot_cv_res(cv_res, ax, metrics, method='xgb', n_sigma=1):

    if method.lower() == 'xgb':
        str_train = 'train'
        str_test  = 'test'
        
    elif method.lower() == 'lgb':
        str_test = ''
        

    for metric, color in zip(metrics, ['red', 'blue', 'green']):
        
        # test results
        mean = get_cv_res_string(cv_res, ['mean', str_test, metric])
        std = get_cv_res_string(cv_res, ['std', str_test, metric])
        
        
        ax.plot(cv_res.index, mean, '-', label=metric.capitalize()+', Test',
                color=color_dict[color])
        ax.fill_between(cv_res.index, mean+n_sigma*std, mean-n_sigma*std,
                        color=color_dict[color], interpolate=True, alpha=0.1)
    
        if method.lower() == 'xgb':
            # get training results as well
            
            mean = get_cv_res_string(cv_res, ['mean', str_train, metric])
            std = get_cv_res_string(cv_res, ['std', str_train, metric])
            
            ax.plot(cv_res.index, mean, '--', label=metric.capitalize()+', Train',
                    color=color_dict[color])
            ax.fill_between(cv_res.index, mean+n_sigma*std, mean-n_sigma*std,
                    color=color_dict[color], interpolate=True, alpha=0.1)
    
    
    ax.legend(loc='best')
    ax.set(xlabel='Number of iterations', ylabel='Value')
    
    
    return None



def plot_cv_test_results(eval_res, method, metrics, title):
    
    
    if method.lower() == 'xgb':
        train = 'validation_0'
        test = 'validation_1'
    else:
        train = 'train'
        test = 'test'
    
    
    eval_steps = range(len(eval_res[train]['auc']))

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    
    for metric, color in zip(metrics, ['red', 'blue', 'green']):
    
        ax.plot(eval_steps, eval_res[train][metric], 
                color = color_dict[color], ls = '-', 
                label = metric.capitalize()+'Train')
        
        ax.plot(eval_steps, eval_res[test][metric], 
                color = color_dict[color], ls = '--', 
                label = metric.capitalize()+'Test')
        
        ax.legend()
        ax.set(xlabel='Number of iterations', ylabel='AUC', title=title)

    return fig, ax



def plot_random_search(rs_res, n_fold, title, ylim=(0.928, 0.936)):
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.errorbar(rs_res.index, 
                rs_res['mean_test_score'], 
                yerr=rs_res['std_test_score'] / np.sqrt(n_fold),
                fmt='.')
    
    ax.errorbar(rs_res.index[0], 
                rs_res['mean_test_score'].iloc[0], 
                yerr=rs_res['std_test_score'].iloc[0] / np.sqrt(n_fold),
                fmt='.r', )

    ax.set(title=title, xlabel='Iteration #', ylabel='AUC', ylim=ylim)

    fig.tight_layout()

    return fig, ax


#%%
    
import pickle
    
def save_model(clf, clf_name):
    pickle.dump(clf, open(clf_name, "wb"))


def load_model(clf_name):
    return pickle.load(open(clf_name, "rb"))


#%%
    


def degrees_of_freedom(assumption, N_bins, N_flavors, verbose=False):
    
    if assumption.lower() in ['sym', 'symmetrical']:
        N_obs = N_bins * (N_bins + 1) // 2
        N_vars = 1 * N_flavors * (N_bins - 1) + (N_flavors - 1)
    
    elif assumption.lower() in ['asym', 'asymmetrical']:
        N_obs = N_bins**2
        N_vars = 2 * N_flavors * (N_bins - 1) + (N_flavors - 1)
        
    else:
        raise IllegalArgumentError('"assumption" should be either sym or asym')
    
    N_dof = N_obs - N_vars
        
    
    if verbose:
        print(f'N_bins = {N_bins:>3}, \t N_obs = {N_obs:>3},\t ' + 
              f'N_vars = {N_vars:>3},\t N_dof = {N_dof:>3}')
    else:
        if N_dof < 0:
            string = ('Not enough degrees of freedom given N_bins, ' 
                    + 'N_flavor and the given assumption: ' 
                    +f'N_bins = {N_bins}, N_flavors = {N_flavors}, '
                    +f'assumption = {assumption}.'
                    )
            raise ValueError(string)

    return N_obs, N_vars, N_dof



#%%
    

def random_split_in_pairs(nnbjet, seed=42):
    """ Takes nnbjet array and randomly assigns one part of the pairs to 
        either nnbjet_first or nnbjet_second and the other part of the pair
        to the other array. 
    """
    
    N_pairs = len(nnbjet) // 2
    np.random.seed(seed)
    uniform = np.random.uniform(size=N_pairs) 

    mask = np.zeros_like(nnbjet, dtype=bool)
    mask[::2] = uniform < 0.5
    mask[1::2] = uniform >= 0.5 

    return nnbjet[mask], nnbjet[~mask]

def bin_edges_to_str(bin_edges):
    return [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]



def nnbjet_to_matrix(nnbjet, bin_edges, df_name):
    """ Takes nnbjet array as input and returns a count matrix as DataFrame
    """
    
    # names for dataframe, otherwise not needed
    names_index_cols = bin_edges_to_str(bin_edges)

    nnbjet_first, nnbjet_second = random_split_in_pairs(nnbjet)
    count_matrix = np.histogram2d(nnbjet_first, nnbjet_second, bin_edges)[0]
    
    count_matrix = pd.DataFrame(count_matrix, dtype='int',
                                    index=names_index_cols, columns=names_index_cols,
                                    )
    
    
    count_matrix.columns.name = df_name
    
    return count_matrix


def chi2_to_P(chi2, N_dof):
    """ Calculates the probability of obtaining a chi2-val higher (i.e. worse)
        than the given chi2-val with the given number of degrees of freedom.
    """
    return sp.stats.chi2.sf(chi2, N_dof)


def chi2_asym_no_fit(count_matrix):
    
    """ Takes a count_matrix as input and calculates the chi2-assymetri 
        sum( (N_ij - N_ji)^2 / (N_ij + N_ji) ) for upper and lower parts 
        of the count matrix. Returns the chi2-val, the number of degrees of 
        freedom and the chi2-probability
    """
    
    if isinstance(count_matrix, pd.DataFrame):
        count_matrix = count_matrix.values
    indices = np.triu_indices_from(count_matrix, k = 1)
    upper = count_matrix[indices]
    lower = count_matrix[indices[::-1]]
    chi2_asym = np.sum( (upper-lower)**2 / (upper+lower) )
    
    N_dof = len(lower)
    P = chi2_to_P(chi2_asym, N_dof)
    
    return chi2_asym, N_dof, P

def print_chi2_asym_no_fit(count_matrix, flavor):
    print("chi2_asym, {} \t {:8.3f},     {},     {:.3f}".format(
                                    flavor, *chi2_asym_no_fit(count_matrix)))




#%%

def check_matrices(matrices):
    if isinstance(matrices[0], pd.DataFrame):
        matrices = [matrix.values for matrix in matrices]
    return matrices



def calc_N_total_from_matrices(matrices):
    
    matrices = check_matrices(matrices)
        
    # the total number of events
    N_total = np.sum(np.sum(matrix) for matrix in matrices)
    return N_total


def calc_C_and_S(matrices, f, eff, set_C_diag_eq_1=True):
    
    matrices = check_matrices(matrices)
    
    N_flavors = len(matrices)
    N_total = calc_N_total_from_matrices(matrices)
    
    # calculate the corrections based on fraction N_ij / y_ij
    # and make sure that the diagonal is 1
    C_MC = np.array([matrices[i] / (N_total * f[i] * np.outer(eff[:, i], eff[:, i])) 
                    for i in range(N_flavors)])
    if set_C_diag_eq_1:
        [np.fill_diagonal(C_MC[i], 1) for i in range(N_flavors)]
    # C_MC_b, C_MC_c, C_MC_l = C_MC
    
    # Calculate S based in diag(S) == 0 
    S_MC = np.array([copy(C_MC[i]) - 1 for i in range(N_flavors)])
    # [np.fill_diagonal(S_MC[i], 0) for i in range(N_flavors)]
    # S_MC_b, S_MC_c, S_MC_l = S_MC
    
    return C_MC, S_MC


from copy import copy
    
def initial_values_from_MC(matrices):

    matrices = check_matrices(matrices)
    
    N_flavors = len(matrices)
    
    N_total = calc_N_total_from_matrices(matrices)
    
    # calculate the fractions for each quark type based on MC
    f_MC = np.array([np.sum(matrices[i]) / N_total for i in range(N_flavors)])
    f_MC_b, f_MC_c, f_MC_l = f_MC
    
    
    # calculate the efficiencies for each quark type based on MC 
    eff_MC = np.array([(np.sum(matrices[i], axis=0) 
                / np.sum(matrices[i])) for i in range(N_flavors)]).T
    eff_MC_b, eff_MC_c, eff_MC_l = eff_MC.T
    
    
    C_MC, S_MC = calc_C_and_S(matrices, f_MC, eff_MC, set_C_diag_eq_1=True)
    
    return N_total, f_MC, eff_MC, C_MC, S_MC





#%% cos(theta) analysis
    
def calc_f_eff_per_cos_theta_bin(df_MC_2j, bin_edges_cos_theta, bin_edges,
                                 mask_MC_2j_b, mask_MC_2j_c, mask_MC_2j_l):

    """ 
        Analysis of the fractions f and efficiencies eff as a function
        of abs(cos(theta)) value.
    """
    
    # first find the average of the absolute values of costheta
    avg = np.mean([np.abs(df_MC_2j['costheta'].iloc[0::2]), 
                   np.abs(df_MC_2j['costheta'].iloc[0::2])], axis=0)
    
    values = {}
    bin_edges_cos_theta_names = bin_edges_to_str(bin_edges_cos_theta)
    
    # loop over the different bins of the cos theta cuts:
    for i, bin_edge in enumerate(bin_edges_cos_theta[:-1]):
        
        # mask out the samples which are within the specific cos theta bin
        mask = np.logical_and(bin_edges_cos_theta[i] < avg, avg < bin_edges_cos_theta[i+1]) 
    
        # extendt the mask to make sure that each element is copied (ie. get both samples)
        mask_double = np.zeros(len(df_MC_2j), dtype=bool)
        mask_double[0::2] = mask
        mask_double[1::2] = mask
    
        # get the nnbjet values for the samples that are both of b-flavour and with the above mask
        nnbjet_b = df_MC_2j.loc[np.logical_and(mask_double, mask_MC_2j_b), 'nnbjet']
        matrix_b = nnbjet_to_matrix(nnbjet_b, bin_edges, 'matrix_b')
        
        # similar to above, however just for c-flavour
        nnbjet_c = df_MC_2j.loc[np.logical_and(mask_double, mask_MC_2j_c), 'nnbjet']
        matrix_c = nnbjet_to_matrix(nnbjet_c, bin_edges, 'matrix_c')
        
        # light
        nnbjet_l = df_MC_2j.loc[np.logical_and(mask_double, mask_MC_2j_l), 'nnbjet']
        matrix_l = nnbjet_to_matrix(nnbjet_l, bin_edges, 'matrix_c')
    
        # combine the different count matrices
        matrix_comb = [matrix_b, matrix_c, matrix_l]
    
        # calculate the values N_total, f, eff, C and S from the count matrices
        pars = initial_values_from_MC(matrix_comb)
        N_total_2j, f_MC_2j, eff_MC_2j, C_MC_2j, S_MC_2j = pars
    
        # save the values of f and eff:
        d = make_initial_values_dict(f_MC_2j, eff_MC_2j)
        names = list(d.keys())
        values[bin_edges_cos_theta_names[i]] = np.array(list(d.values()))
    
    # concatenate all the values into a pandas dataframe
    
    df = pd.DataFrame(values, index=names).T
    
    return df
    




dash_styles = OrderedDict([
     ('solid',               ()),
     
     ('dotted',              (2, 5)),
     ('densely dotted',      (2, 2)),
     
     ('dashed',              (5, 5)),
     ('densely dashed',      (5, 1)),

     ('dashdotted',          (3, 5, 1, 5)),
     ('densely dashdotted',  (3, 1, 1, 1)),
     
     ('dashdotdotted',         (3, 5, 1, 5, 1, 5)),
     ('densely dashdotdotted', (3, 1, 1, 1, 1, 1)),
     
     ('loosely dotted',      (1, 10)),
     ('loosely dashed',      (5, 10)),
     ('loosely dashdotted',  (3, 10, 1, 10)),
     ('loosely dashdotdotted', (3, 10, 1, 10, 1, 10))])
     


def plot_cos_theta_analysis(ax, df_cos_theta_analysis, return_ax=False):
    
    N_bins_cos_theta = len(df_cos_theta_analysis)
    
    for colname, col in df_cos_theta_analysis.iteritems():
        
        if colname.startswith('f'):
            
            color = color_dict['blue']
            
            if 'b' in colname:
                dashes = dash_styles['solid']
            elif 'c' in colname:
                dashes = dash_styles['dashed']
                
        else:
            if 'b' in colname:
                color = color_dict['red']
            elif 'c' in colname:
                color = color_dict['green']
            elif 'l' in colname:
                color = color_dict['purple']
            
            for i, dash in enumerate(dash_styles.values()):
                if str(i) in colname:
                    dashes = dash
        
        x = np.arange(N_bins_cos_theta) + 0.2
        y = col.values
        
        ax.plot(x, y, label=col.name, color=color, dashes=dashes)
    
    ax.set(xlabel=r'$|\cos(\theta)|$', ylabel=r'value of $f$ or $\epsilon$', 
           xlim=(0, N_bins_cos_theta-0.1))
    ax.set_xticks(x)
    ax.set_xticklabels(list(df_cos_theta_analysis.index))
    ax.legend(loc='upper right')


    if return_ax:
        return ax
    else:
        return None









#%% fitting
    

def from_1D_to_2D(lst, N_bins, N_flavors):
    """ Takes a 1D-list/array and converts it into a 2D-array 
        with correct dimensions
    """
    if isinstance(lst, pd.DataFrame):
        lst = lst.values
    elif isinstance(lst, list):
        lst = np.array(lst)
    return lst.reshape((N_flavors, N_bins-1)).T

def from_2D_to_1D(arr, N_bins, N_flavors):
    """ Takes a 2D-array and converts it into a 1D-array 
        with correct dimensions
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    return arr.T.flatten()


def from_input_vals_to_output(f, eff, S):
    """ Takes 'reduced' 
        - fractions f           with shape (bins, flavors-1), 
        - efficiencies eff      with shape (bins-1, flavors),
        
        and calculates the missing values based on the constrains:
            sum(f) = 1
            sum(eff_k) = 1
    """
    
    n_bins, n_flavors = eff.shape
    n_bins += 1
    
    f_all = np.zeros(n_flavors) 
    f_all[:-1] = f
    f_all[-1] = 1 - np.sum(f)
    
    eff_all = np.zeros((n_bins, n_flavors))
    eff_all[:-1, :] = eff
    eff_all[-1, :] = 1 - np.sum(eff, axis=0)
    
    C_all = np.ones_like(S) + S

    return f_all, eff_all, C_all


def calc_sum_in_y(f_k, eff_k, C_k):
    """ Calculates the individual terms in the sum of:
        sum f_k * E_k * C_k """
    return f_k * np.outer(eff_k, eff_k) * C_k


def calc_y(N_tot, f, eff, S):
    """ Given N_tot, f, eff and S, calculates y """
    
    f_all, eff_all, C_all = from_input_vals_to_output(f, eff, S)

    n_bins, n_flavors = eff_all.shape
    
    # compute list of matrices  based in the outer product of the columns in eff
    sum_in_y_all = np.array([calc_sum_in_y(f_all[i], eff_all[:, i], C_all[i, :, :]) 
                        for i in range(n_flavors)])
    
    y = N_tot * np.sum(sum_in_y_all, axis=0)
    return y



# below alternitve ways of writing it, faster

# def func1():
#     Eff_all = [np.outer(eff_all[:, i], eff_all[:, i]) for i in range(n_flavors)]
#     F_all = [f_all[i] * Eff_all[i] for i in range(n_flavors)]
#     return np.sum(F_all, 0)

# def func2():
#     Eff_all_np = np.array(Eff_all)
#     f_all_np = f_all.reshape((1, 3))
#     return np.trace(np.tensordot(f_all_np, Eff_all_np, axes=0)[0])

# def func3():
    # eff2 = np.array([eff_all[:, i] for i in range(3)])
    # return np.tensordot((np.diag(f_all) @ eff2).T, eff2, axes=1)








def calc_chi2_sym(data, y):
    """ Calculates the chi2-value between data and y by assuming that
        the data is symmetrically distributed, ie. only need the upper 
        diagonal of the data (incl. diagonal).
    
    """
    indices = np.triu_indices_from(data, k = 0)
    
    data_sym = data + data.T
    y_sym = y + y.T
    
    frac = (data_sym - y_sym)**2 / y_sym
    chi2 = np.sum(frac[indices])
    
    return chi2


def flatten_list_of_lists(lst):
    """ Flattens a lists of lists into a 1D-list """
    return [item for sublist in lst for item in sublist]


def make_initial_values_dict(f, eff):
    """ Takes f and eff and returns a dictionary with the correct naming
        of the variables i.e. prepares the values to be used by iminuit.
    """

    d = OrderedDict()

    nbins, nflavors = eff.shape
    nbins += 1

    f_names = ['f_b', 'f_l']
    if nflavors == 3:
        f_names.insert(1, 'f_c')
    for name, val in zip(f_names[:-1], f):
        d[name] = val
        
    eff_names = [[f'eff_b{i}' for i in range(nbins-1)], 
                 [f'eff_l{i}' for i in range(nbins-1)]]
    if nflavors == 3:
        eff_names.insert(1, [f'eff_c{i}' for i in range(nbins-1)])
    eff_names = flatten_list_of_lists(eff_names)
    for name, val in zip(eff_names, from_2D_to_1D(eff, nbins, nflavors)):
        d[name] = val
        
    return d



def list_of_pars_to_f_eff(nbins, nflavors, *fit_pars):
    """ The inverse function of make_initial_values_dict. 
        Takes the 1D-list of fit_pars and returns f and eff
        in correct dimensions.
    """
    
    f = np.array(fit_pars[:nflavors-1])
    eff_1D = np.array(fit_pars[nflavors-1:])
    eff_2D = from_1D_to_2D(eff_1D, nbins, nflavors)
    
    return f, eff_2D



def set_limits_on_params(d):
    """ Sets limits on the fit_pars used by iminuit """
    d_err = OrderedDict()
    for name in d.keys():
        d_err['limit_'+name] = (0, 1)
    return d_err




from iminuit.util import make_func_code


class minuit_wrapper:  
    
    """ A wrapper module for minuit such that minuit can describe the 
        object created by this class and thus also fit it.
        
        Takes in the actual data, the symmetry-corrections S and the 
        initial values dictionary d
    """
    
    
    def __init__(self, data, S, d):
        
        self.data = data
        self.S = S
        self.d = d
        self.func_code = make_func_code(list(d.keys()))
        
        self.N_flavors, self.N_bins, _ = S.shape
        if not (self.N_flavors == 2 or self.N_flavors == 3):
            raise IllegalArgumentError('Number of flavors should be 2 or 3')
    

    def __call__(self, *pars):  # par are a variable number of model parameters
        
        f, eff = list_of_pars_to_f_eff(self.N_bins, self.N_flavors, *pars)
        N_tot = np.sum(self.data)
        
        y = calc_y(N_tot, f, eff, self.S)
        chi2 = calc_chi2_sym(self.data, y)
        
        return chi2

#%%
        
    

def get_f_eff_covariance_matrices(cov_dict):
    
    """
        Takes iminuit covariance dict as input and returns the covariance 
        matrices (as DataFrames) for fractions (f) and efficiencies (eff).
    """
    
    if cov_dict is None:
        return None, None
 
    tuples = list(cov_dict.keys())
    index = pd.MultiIndex.from_tuples(tuples)
    
    cov_series = pd.Series(cov_dict, index=index)
    
    cov = cov_series.unstack()
    
    f_names = [s for s in cov.columns if s.startswith('f_')]
    eff_names = [s for s in cov.columns if s.startswith('eff_')]
    
    cov_f = cov.loc[f_names, f_names]
    cov_eff = cov.loc[eff_names, eff_names]
    
    return cov_f, cov_eff    

        
from iminuit import Minuit

class fit_object:
    
    def __init__(self, f0, eff0, S0, data, verbose, assumption, use_limits=False):
        self.f0 = f0
        self.eff0 = eff0
        self.S0 = S0
        self.data = data
        
        self.verbose = verbose
        self.assumption = assumption.lower()
        self.use_limits = use_limits
        
        self.N_bins, self.N_flavors = eff0.shape
        self.N_bins += 1
        
        self.N_obs, self.N_vars, self.N_dof = degrees_of_freedom(assumption,
                                                                 self.N_bins, 
                                                                 self.N_flavors, 
                                                                 verbose)
        
        
    def initial_chi2_check(self):
        self.y0 = calc_y(np.sum(self.data), self.f0, self.eff0, self.S0)
        self.chi2_0 = calc_chi2_sym(self.data, self.y0)
        
        
        print(f'\nChi2-value based on MC-values of f and eff with ' +
              f'no fit: {self.chi2_0:.4f}')
        print(f'Number of degrees of freedom: {self.N_obs}')
        print(f'Chi2-probability: {chi2_to_P(self.chi2_0, self.N_obs):.4f}')


    def fit(self):
        
        self.d_initial_vals = make_initial_values_dict(self.f0, self.eff0)
        self.d_limits = set_limits_on_params(self.d_initial_vals)
        
        self.chi2_minimize = minuit_wrapper(self.data, self.S0, self.d_initial_vals)
    
        if not self.use_limits:
            self.m = Minuit(self.chi2_minimize, errordef=1, pedantic=False, 
                       **self.d_initial_vals)
        else:
            self.m = Minuit(self.chi2_minimize, errordef=1, pedantic=False, 
                       **self.d_initial_vals, **self.d_limits)
        
        self.fmin, self.param = self.m.migrad()
        
        if not self.fmin['is_valid']:
            warnings.warn('\n\nFit is not valid!\n\n', FitWarning)
        
            
        self.chi2_fit = self.m.fval
        print(f'Fitted Chi2-value from Minuit: {self.chi2_fit:.4f}')
        print(f'Number of degrees of freedom: {self.N_dof}')
        print(f'Fitted Chi2-probability: {chi2_to_P(self.chi2_fit, self.N_dof):.4f}')
        if self.verbose:
            print("\n")
            print(self.fmin)
        
        self.f_fit, self.eff_fit = list_of_pars_to_f_eff(self.N_bins, 
                                                         self.N_flavors, 
                                                         *self.m.values.values())
        self.s_f_fit, self.s_eff_fit = list_of_pars_to_f_eff(self.N_bins, 
                                                             self.N_flavors, 
                                                             *self.m.errors.values())
        
        self.cov_f, self.cov_eff = get_f_eff_covariance_matrices(self.m.covariance)
        
        return (self.f_fit,   self.eff_fit, 
                self.s_f_fit, self.s_eff_fit, 
                self.cov_f,   self.cov_eff)
    
    
    
    def get_fit_values(self):
        if self.s_f_fit is None:
            return self.fit()
        else:
            return (self.f_fit,   self.eff_fit, 
                    self.s_f_fit, self.s_eff_fit, 
                    self.cov_f,   self.cov_eff)
    


    # Below is a method to make sure that if we input a pandas DataFrame 
    # as data it is converted to a numpy array

    
    def check_data(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        return data
    

    @property
    def data(self):
        # print("Getting data")
        return self.__data

    @data.setter
    def data(self, data):
        # print("Setting data")
        self.__data = self.check_data(data)



    # def update_parameters(self, data):
    #     """
            
    #     """
        
    #     if isinstance(data, pd.DataFrame):
    #         data = data.values
    #     self.data = data
        
    #     self.f0, self.eff0, _, _ = self.get_fit_values()
        



#%% PandasContainer
    



import warnings

class IllegalArgumentWarning(UserWarning):
    pass

class FitWarning(UserWarning):
    pass



def get_dict_split(y_train, y_test, y_val = None):
    index_train = y_train.index
    index_test = y_test.index
    
    dict_split = {'train': index_train, 'test': index_test}
    
    if y_val is not None:
        dict_split['val'] = y_val.index
    
    return union_of_all_indices_in_dict(dict_split)



def get_dict_flavor(y_MC):
    
    """
        Takes y_MC_3f series and finds all the indices of b's, c's, l's and cl's.
        In the end also adds the key 'all' which is simply the union of all.
    
        Input: y_MC with values: 0 for l, 1 for b and 2 for c
    """

    if (y_MC == 2).sum() == 0:
        warnings.warn('Input y_MC has not a single "2" in it.', IllegalArgumentWarning)

    index_b = y_MC[y_MC == 1].index
    index_c = y_MC[y_MC == 2].index
    index_l = y_MC[y_MC == 0].index
    index_cl = index_c.union(index_l)

    dict_flavor = {'b': index_b, 
                   'c': index_c, 
                   'l': index_l, 
                   'cl': index_cl,
                   'all': y_MC.index}
    
    return dict_flavor


def unpack(lst, sep=', '):
    return sep.join(str(x) for x in lst)  # map(), just for kicks

def split_by_indices(series, index1, index2):
        return series.loc[index1.intersection(index2)]


def union_of_all_indices_in_dict(d_index):
        
        """ Takes a dictionary of indices and calculates the union of all 
            the indices which it adds to the original dictionary with the
            key 'all'.
        """
        
        key0 = next(iter(d_index))
        val0 = d_index[key0]
        
        union_index = val0
        for key, val in d_index.items():
            union_index = union_index.union(val)
        d_index['all'] = union_index
        return d_index



#%%
    



class PandasContainer():
    
    
    """ Container for Pandas Dataframes/Series which allows one to use bracket
        notation to get specific flavours (b,c,l,cl,all), splits (train, test, 
        all) or a combination of both.  
    """
    
    
    def __init__(self, df, dict_flavor, dict_split, max_rows=5, max_cols=7):
        
        self.df = df
        self.set_container_dicts(dict_flavor, dict_split)

        #only print max_rows in dataframe
        self.set_max_rows(max_rows)
        
        # colors for flavors
        self.color = {'b': 'blue', 'c': 'green', 'l': 'red', 'cl': 'orange'}
        
        # below way of using the GetColumnClass (i.e. square bracket indexing)
        self._get_column_as_dfc = GetColumnClass(self)

    def set_container_dicts(self, dict_flavor, dict_split):
        
        if isinstance(self.df, pd.DataFrame):
            if len(set(dict_flavor.keys()).intersection(self.df.columns)) != 0:
                raise ValueError('Overlap between dict_flavor keys and dataframe variables!')
            if len(set(dict_split.keys()).intersection(self.df.columns)) != 0:
                raise ValueError('Overlap between dict_split keys and dataframe variables!')
            
        
        # self.dict_flavor = union_of_all_indices_in_dict(dict_flavor)
        # self.dict_split = union_of_all_indices_in_dict(dict_split)
        self.dict_flavor = dict_flavor
        self.dict_split = dict_split
        return None
        

    def __getitem__(self, arg):
        """
        blabla.
        """
        
        arg_optional = None
        
        if isinstance(arg, slice):
            # print("TODO: returning entire df")
            return self.df
        
        
        if isinstance(arg, tuple) or isinstance(arg, list):
            
            # print("TODO: arg is tuple or list")
            
            if len(arg) == 1:
                arg = arg[0]
            elif len(arg) == 2:
                arg, arg_optional = arg
            else:
                raise ValueError('argument has to be less than 3 elements')
        
        if len(arg) == 0 or arg.lower()=='all':
            # print("TODO: returning entire df")
            return self.df
                
        if isinstance(arg, str):
            # print("TODO: arg is str")
            result = self.__custom_get_item(arg, arg_optional)
            
            if result is not None: 
                # print("TODO: result is not None")
                return result
        
        
        raise ValueError('argument in call (dfc[arg]) - must be in either '+
                         'dict_flavor or dict_split.')
        
    def __repr__(self):
        
        s = self.df.__repr__() + '\n\n'
        s += 'dict_flavor keys: ' + str(list(self.dict_flavor.keys()))
        s += '\n' 
        s += 'dict_split keys:  ' + str(list(self.dict_split.keys()))
        return s
    
    
    def __custom_get_item(self, arg, arg_optional):
    
        if arg == 'all':
            if arg_optional is not None:
                s = 'When the first argument is "all", the entire series is '
                s += 'being returned (i.e. ignores second argument).'
                warnings.warn(s, IllegalArgumentWarning)
        
        if arg_optional is None:
            if arg in self.dict_flavor.keys():
                return split_by_indices(self.df, self.dict_flavor[arg], self.dict_split['all'])
            elif arg in self.dict_split.keys():
                return split_by_indices(self.df, self.dict_flavor['all'], self.dict_split[arg])
            
        if (arg in self.dict_flavor.keys() and arg_optional in self.dict_split.keys()):
            return split_by_indices(self.df, self.dict_flavor[arg], self.dict_split[arg_optional])
        
        if (arg_optional in self.dict_flavor.keys() and arg in self.dict_split.keys()):
            return split_by_indices(self.df, self.dict_flavor[arg_optional], self.dict_split[arg])

        return None
        
    # below method to set the varible get_column_as_dfc to immutable. 
    @property
    def get_column_as_dfc(self):
        return self._get_column_as_dfc

    def set_max_rows(self, max_rows):
        #only print max_rows in dataframe
        pd.set_option('display.max_rows', max_rows)
        return None
    
    def set_max_cols(self, max_cols):
        #only print max_cols in dataframe
        pd.set_option('display.max_cols', max_cols)
        return None


    # original dataframe iterator
    def iteritems(self):
        return self.df.iteritems()
    
    # original dataframe iterator
    def iterrows(self):
        return self.df.iterrows()

    


    # iterate over flavors
    def iterflavors(self, include_all=False, include_cl=False, 
                    include_color=False):
        for flavor in self.dict_flavor.keys():
            if ((include_all or flavor.lower() != 'all') and 
                (include_cl or flavor.lower() != 'cl')):
                if include_color:
                    yield flavor, self.__getitem__(flavor), self.color[flavor]
                else:
                    yield flavor, self.__getitem__(flavor)
            
    # iterate over splits
    def itersplits(self, include_all=False, include_cl=False):
        for split in self.dict_split.keys():
            if ((include_all or split.lower() != 'all') and 
                (include_cl or split.lower() != 'cl')):                
                yield split, self.__getitem__(split)
    

class GetColumnClass():
    
    """ Class that allows us to retrieve the parent's data frame series with 'colname'
        by using square bracket notation. 
    """
    
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, colname):
        col = self.parent.df.loc[:, colname]
        dict_flavor = self.parent.dict_flavor
        dict_split = self.parent.dict_split
        return PandasContainer(col, dict_flavor, dict_split) 
    
    
        
    
    

# def PandasWrapperFunction(base):
       
        

    # class PandasWrapper(base):
    #     __doc__ = inspect.cleandoc("""Custom modification of Pandas module. 
    #                                This one allows one to 
    #                                set_wrapper_dicts(dict_flavor, dict_split) 
    #                                and thus index the dataframe 
    #                                df['b', 'train'] \n\n""")
    #     __doc__ += base.__doc__
    
    #     def __init__(self, *args, **kwargs):
    #         super(PandasWrapper, self).__init__(*args, **kwargs)
    
    
    #     def set_wrapper_dicts(self, dict_flavor, dict_split):
    #         for s in ['is_wrapper', 'dict_flavor', 'dict_split']:
    #             self._metadata.append(s)
            
    #         if len(dict_flavor.keys().intersection(self.columns)) != 0:
    #             raise ValueError('Overlap between dict keys and dataframe variables!')
            
            
    #         self.is_wrapper = True
    #         self.dict_flavor = union_of_all_indices_in_dict(dict_flavor)
    #         self.dict_split = union_of_all_indices_in_dict(dict_split)
    
    #     def set_wrapper(self, bool_val):
    #         self.is_wrapper = bool_val
            
    #     def default(self):
    #         self.is_wrapper = None
    
    #     def __getitem__(self, arg):
    #         """
    #         blabla.
    #         """
            
    #         if hasattr(self, 'is_wrapper'):
                
    #             print("TODO: hasattr")
                
    #             if self.is_wrapper:
                    
    #                 print("is_wrapper")
            
    #                 arg_optional = None
                    
    #                 if isinstance(arg, tuple):
                        
    #                     print("TODO: arg is tuple")
                        
    #                     if len(arg) == 2:
    #                         arg, arg_optional = arg
                            
    #                 if isinstance(arg, str):
    #                     print("TODO: arg is str")
    #                     result = self.custom_get_item(arg, arg_optional)
                        
    #                     if result is not None: 
    #                         print("TODO: result is not None")
    #                         return result
            
    #         print("TODO: returning result")
    #         result = super(PandasWrapper, self).__getitem__(arg)
    #         # print("TODO: changing class")
    #         # result.__class__ = PandasWrapper
    #         return result
        
        
    #     # result = super(PandasWrapper, self).__getitem__(key)
    #     #     result.__class__ = PandasWrapper
            
    #     #     return result
        
            
    #     def __repr__(self):
            
    #         s = super(PandasWrapper, self).__repr__() + '\n\n'
            
    #         if hasattr(self, 'is_wrapper'):
    #             if self.is_wrapper:
            
    #                 s += 'dict_flavor keys: ' + str(list(self.dict_flavor.keys())) + 2*'\n' 
    #                 # s += f'len(dict_flavor) = {len(self.dict_flavor)}' + 2*'\n' 
    #                 s += 'dict_split keys: ' + str(list(self.dict_split.keys())) + 2*'\n'
    #                 # s += f'len(dict_split) = {len(self.dict_split)}' + 2*'\n' 
    #         return s
        
        
        
        
    #     def custom_get_item(self, arg, arg_optional):
        
    #         if arg == 'all':
    #             if arg_optional is not None:
    #                 s = 'When the first argument is "all", the entire series is '
    #                 s += 'being returned (i.e. ignores second argument).'
    #                 warnings.warn(s, IllegalArgumentWarning)
            
    #         if arg_optional is None:
    #             if arg in self.dict_flavor.keys():
    #                 print("\nA\n")
    #                 return split_by_indices(self, self.dict_flavor[arg], self.dict_split['all'])
    #             elif arg in self.dict_split.keys():
    #                 print("\nB\n")
    #                 return split_by_indices(self, self.dict_flavor['all'], self.dict_split[arg])
                
    #         if (arg in self.dict_flavor.keys() and arg_optional in self.dict_split.keys()):
    #             print("\nC\n")
    #             return split_by_indices(self, self.dict_flavor[arg], self.dict_split[arg_optional])
            
    #         elif (arg_optional in self.dict_flavor.keys() and arg in self.dict_split.keys()):
    #             print("\nD\n")
    #             return split_by_indices(self, self.dict_flavor[arg_optional], self.dict_split[arg])
            
    #         return None
        
        
        
        
    #     def __finalize__(self, other, method=None, **kwargs):
    #         """propagate metadata from other to self """
            
    #         for name in self._metadata:
    #             object.__setattr__(self, name, getattr(other, name, None))
    #         return self

        
        
        
    #     def copy(self, deep=True):
    #         """
    #         Make a copy of this PandasWrapper object
    #         Parameters
    #         ----------
    #         deep : boolean, default True
    #             Make a deep copy, i.e. also copy data
    #         Returns
    #         -------
    #         copy : PandasWrapper
    #         """
    #         # FIXME: this will likely be unnecessary in pandas >= 0.13
    #         data = self._data
    #         if deep:
    #             data = data.copy()
    #         return PandasWrapper(data).__finalize__(self) # TODO probabily dont need finalize
        
        
    #     # def __getitem__(self, key):
    #     #     """
    #     #     If the result is a column containing only 'geometry', return a
    #     #     GeoSeries. If it's a DataFrame with a 'geometry' column, return a
    #     #     GeoDataFrame.
    #     #     """
            
    #     #     result = super(PandasWrapper, self).__getitem__(key)
    #     #     result.__class__ = PandasWrapper
            
    #     #     return result
            
    #         # result = super(PandasWrapper, self).__getitem__(key)
    #         # geo_col = self._geometry_column_name
    #         # if isinstance(key, string_types) and key == geo_col:
    #         #     result.__class__ = GeoSeries
    #         #     result.crs = self.crs
    #         #     result._invalidate_sindex()
    #         # elif isinstance(result, DataFrame) and geo_col in result:
    #         #     result.__class__ = PandasWrapper
    #         #     result.crs = self.crs
    #         #     result._geometry_column_name = geo_col
    #         #     result._invalidate_sindex()
    #         # elif isinstance(result, DataFrame) and geo_col not in result:
    #         #     result.__class__ = DataFrame
    #         # return result
    
    
#     return PandasWrapper


# SeriesWrapper = PandasWrapperFunction(pd.Series)
# DataFrameWrapper = PandasWrapperFunction(pd.DataFrame)


