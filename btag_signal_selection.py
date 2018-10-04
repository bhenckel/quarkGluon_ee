#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:07:35 2018

@author: benjamin
"""
import pandas as pd
import numpy as np


lgb_btags = pd.read_hdf('lgb_btags_3jet.h5')
sklrf_btags = pd.read_hdf('sklrf_btags_3jet.h5')

array = sklrf_btags.values
array_lgb = lgb_btags.values

sklrf_data = []
lgb_data = []
count_sig = 0
count_bckgr = 0
count_war = 0
check_war = False
print ("Selection of Signal and Background events")
for i in range(0,array.shape[0],3):
    sum = array[i,2]+array[i+1,2]+array[i+2,2]
    if(sum == 3 and array[i,1] == 5):
        sklrf_data.append([array[i,0],array[i+1,0],array[i+2,0],1])
        lgb_data.append([array_lgb[i,0],array_lgb[i+1,0],array_lgb[i+2,0],1])
        count_sig += 1
    else:
        sklrf_data.append([array[i,0],array[i+1,0],array[i+2,0],0])
        lgb_data.append([array_lgb[i,0],array_lgb[i+1,0],array_lgb[i+2,0],0])
        count_bckgr += 1
    if((i % 100000 ) == 0 and check_war):
        print ("qmatch: %i, %i, %i and sum = %i" %(array[i,2],array[i+1,2],array[i+3,2],sum))
    if(array[i,2] == array[i+1,2] == array[i+2,2] == 1):
        print ("warning all matched as quarks")
        count_war += 1

sklrf_data = np.reshape(sklrf_data, (-1,4))
lgb_data = np.reshape(lgb_data, (-1,4))

print (f'Selection Done - Results below')
print (f'Number of signal events: {count_sig}')
print (f'Number of background events: {count_bckgr}')
print (f'Number of qmatch warnings: {count_war}')

df_lgb = pd.DataFrame(data=lgb_data, columns=['btag1', 'btag2', 'btag3','label'])
df_lgb.to_hdf('lgb_events_btag.h5', key='df', mode='w')
df_sklrf = pd.DataFrame(data=sklrf_data, columns=['btag1', 'btag2', 'btag3','label'])
df_sklrf.to_hdf('sklrf_events_btag.h5', key='df', mode='w')