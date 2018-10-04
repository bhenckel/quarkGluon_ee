#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:44:17 2018

@author: benjamin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
import joblib


if joblib.cpu_count() >=30:
    cpu_n_jobs = 15
else:
    cpu_n_jobs = joblib.cpu_count()-1
    
lgb_data = pd.read_hdf('lgb_events_btag.h5')
sklrf_data = pd.read_hdf('sklrf_events_btag.h5')

# Produce two selections based on cuts - A tight and a loose selection.

