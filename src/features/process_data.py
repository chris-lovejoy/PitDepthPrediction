# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:34:57 2020

@author: gmueller

use:
sd0 = process_signal(data0)
"""

import os
import sys
path = os.path.dirname(os.getcwd())
new_path = os.path.join(path)
sys.path.insert(0,new_path)

import pandas as pd
import numpy as np                       #visualisation
         #visualisation
     
#sns.set(color_codes=True)
from constants import FilePath,Channel
#from data.make_dataset import CompleteData
from pathlib import Path
import scipy.signal as signal
from scipy import stats


#function to remove linear trend and shift the sinal to be centered at zero at the ends
def detrend_and_center (data):
    #detrends and sets edges to zero 
    # data has to be numeric columns only
    num=data.shape[0]
    data_det = subtract_linear_trend(data)
     
    ave1 = pd.DataFrame(data_det[data_det.index<5].sum(axis=0)/5).T
    
    return data_det-(ave1.loc[0,:]).tolist()


#function to calculate and subtrac a linear trend in the data
def subtract_linear_trend(data):
    #detrends - data has to be numeric columns only
    my_headers=data.columns
    num=data.shape[0]
   
    ave1 = pd.DataFrame(data[data.index<5].sum(axis=0)/5).T
    ave2 = pd.DataFrame(data[data.index>=num-5].sum(axis=0)/5).T

    DXY = ave2-ave1
    dx = num-5
    gradient = DXY.div(dx)

    x=np.arange(0,num)
    lines=pd.DataFrame(np.tile(x, (40,1))).T
    data_delta =lines.mul(gradient.loc[0,:].tolist())
    data_delta.columns=my_headers

    data_det=data-data_delta

    return data_det


#function to get signal to noise ratio (from old numpy versin)
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, abs(m/sd))


#This is the main function that processes the data set
#1) detrend signal
#2)) smooth data with polynomial filter
#3) for columns with poor signal to noise ratio smooth more strongly
#
#    

def process_signal(data):
    # id_list = [data['Tube_Alias'][0], data['Flaw_ID'][0],str(data['Angle'][0])]
    # full_id = "_".join(id_list)
    # print('Preprocessing data for {}'.format(full_id))
    data_num=data[Channel.ALL]
    my_headers = data_num.columns 
    data_det= detrend_and_center(data_num)
    smooth_data = pd.DataFrame(signal.savgol_filter(data_det, 3, 2, deriv=0, delta=1.0, axis=0))
    smooth_data.columns=my_headers
    smooth_data.head()

    for col in smooth_data:
        #if the noise is still very high smooth more 
        test=pd.DataFrame(signaltonoise(smooth_data, axis=0, ddof=0)< 0.02).T
        test.columns=my_headers
        if test[col][0]:
            data[col]=pd.DataFrame(signal.savgol_filter(smooth_data[col],
                                  9, 2, deriv=0, delta=1.0, axis=0),columns=[col])
    
    return smooth_data

