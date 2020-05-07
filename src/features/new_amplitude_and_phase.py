# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:53:43 2020

@author: gmueller

Functions for extracting phase and amplitude from  fitting a trandline at 
the origin for absolute  and through the crossing point of the curves for differential.

get_abs_and_real_phases(data): returns a list of amplitudes and a list of phases
use: 
    
sd0 = process_signal(data0)
amp0, phase0 = get_abs_and_real_phases(sd0)
    
"""

import os
import sys
path = os.path.dirname(os.getcwd())
new_path = os.path.join(path, 'src')
sys.path.insert(0,new_path)
import cmath
import math

import pandas as pd
import numpy as np                       #visualisation
         #visualisation
     
#sns.set(color_codes=True)
from constants import FilePath,Channel
#from data.make_dataset import CompleteData
from pathlib import Path
import scipy.signal as signal
from scipy import stats



def new_phase(gradient):
    phase = cmath.phase(complex(1.0,gradient))
    return phase


def zero_crossings(x, y):
    ndx=np.diff(np.sign(y)) != 0
    if len(ndx) < len(y):
         ndx=np.append(ndx,False)
    
    return x[ndx]

#functions to find the fitting range by finding the positions of maxima and minima
    #if the data is so noisy that no peaks can be found it defaults to using the box center
def find_diff_trend_limit(data,dp):
   #assumes two peaks
    zc=zero_crossings(np.arange(0,data.shape[0]),data[dp[0]]-data[dp[1]])
   
    dx = signal.savgol_filter(data[dp[0]], 9, 2, deriv=1, delta=1.0, axis=0)
    maxx = data[dp[0]][zero_crossings(np.arange(0,len(dx)),dx)].idxmax()
    minx = data[dp[0]][zero_crossings(np.arange(0,len(dx)),dx)].idxmin()
    
    dy = signal.savgol_filter(data[dp[1]], 9, 2, deriv=1, delta=1.0, axis=0)
    miny = data[dp[1]][zero_crossings(np.arange(0,len(dy)),dy)].idxmin()
    maxy = data[dp[1]][zero_crossings(np.arange(0,len(dy)),dy)].idxmax()
    
    if (abs(maxx-minx)>4) or (abs(maxy-miny)>4):
         #index of crossing point of curves closest to the box center
         crossing_point = int(abs(zc-(data.shape[0]/2)).min()+data.shape[0]/2)
         if (crossing_point < int(data.shape[0]/6)) or (crossing_point > int(data.shape[0]*5/6)):
             crossing_point = int(data.shape[0]/2) 
    else:
         #index of crossing point of curves closest to the end
         crossing_point = int(abs(zc-(data.shape[0])).min()+data.shape[0])-3 
         
  
    return crossing_point, list([min(maxx,minx),max(maxx,minx),max(maxy,miny),min(maxy,miny)])
    
def find_abs_trend_limit(data, ap):
  
    #find largest maxima/minima
    dx = signal.savgol_filter(data[ap[0]], 9, 2, deriv=1, delta=1.0, axis=0)
    if data[ap[0]][zero_crossings(np.arange(0,len(dx)),dx)].max()> abs(data[ap[0]][zero_crossings(np.arange(0,len(dx)),dx)].min()):
        maxx = data[ap[0]][zero_crossings(np.arange(0,len(dx)),dx)].idxmax()
    else:
        maxx= data[ap[0]][zero_crossings(np.arange(0,len(dx)),dx)].idxmin()
   
    #if value is too close to the window limit replace by center
    if (maxx <= int(data.shape[0]/6)) or (maxx >= int(data.shape[0]*5/6)):
        maxx = int(data.shape[0]/2)
    
    
    dy = signal.savgol_filter(data[ap[1]], 9, 2, deriv=1, delta=1.0, axis=0)
    if abs(data[ap[1]][zero_crossings(np.arange(0,len(dy)),dy)].min()) > data[ap[1]][zero_crossings(np.arange(0,len(dy)),dy)].max():
        miny = data[ap[1]][zero_crossings(np.arange(0,len(dy)),dy)].idxmin()
    else:
        miny = data[ap[1]][zero_crossings(np.arange(0,len(dy)),dy)].idxmax() 
    
    if (miny <= int(data.shape[0]/6)) or (miny >= int(data.shape[0]*5/6)):
        miny = int(data.shape[0]/2)
    
    return int((maxx+miny)/2)-2, list([maxx,miny])


# Main Function to calculate the new amplitudes and phases. 
# !! This takes in the smoothed dataframe - that has only numerical columns (from process_signal function)    
#fit the liness
#calculate phase. 
def get_abs_and_real_phases(data):
    abs_pairs=[['X02','Y02'],['X04','Y04'],['X06','Y06'],['X08','Y08'],['X10','Y10'],
              ['X12','Y12'],['X14','Y14'],['X16','Y16'],['X18','Y18'],['X20','Y20']]
    diff_pairs=[['X01','Y01'],['X03','Y03'],['X05','Y05'],['X07','Y07'],['X09','Y09'],
              ['X11','Y11'],['X13','Y13'],['X15','Y15'],['X17','Y17'],['X19','Y19']]
    i=0
    phases=[]
    amplitudes=[]
    for dp in diff_pairs:
        #for the differential data -find Npoints to use from maximum and crossing point as origin of the trendline
        x0, minmax = find_diff_trend_limit(data, dp)
        delta = (abs(x0-(minmax[0]+minmax[3])/2)+abs(x0-(minmax[1]+minmax[2])/2))/2
        if delta < 1:
            delta=1
        x_low=max(0,x0-delta)   
      
        xset=data[dp[0]][(data.index>(x_low)) & (data.index<(x0+delta))]
        yset=data[dp[1]][(data.index>(x_low)) & (data.index<(x0+delta))]
        gradient, intercept, r_value, p_value, std_err = stats.linregress(xset,yset)
  
        phases.append(cmath.phase(complex(1.0,gradient)))
     
        x=(data[dp[0]][ data.index==minmax[0]])[minmax[0]]
        y=(data[dp[1]][ data.index==minmax[3]])[minmax[3]]
        amplitudes.append(math.sqrt(x**2 + y**2))                          
           
        #for the absolute data -find Npoints to use from maximum                    
        ap=abs_pairs[i]
        i=i+1
        npoints, minmax_abs = find_abs_trend_limit(data, ap)
        
        xset=data[ap[0]][(data.index<max(npoints,2))]
        yset=data[ap[1]][(data.index<max(npoints,2))]   
        gradient, intercept, r_value, p_value, std_err = stats.linregress(xset,yset)
        phases.append(cmath.phase(complex(1.0,gradient)))
        
        x=(data[ap[0]][ data.index==minmax_abs[0]])[minmax_abs[0]]
        y=(data[ap[1]][ data.index==minmax_abs[1]])[minmax_abs[1]]
        amplitudes.append(math.sqrt(x**2+y**2)) 
    return amplitudes, phases