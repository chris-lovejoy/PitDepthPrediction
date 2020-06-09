# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:15:58 2020

Module to 

1)subtract background and entry peaks of raw Tube data base on a 
polynomial savgol_filter. Saves clipped tube data and derivatives. 

2) auto-detect the set of flaws in the training tubes based on superposition 
and low-pass filtering of the signals. 
This function assumes 9 Flaws with decreasing depth

use as in notebook XXX 

@author: gmueller
"""

import os
import sys
import glob

import pandas as pd
import numpy as np 
#from sklearn import preprocessing
from scipy import stats
import scipy as sp
import math
import scipy.signal as signal

from constants import FilePath, Channel

#function to detect the entry peaks and plateaus of the tube data
def plateaus_bounds(input_signal, channels, interval=10):
    
    ax_length=input_signal.shape[0]
    halfax_length=int(ax_length/2)
    
    low_bound = 0
    high_bound = ax_length-1
    low_ave=0
    high_ave=0
    
    for ch in channels:
        imin=input_signal[ch].idxmin()
        imax=input_signal[ch].idxmax()
        if (imax > imin):
            if (imax > halfax_length):
                high_ave=high_ave+imax
            if (imin < halfax_length):
                low_ave = low_ave + imin
        else:
            if (imin > halfax_length):
                high_ave=high_ave+imin
            if (imax < halfax_length):
                low_ave = low_ave+imax
                
    low_ave=float(low_ave)/len(channels)  
    high_ave=float(high_ave)/len(channels)
    
    low_lim=low_ave + float(halfax_length-low_ave)/2
    high_lim=high_ave - float(-halfax_length+high_ave)/2
    
    for ch in channels:
        imin=input_signal[ch].idxmin()
        imax=input_signal[ch].idxmax()
        if (imax > imin):
            if (imax > high_lim)& (imax < high_bound):
                high_bound=imax
            if (imin > low_bound) & (imin < low_lim):
                low_bound = imin
        else:
            if (imin > high_lim)& (imin < high_bound):
                high_bound=imin
            if (imax > low_bound) & (imax < low_lim):
                low_bound = imax
    
    return [low_bound+interval, high_bound-interval]



#function to remove the areas outside the detected limits from the dataframe
def clip_signal(input_signal, channels=None):
    if channels==None:
        channels = Channel.ALL
    bounds=plateaus_bounds(input_signal,channels)
    sig = input_signal[bounds[0]:bounds[1]].reset_index()
    return sig


#function to subtract a low order polynomial fit all channels separately
def subtract_background (input_signal, window1=401, window2=201 ):
    #fit background by low order polynomial and subtract
    channels = Channel.ALL
    
    bg_list=[]
    for ch in channels:    
        backgr1=signal.savgol_filter(input_signal[ch], 401, 2, deriv=0, delta=1.0, axis=-1)
        backgr2=signal.savgol_filter(input_signal[ch], 201, 2, deriv=0, delta=1.0, axis=-1)
        bg_list.append(((backgr1+backgr2)/2).tolist())
    
    bg=pd.DataFrame(bg_list).transpose()
    bg.columns=channels
    sig=input_signal[channels].sub(bg, axis='columns')
    
    return sig


#calculates the derivatives from savgol filter
def get_derivatives(input_signal):
    channels = Channel.ALL
    der_list=[]
    my_headers=[]
    for ch in channels:
        my_headers.append(ch+'_d1')
        my_headers.append(ch+'_d2')
        der_list.append(signal.savgol_filter(input_signal[ch], 11, 4, deriv=1, delta=1.0, axis=-1))
        der_list.append(signal.savgol_filter(input_signal[ch], 11, 4, deriv=2, delta=1.0, axis=-1))
    derivatives=pd.DataFrame(der_list).transpose() 
    derivatives.columns=my_headers
    
    return derivatives


"""###############################
 Now the functions to detect peaks in the pre-filtered data
"""

#for cutting out peaks from the whole signal 
def clip_signal_to_peaks(input_signal, peak_widths):
    peak_segments=[]
       
    for i in range(0,len(peak_widths[2])):
        bound_l=int(peak_widths[2][i])-4
        bound_u=int(peak_widths[3][i])+5
        peak_segments.append(input_signal[bound_l:bound_u])
    return peak_segments   


#superimpose the signals from all channels to facilitate peak detection
def sum_up_channels(input_data): 
    
    #Function for extracting the peak locations of the bg removed curves
    abs_x_channels = ['X02','X04','X06','X08','X10','X12','X14','X16','X18','X20']
    
    abs_y_channels = ['Y02','Y04','Y06','Y08','Y10','Y12','Y14','Y16','Y18','Y20']

    diff_x_channels = ['X01','X03','X05','X07','X09','X11','X13','X15','X17','X19']

    diff_y_channels = ['Y01','Y03','Y05','Y07','Y09','Y11','Y13','Y15','Y17','Y19']

    #add all channels of the same type:
    A01_xa=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))
    A01_xd=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))
    A01_ya=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))
    A01_yd=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))

    Sum_all=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))
    Sum_all_abs=pd.DataFrame(np.zeros(input_data['X04'].shape[0]))

    for ch in abs_x_channels:
        A01_xa=A01_xd.add(input_data[ch],axis=0)
        Sum_all=Sum_all.add(input_data[ch],axis=0)
        Sum_all_abs=Sum_all_abs.add(abs(input_data[ch]),axis=0)
    for ch in diff_x_channels:
        A01_xd=A01_xd.add(input_data[ch],axis=0)
        Sum_all=Sum_all.add(input_data[ch],axis=0)
        Sum_all_abs=Sum_all_abs.add(abs(input_data[ch]),axis=0)
    for ch in abs_y_channels:
        A01_ya=A01_ya.add(input_data[ch],axis=0)
        Sum_all=Sum_all.add(input_data[ch],axis=0)
        Sum_all_abs=Sum_all_abs.add(abs(input_data[ch]),axis=0)
    for ch in diff_y_channels:
        A01_yd=A01_yd.add(input_data[ch],axis=0)
        Sum_all=Sum_all.add(input_data[ch],axis=0)
        Sum_all_abs=Sum_all_abs.add(abs(input_data[ch]),axis=0)
    
    summed_channels = pd.concat([A01_xd, A01_yd,A01_xa, A01_ya,Sum_all, Sum_all_abs], axis=1, sort=False)
    summed_channels.columns=['X_diff','Y_diff','X_abs','Y_abs','Sum','Sum_Abs']
    
    return summed_channels

#for eliminationg extra peaks, check the average peak spacing 
def check_peak_spacing(my_peaks):
    ave_dist =0
    std=0
    dist=[]
    for i in range(1,my_peaks.shape[0]):
        ave_dist=ave_dist+(my_peaks.iat[i,0]-my_peaks.iat[i-1,0])
        dist.append(my_peaks.iat[i,0]-my_peaks.iat[i-1,0])
    ave_dist=int(float(ave_dist)/my_peaks.shape[0]) 

    for i in range(1,my_peaks.shape[0]):
        std=std + ((my_peaks.iat[i,0]-my_peaks.iat[i-1,0])**2-ave_dist**2)
    std=math.sqrt(std)/my_peaks.shape[0]
            
    return dist, ave_dist, std
      

#remove peaks that have distences much below the average from their neighbors
def remove_intermediate_peaks(my_peaks, ind, inda, ave_dist, std, Number):
    drop_list=[]
    num=my_peaks.shape[0]
    #remove intermediate peaks
    for i in ind:
        #the low distance is to the right - check if the next distance is also short
        # if so check distance to 2nd neigh and if not > ave_dist+x std remove the intermediate peak
        #print(i,ind)
        if (i<=num-2) & (i+1 in ind):
            if(abs(my_peaks.iat[i+2,0]-my_peaks.iat[i,0]-ave_dist)) <2*std :
                drop_list.append(i+1)  
            elif (i==0) & (not i+1 in ind):
                drop_list.append[i]      

    for i in inda:
        if (i==0) & (my_peaks.iat[0,1] < 2.0/3*my_peaks.iat[1,1]): 
            drop_list.append(i) 
        if(i == num-2):
            drop_list.append(i+1)

    if (num-len(drop_list)>=Number):
        my_peaks=my_peaks.drop(index=my_peaks.index[drop_list])
        num=num - len(drop_list)
    else:
        #print(drop_list)
        #eliminate starting with largest deviation from mean distances
        while my_peaks.shape[0] > Number: 
            max_dev = 0
            max_ndx =0
            for i in range(len(drop_list)-1): 
                if abs(abs(drop_list[i+1]-drop_list[i])-ave_dist) > max_dev:
                    max_dev= abs(abs(drop_list[i+1]-drop_list[i])-ave_dist)
                    max_ndx=drop_list[i]
            my_peaks=my_peaks.drop(index=my_peaks.index[max_ndx])
            
    return my_peaks



#the main function to be called on the data
#input_data should have the background subtraced already and large entry peaks removed.  

def extract_peak_locations(input_data, filename):
    #filename ...for error codes. 
     
    summed_channels = sum_up_channels(input_data)
        
    #apply a low pass filter to the summed signals
    b, a = signal.butter(2, 0.03)
    #sig = summed_channels.Sum_Abs

    sig_ff = signal.filtfilt(b, a, summed_channels.Sum_Abs) 
   
    #and detect signals
    peaks, prop=signal.find_peaks((sig_ff), width=[20,200], distance=50)

    #collect peak properties in a dataframe
    my_peaks=pd.DataFrame(prop)
    my_peaks.insert(0,'ind',peaks)

    if my_peaks.shape[0] < 10:
        print ('few peaks detected - there may be an error {}'.format(filename)) #eventially also filename

    gradient, intercept, r_value, p_value, std_err = stats.linregress(my_peaks.ind,my_peaks.prominences)
    #shift to include small peaks at the end
    intercept=intercept-500
    my_high_peaks=my_peaks[my_peaks.prominences > gradient*my_peaks.ind+intercept ]

    while (my_high_peaks.shape[0]<9) & (intercept > 0):
        intercept=intercept-500
        my_high_peaks=my_peaks[my_peaks.prominences > gradient*my_peaks.ind+intercept ]

    if my_high_peaks.shape[0]<9:
        print('error too few peaks in {}'.format(filename))
     
    # check spacing between peaks and eliminate the rest of intermediate peaks:
    dist, ave_dist, std = check_peak_spacing(my_high_peaks) 
    dista=np.array(dist)

    # If given element doesn't exist in the array 
    #then it will return an empty array 
    #d=dista[dista<(ave_dist-0.5 *std)] 
    #da =dista[dista>(ave_dist+ 1.5*std)]
    ind=np.where(dista< ave_dist-0.5 *std)
    inda=np.where(dista> ave_dist+1.5*std)
    
    if len(my_high_peaks)>9:
        #print(len(my_high_peaks))
        my_high_peaks = remove_intermediate_peaks(my_high_peaks, ind[0], inda[0],ave_dist, std, 9)
        if my_high_peaks is None:
            print('error intermediate peak removal failed{}'.format(filename))
            return None
    
    #if all peaks are evenly spaced and still more than 9
    num=my_high_peaks.shape[0]
    while num > 9: 
        if my_high_peaks.prominences.iat[0]< my_high_peaks.prominences.iat[-1]:
            my_high_peaks=my_high_peaks.drop(index=my_high_peaks.index[[0]])
        elif my_high_peaks.prominences.iat[0] < 0.5*my_high_peaks.prominences.iat[1] :
             my_high_peaks=my_high_peaks.drop(index=my_high_peaks.index[[0]])
        else:
            my_high_peaks=my_high_peaks.drop(index=my_high_peaks.index[[num-1]])
        num=my_high_peaks.shape[0]
        
    widths=signal.peak_widths(sig_ff, my_high_peaks.ind, rel_height=0.7)
    
    return my_high_peaks, widths
