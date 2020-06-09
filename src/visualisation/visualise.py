import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

import features.build_features as build_features


def plot_ec_waveforms(df):
    '''Plots the real and imag parts of EC waveforms on a single plot
    
    The input df is a dataframe that contains a series of EC waveforms.
    
    The waveforms are arranged in columns as X1,Y1,X2,Y2 and so on...
    '''
    
    # Number of EC waveforms to be plotted
    num_plots = int(len(df.columns)/2)

    fig = plt.figure()  
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    
    # Initialize a color map for plotting
    color = iter(cm.rainbow(np.linspace(0, 1, num_plots)))
    
    for i in range(num_plots):
        # Loop over i 'num_plots' times and generate a plot for each waveform
        plot_real = df.iloc[:, (i*2)-1]
        plot_imag = df.iloc[:, (i*2)]
        plot_label = df.columns[i*2] 
        
        c = next(color)
        plt.plot(plot_real, plot_imag,'-ok', label = plot_label, c = c)
        plt.legend(bbox_to_anchor = (1.04, 1), loc = "upper left")
        plt.tight_layout(rect = [0, 0, 2, 2])
        
        
        
def plot_origin_and_furthest_points(real_col, imag_col):
    '''
    Plots waveforms, with X's corresponding to the starting point and the furthest point
    
    (as extracted by the define_starting_point and find_furthest_point functions in build_features.py).   
    '''
    starting_point = build_features.define_starting_point(real_col, imag_col)
    furthest_point = build_features.find_furthest_point(real_col, imag_col, starting_point)
    
    fig, ax = plt.subplots()
    ax.plot(real_col, imag_col)
    ax.plot(starting_point[0], starting_point[1], marker="x", markersize=30)
    ax.plot(furthest_point[0], furthest_point[1], marker="x", markersize=30)
    ax.set(xlabel='real', ylabel='imaginary',
            title='Plot with starting and ending points')
    ax.grid()

    plt.show()

