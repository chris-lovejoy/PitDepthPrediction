import cmath
import math
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
from time import sleep
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import  mutual_info_regression
sys.path.insert(0,os.path.dirname(os.getcwd()))
import data.make_dataset as make_dataset
from constants import FilePath, Tube, Channel
from features.new_amplitude_and_phase import get_abs_and_real_phases    
	
class FeatureData:
    def __init__(self, name):
    # expects name to be 'filtered_cg' or 'filtered' or 'filtered_ag' or 'test'
        self.name = name
        self.path, self.save_dir = self.get_directories()
        self.full_feature_data = self.combine_feature_and_flaw_info()
    
    def get_directories(self):
        # determine base directory
        p = 0
        while not os.path.exists(os.path.join(Path(os.getcwd()).parents[p], 'src')):
            p += 1
            if p > 6:
                print('no src folder on tree')
                break
        
        path_dict = {
            'filtered': os.path.join(Path(os.getcwd()).parents[p], FilePath.FILTERED_FILE),
            'filtered_cg': os.path.join(Path(os.getcwd()).parents[p], FilePath.FILTERED_CG_FILE),
            'test': os.path.join(Path(os.getcwd()).parents[p], FilePath.TEST_FILE)
            }
        
        # path to file
        path = path_dict[self.name]
        
        save_dict = {
            'filtered': 'full_feature',
            'filtered_cg': 'full_feature_cg',
            'test':  'full_feature_test'
            }
        
        # path to directory to save csv
        save_dir = os.path.join(Path(os.getcwd()).parents[p],FilePath.INTERIM_FOLDER, '{}_data.csv'.format(save_dict[self.name]))         
        return path, save_dir
              
    def get_data(self):
        # selects the appropriate data file
        data = pd.read_csv(self.path)
        data = data.drop(columns=['Mode', 'Tube_SN'])       
        return data
    
    def get_amp_phase_frame(self, smoothed_data):
        print('Building amplitude and phase features...')
        sleep(2)
        # calculates amplitude and phase for each tube/flaw/angle
        amp, phase = map(list,zip(*[get_abs_and_real_phases(g.iloc[:,-40:].reset_index(drop=True)) for k,g in tqdm(smoothed_data.groupby(['Tube_Alias', 'Flaw_ID', 'Angle']))]))
        sleep(2)
        print('Calculations for amplitude and phase complete. Building dataframe...')
        
        #builds a dataframe of the id, amp, phase data
        id_df = pd.DataFrame([k for k,g in smoothed_data.groupby(['Tube_Alias', 'Flaw_ID', 'Angle'])])
        amp_df = pd.DataFrame(amp)
        phase_df = pd.DataFrame(phase)
        full_df = pd.merge(pd.merge(id_df, amp_df, left_index=True, right_index=True), phase_df, left_index=True, right_index=True)
        
        # create header for the data frame
        header = ['Tube_Alias', 'Flaw_ID', 'Angle']        
        amp_header = ['Amp' + '_' + str(i) for i in range(1,21)]
        phase_header = ['Phase' + '_' + str(i) for i in range(1,21)]

        # create the full header
        full_header = header + amp_header + phase_header  
        
        # set full_header as the column names
        full_df.columns = full_header
        return full_df
    
    def calc_ab(self, real, imag):
        a = max(real.values) - min(real.values)
        b = max(imag.values) - min(imag.values)

        return a, b
    
    def get_ratio_features(self, df):
    # calculates the max-min ratio for each channel of every
    # tube/flaw/angle combination
    
        # creates a list of pairs of X/Y columns to feed into calc_tan_ab
        cols = list(df.columns[-40:])
        paired_cols = []
        i = 0
        while i<len(cols):
          paired_cols.append(cols[i:i+2])
          i += 2
        
        # empty lists to store the values
        a_list = []
        b_list = []
        
        # loop through each pair of X/Y and append the data to the  list
        for j in range(0,20):
            real = df[paired_cols[j][0]]
            imag = df[paired_cols[j][1]]
            a, b = self.calc_ab(real, imag)
            a_list.append(a)
            b_list.append(b)
        
        # returns list of ratio values
        return a_list, b_list
    
    def create_ab_frame(self, df):
        print('Calculating A and B features....')
        sleep(2)
        a_list, b_list = map(list,zip(*[self.get_ratio_features(g) for k,g in tqdm(df.groupby(['Tube_Alias', 'Flaw_ID', 'Angle']))]))
        sleep(2)
        print('Ratio features done!')
        
        # create header for the data frame
        a_header = ['A_Value' + '_' + str(i) for i in range(1,21)]
        b_header = ['B_Value' + '_' + str(i) for i in range(1,21)]  
        header = a_header + b_header
        
        # create dataframes for A and B
        df_a = pd.DataFrame(a_list, columns=a_header)
        df_b = pd.DataFrame(b_list, columns=b_header)
        new_df = df_a.join(df_b)
        
        return new_df
    
    def get_ab_ratio(self,df):
        # calculates arc tan of A/b
        data = self.create_ab_frame(df)
        print('Calculating A/B...')
        sleep(2)
        for i in tqdm(range(1,21)):
            data['AB_Ratio_{}'.format(i)] = data['A_Value_{}'.format(i)].values / data['B_Value_{}'.format(i)].values
        
        return data
            
    def add_flaw_info(self, flaw_id, tube_alias):
        # get the info on each flaw characteristic
        depth = [Tube.FLAW_DEPTH[x][y] for x,y in zip(flaw_id, tube_alias)]
        pct_depth = [round(Tube.FLAW_DEPTH[x][y] / Tube.WALL_THICKNESS[y] * 100, 1) for x,y in zip(flaw_id, tube_alias)]
        volume = [Tube.VOLUME_LOSS[x][y] for x,y in zip(flaw_id, tube_alias)]
        area = [Tube.FLAW_AREA[y] for y in tube_alias] 
        
        # returns lists of flaw info
        return depth, pct_depth, volume, area

    def combine_feature_and_flaw_info(self):        
        print('Starting data processing for {}_data.csv...'.format(self.name))
        smoothed_data = self.get_data()
        amp_phase_data = self.get_amp_phase_frame(smoothed_data)
        ratio_features = self.get_ab_ratio(smoothed_data)
        full_frame = amp_phase_data.join(ratio_features)
        
        # add the flaw info to the amplitude/phase dataframe    
        if self.name != 'test':
            full_frame['Flaw_Depth'], full_frame['Pct_Depth'], full_frame['Flaw_Volume'], full_frame['Flaw_Area'] = self.add_flaw_info(full_frame['Flaw_ID'], full_frame['Tube_Alias'])      
        sleep(2)
        print('Dataframe complete. Saving dataframes as .csv files.')

        # save the data as a csv
        full_frame.to_csv(self.save_dir,index=False)
        
        # returns the complete dataframe
        return full_frame

#redefine this function outside the features class, so it can be used without creating a features object 
def add_flaw_info_to_df(flaw_id, tube_alias):
        # get the info on each flaw characteristic
        depth = [Tube.FLAW_DEPTH[x][y] for x,y in zip(flaw_id, tube_alias)]
        pct_depth = [round(Tube.FLAW_DEPTH[x][y] / Tube.WALL_THICKNESS[y] * 100, 1) for x,y in zip(flaw_id, tube_alias)]
        volume = [Tube.VOLUME_LOSS[x][y] for x,y in zip(flaw_id, tube_alias)]
        area = [Tube.FLAW_AREA[y] for y in tube_alias] 
        
        return depth, pct_depth, volume, area        
    
      
    
def extract_PCA_components(dataframe, features_for_components,
                           num_components=2, comp_names=""):
    """
    This function takes in a dataframe of features and creates principal
    components across all of them. 
    
    The 'features_for_components' are the columns which will be included.
    
    It returns the components themselves as a dataframe, as well as a 
    dataframe of the amount of variance explained by each component.
    """
    
    # probably take in: dataframe with the appropraite columns, the column names to become components,
 #       number of components
 # returns a dataframe of the components
         # (in a format that can then be easily combined with other functions)
    
    selected_dataframe = dataframe.loc[:, features_for_components]
    scaled_dataframe = StandardScaler().fit_transform(selected_dataframe)
    
    pca = PCA(n_components = num_components)

    columns = []
    for i in range(1, num_components+1):
        columns.append(f'{comp_names}Comp{i}')
    
    components = pd.DataFrame(pca.fit_transform(scaled_dataframe), columns=columns)

    explained_variance = pd.DataFrame([pca.explained_variance_ratio_], columns=columns)
    
    print("The explained variance (as a ratio) for the", num_components,
          f"principle components (using {comp_names} if specified) are as follows:\n",
          pca.explained_variance_ratio_)

    return components, explained_variance


# =====================================================================
#functions that calculate the mutual information across a range of different frequencies,
#as used by the add_mutual_information() function.

#normalized by the maximum mutual information
def calc_frequency_norm_mutual_info(data, channels): 

    DIFF_REAL=np.intersect1d(Channel.DIFF, Channel.REAL)
    DIFF_IMAG=np.intersect1d(Channel.DIFF, Channel.IMAGINARY)
    ABS_REAL=np.intersect1d(Channel.ABS, Channel.REAL)
    ABS_IMAG=np.intersect1d(Channel.ABS, Channel.IMAGINARY)

    x_dr = data[DIFF_REAL].to_numpy()
    x_di = data[DIFF_IMAG].to_numpy()
    x_ar = data[ABS_REAL].to_numpy()
    x_ai = data[ABS_IMAG].to_numpy()   

    mi_dr = []
    mi_di = []
    mi_ar = []
    mi_ai = []

    for ch in channels:

        y = data[ch].to_numpy().reshape(-1,)

        mi=mutual_info_regression(x_dr, y)
        mi_dr.append(mi/np.max(mi))
        mi=mutual_info_regression(x_di, y)
        mi_di.append(mi/np.max(mi))
        mi=mutual_info_regression(x_ar, y)
        mi_ar.append(mi/np.max(mi))
        mi=mutual_info_regression(x_ai, y)
        mi_ai.append(mi/np.max(mi))

    #now we can do some averaging over the high frequency range
    mi_dr_av=[]
    mi_di_av=[]
    mi_ar_av=[]
    mi_ai_av=[]
    for mi in mi_dr:
        mi_dr_av.append(np.average(mi[0:4,]))

    mi_di_av=[]
    for mi in mi_di:
        mi_di_av.append(np.average(mi[0:4,]))

    mi_ar_av=[]
    for mi in mi_ar:
        mi_ar_av.append(np.average(mi[0:4,]))

    mi_ai_av=[]
    for mi in mi_ai:
        mi_ai_av.append(np.average(mi[0:4,]))

    mutual_info = np.concatenate((mi_dr_av, mi_di_av, mi_ar_av, mi_ai_av))

    return mutual_info


#absolute
def calc_frequency_mutual_info(data, channels): 

    DIFF_REAL=np.intersect1d(Channel.DIFF, Channel.REAL)
    DIFF_IMAG=np.intersect1d(Channel.DIFF, Channel.IMAGINARY)
    ABS_REAL=np.intersect1d(Channel.ABS, Channel.REAL)
    ABS_IMAG=np.intersect1d(Channel.ABS, Channel.IMAGINARY)

    x_dr = data[DIFF_REAL].to_numpy()
    x_di = data[DIFF_IMAG].to_numpy()
    x_ar = data[ABS_REAL].to_numpy()
    x_ai = data[ABS_IMAG].to_numpy()   

    mi_dr = []
    mi_di = []
    mi_ar = []
    mi_ai = []

    for ch in channels:

        y = data[ch].to_numpy().reshape(-1,)

        mi_dr.append(mutual_info_regression(x_dr, y))
        mi_di.append(mutual_info_regression(x_di, y))
        mi_ar.append(mutual_info_regression(x_ar, y))
        mi_ai.append(mutual_info_regression(x_ai, y))

    #now we can do some averaging over the high frequency range
    mi_dr_av=[]
    mi_di_av=[]
    mi_ar_av=[]
    mi_ai_av=[]
    for mi in mi_dr:
        mi_dr_av.append(np.average(mi[0:4,]))

    mi_di_av=[]
    for mi in mi_di:
        mi_di_av.append(np.average(mi[0:4,]))

    mi_ar_av=[]
    for mi in mi_ar:
        mi_ar_av.append(np.average(mi[0:4,]))

    mi_ai_av=[]
    for mi in mi_ai:
        mi_ai_av.append(np.average(mi[0:4,]))

    mutual_info = np.concatenate((mi_dr_av, mi_di_av, mi_ar_av, mi_ai_av))

    return mutual_info

#using the low instead of high frequency range:

def calc_lowf_norm_mutual_info(data, channels): 

    DIFF_REAL=np.intersect1d(Channel.DIFF, Channel.REAL)
    DIFF_IMAG=np.intersect1d(Channel.DIFF, Channel.IMAGINARY)
    ABS_REAL=np.intersect1d(Channel.ABS, Channel.REAL)
    ABS_IMAG=np.intersect1d(Channel.ABS, Channel.IMAGINARY)

    x_dr = data[DIFF_REAL].to_numpy()
    x_di = data[DIFF_IMAG].to_numpy()
    x_ar = data[ABS_REAL].to_numpy()
    x_ai = data[ABS_IMAG].to_numpy()   

    mi_dr = []
    mi_di = []
    mi_ar = []
    mi_ai = []

    for ch in channels:

        y = data[ch].to_numpy().reshape(-1,)

        mi=mutual_info_regression(x_dr, y)
        mi_dr.append(mi/np.max(mi))
        mi=mutual_info_regression(x_di, y)
        mi_di.append(mi/np.max(mi))
        mi=mutual_info_regression(x_ar, y)
        mi_ar.append(mi/np.max(mi))
        mi=mutual_info_regression(x_ai, y)
        mi_ai.append(mi/np.max(mi))

    #now we can do some averaging over the high frequency range
    mi_dr_av=[]
    mi_di_av=[]
    mi_ar_av=[]
    mi_ai_av=[]
    for mi in mi_dr:
        mi_dr_av.append(np.average(mi[6:,]))

    mi_di_av=[]
    for mi in mi_di:
        mi_di_av.append(np.average(mi[6:,]))

    mi_ar_av=[]
    for mi in mi_ar:
        mi_ar_av.append(np.average(mi[6:,]))

    mi_ai_av=[]
    for mi in mi_ai:
        mi_ai_av.append(np.average(mi[6:,]))

    mutual_info = np.concatenate((mi_dr_av, mi_di_av, mi_ar_av, mi_ai_av))

    return mutual_info

def calc_lowf_mutual_info(data, channels): 


    x_dr = data[DIFF_REAL].to_numpy()
    x_di = data[DIFF_IMAG].to_numpy()
    x_ar = data[ABS_REAL].to_numpy()
    x_ai = data[ABS_IMAG].to_numpy()   

    mi_dr = []
    mi_di = []
    mi_ar = []
    mi_ai = []

    for ch in channels:

        y = data[ch].to_numpy().reshape(-1,)

        mi_dr.append(mutual_info_regression(x_dr, y))
        mi_di.append(mutual_info_regression(x_di, y))
        mi_ar.append(mutual_info_regression(x_ar, y))
        mi_ai.append(mutual_info_regression(x_ai, y))

    #now we can do some averaging over the high frequency range
    mi_dr_av=[]
    mi_di_av=[]
    mi_ar_av=[]
    mi_ai_av=[]
    for mi in mi_dr:
        mi_dr_av.append(np.average(mi[6:,]))

    mi_di_av=[]
    for mi in mi_di:
        mi_di_av.append(np.average(mi[6:,]))

    mi_ar_av=[]
    for mi in mi_ar:
        mi_ar_av.append(np.average(mi[6:,]))

    mi_ai_av=[]
    for mi in mi_ai:
        mi_ai_av.append(np.average(mi[6:,]))

    mutual_info = np.concatenate((mi_dr_av, mi_di_av, mi_ar_av, mi_ai_av))

    return mutual_info

