import glob
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.getcwd()))
from constants import FilePath, Channel, Tube
from features.process_data import process_signal
from time import sleep
from tqdm import tqdm

# create a class that includes all data and facilitates selection
# of subsets
class CompleteData:
    # expects name to be 'raw', 'filtered', or 'filtered_cg' or test
    def __init__(self, name):
        self.name = name
        self.path, self.save_dir = self.get_directories()
        self.full_data_frame = self.rearrange_columns()
   
    def get_directories(self):
        # determine base directory
        p = 0
        while not os.path.exists(os.path.join(Path(os.getcwd()).parents[p], 'src')):
            p += 1
            if p > 6:
                print('no src folder on tree')
                break
        
        path_dict = {
            'raw': os.path.join(Path(os.getcwd()).parents[p], FilePath.RAW_FOLDER),
            'filtered': os.path.join(Path(os.getcwd()).parents[p], FilePath.FILTERED_FOLDER),
            'filtered_cg': os.path.join(Path(os.getcwd()).parents[p], FilePath.FILTERED_CG_FOLDER),
            'test': os.path.join(Path(os.getcwd()).parents[p], FilePath.TEST_FOLDER)
            }
         
        path = path_dict[self.name]        
        save_dir = os.path.join(Path(os.getcwd()).parents[p], FilePath.FULL_RAW_FOLDER, '{}_data.csv'.format(self.name))
        return path, save_dir
 
    def get_all_files_with_path(self, path):
        # returns all the file names from the specified directory
        all_files = glob.glob(os.path.join(path, '*.csv'))
        return  all_files
    
    def get_tube_info(self, file, path):
        # gets tube ID info the file name
        from_name = os.path.basename(file).split('_')
        if self.name != 'test':
            alias = from_name[0]
            sn = from_name[1]
            flaw = from_name[2]
            angle = from_name[3]
        else:
            alias = from_name[1]
            angle = -1
            sn = '000'
            flaw = 'Q'
        mode = self.name
        return alias, sn, flaw, angle, mode
        
    def get_probe_data(self, file): 
        # gets the actual channel data from each csv
        # and stores it as a dataframe
        if self.name == 'filtered_cg':
            probe_data = process_signal(pd.read_csv(file ,dtype=np.float32))
        else:
            probe_data = process_signal(pd.read_csv(file,header=[5], dtype=np.float32))
        return probe_data 
    
    def build_full_data_frame(self):
        # this is the driver function that builds a dataframe for
        # each file and appends that to a complete dataframe

        # set the path to all the files                
        all_files = self.get_all_files_with_path(self.path)
        
        # empty list that each file's dataframe will be appended to
        full_data_frame = []
    
        # loop through all the files and run each dataframe-building function
        # appends each dataframe to the full_data_frame list
        print('Building dataframe from csv files...')
        sleep(2)
        for file in tqdm(all_files):
            probe_data = self.get_probe_data(file)
            probe_data['Tube_Alias'],probe_data['Tube_SN'],probe_data['Flaw_ID'], probe_data['Angle'], probe_data['Mode'] = self.get_tube_info(file, self.path)
            full_data_frame.append(probe_data)
            
        # concat the list of dataframes into one final dataframe
        full_data_frame = pd.concat(full_data_frame,axis=0)
        full_data_frame['Angle'] = pd.to_numeric(full_data_frame['Angle'])
        
        return full_data_frame
    
    def rearrange_columns(self):
        # simply rearranges columns so tube info is at front
        df = self.build_full_data_frame()
        cols = list(df.columns)
        cols_to_move = ['Mode', 'Angle', 'Flaw_ID','Tube_SN','Tube_Alias']
        for col in cols_to_move:
            cols.insert(0, cols.pop(cols.index(col)))
        df = df.loc[:,cols]
        
        # save the data as a csv

        print('Saving {} data....'.format(self.name))
        df.to_csv((self.save_dir), index=False)
        
        return df
        

# updated version of the import_waveforms function
def get_waveform(dataframe, data_type=None, tube_list=None, flaw_list=None, angle_list=None, diff_abs=None, real_img=None, freq_list=None):
    # returns a dataframe with a subset of data based on the critera above
    # one can select for tube alias, flaw, and angle, as well as any of the frequencies
    # requires df_name to be either "raw" or "filtered"
    # each other variable must be a list of strings.
    # pass empty lists if all values are desired
    
    df = dataframe
    
    # dictionary that selects the desired dataframe
    
    df_dict = {
        'raw': df.loc[df['Mode'] == 'Raw'],
        'filtered': df.loc[df['Mode'] == 'Filtered']
        }

    # create a dictionary of input tables for tube ID info
    parameter_dict = {
        'Tube_Alias': tube_list, 
        'Flaw_ID': flaw_list, 
        'Angle':angle_list
        }
    # dictionary for channel parameters
    channel_dict = {
        'abs': Channel.ABS,
        'diff': Channel.DIFF,
        'real': Channel.REAL,
        'imaginary': Channel.IMAGINARY
        }
    # list of channel parameter inputs
    channel_list = [diff_abs, real_img]
    
    if data_type == None:
        selected_data = df.copy()            
    else:
        selected_data = df_dict[data_type]
    
    # check to see if any parameter list (i.e., tube alias, flaw, angle)
    # is empty; if so, it uses all values
    for key,value in parameter_dict.items():
        if value == None:
            parameter_dict[key] = list(selected_data[key].unique())
        else:
            value = value
    
    # narrows down the data to the desired tube/flaw/angle info
    waveform_data = selected_data.loc[(selected_data['Tube_Alias'].isin(parameter_dict['Tube_Alias'])) 
                                  & (selected_data['Flaw_ID'].isin(parameter_dict['Flaw_ID'])) 
                                  & (selected_data['Angle'].isin(parameter_dict['Angle']))]
    
    # checks channel parameters (i.e., abs/diff and real/imaginary)
    # if empty, uses all values; else chooses the ones declared
    ID_list = list(parameter_dict.keys()) + ['Mode']
    for param in channel_list:
        if param == None:
            pass
        else:
            param_list = channel_dict[param]                
            param_keep_list = ID_list + param_list
            waveform_data = waveform_data[waveform_data.columns.intersection(param_keep_list)]
    
    # checks the frequency parameters.if none are chosen, all are returned
    if freq_list == None:
        pass
    else:
        frequencies_used = []
        for frequency in freq_list:
            frequencies_used += Channel.FREQUENCY[frequency]
        freq_keep_list = ID_list + frequencies_used
        waveform_data = waveform_data[waveform_data.columns.intersection(freq_keep_list)]
        waveform_data.reset_index(inplace=True,drop=True)
    return waveform_data
    
    
#### ====== LEGACY CODE - disused functions, kept for reference ====== ####

def import_waveforms(original_data, tube_list=[], flaw_list=[],angle_list=[],frequency_list=[] ):
    '''  
    Data Extraction Functions:
    (Task2: Make a function that can conveniently extract the kind of data we want from the excel 
    files which Thiago provided)
    functions that use original_data from the Complete_data class and return a list of dataframes  
    with a subsets of data that we want 
    it returns a list of frames rather than combined data as I think that is easier for plotting
    combined data is easy to get with pandas functions
    default returns all
    ''' 
    #initializing default values
    waveforms = []
    if(len(tube_list)==0):
        tube_list=['AP01','AP02','AP03','AP04','AP05','CP01','CP02','CP03','CP04','CP05','RP01',
                    'RP02','RP03','RP04','RP05','RP06','WT01','WT02','WT03','WT04','WT05']
   
    if(len(flaw_list)==0):
        flaw_list=['A','C','B','E','D','F','G','H','I']
        
    if(len(angle_list)==0):
        angle_list=Tube.Angles
    if(len(frequency_list)!=0):
        frequency_list=['Alias','SN','Flaw','Angle','Mode']+frequency_list
       
        print('importing data - for large sets it will take a few minutes')
    for tube in tube_list:
        for flaw in flaw_list:
            for angle in angle_list:
                if (len(frequency_list)==0):
                    sub_data=original_data[original_data.Angle.isin([angle]) & original_data.Flaw.isin([flaw]) &
                                            original_data.Alias.isin([tube])]
                else:
                    sub_data=original_data[original_data.Angle.isin([angle]) & original_data.Flaw.isin([flaw]) &
                                            original_data.Alias.isin([tube])] 
                    sub_data=sub_data[frequency_list]
          
                if sub_data.shape[0]>0:
                    waveforms.append(sub_data)

    return waveforms

#### ====== END OF LEGACY CODE ====== ####

