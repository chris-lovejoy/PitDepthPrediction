import pandas as pd
import random
from math import sqrt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def subset_data_with_features(df, feature_list=None, y_var=['Flaw_Depth']):
    # subsets the dataframe based on a list of features
    # default value simply returns the full dataframe
    if feature_list == None:
        last_cols = ['Flaw_Depth', 'Pct_Depth', 'Flaw_Volume', 'Flaw_Area']
        cols_to_drop = list(set(last_cols) - set(y_var))    
        return df[df.columns.drop(cols_to_drop)]
    else:
        id_list= ['Tube_Alias', 'Flaw_ID', 'Angle']
        cols = id_list + feature_list + y_var
        return df[cols]

def exclude_random_tube(df, seed=42):
    # excludes one random tube alias from the data
    tubes = list(df['Tube_Alias'].unique())
    df = df.loc[~(df['Tube_Alias'] == random.Random(seed).choice(tubes))].reset_index(drop=True)
    return df    

def pick_random_angle_rows(df,num,seed=42):
    # subsets the data by picking random angle values for each tube/flaw pair
    
    # create empty lit to store dataframe subsets    
    rand_list = []
    
    # group the data by tube alias and flaw id
    # randomly pick one or mre angle values for each tube/flaw pair    
    for k,g in df.groupby(['Tube_Alias', 'Flaw_ID']):
        angles = list(g['Angle'].unique())
        if len(angles) >= num:
            rand_angles = random.Random(seed).sample(angles,num)
            arg_list = ["(df['Angle'] == {})".format(angle) for angle in rand_angles]
            joined_args = ' | '.join(arg_list)
            rand_list.append(g.loc[(eval(joined_args))])           
        else:
            rand_list.append(g)
        
    # turn the list of dfs back into one df        
    rand_data = pd.concat(rand_list).reset_index(drop=True)
    return rand_data

def evenly_distribute(df, seed=42):
    # function to ensure that the training data has equal numbers of
    # cases for each flaw depth
    
    # list of the flaw IDs
    flaw_list = list(df['Flaw_ID'].unique())
    
    #  list for storing number of examples per flaw ID
    flaw_df_len = [g.shape[0] for k,g in df.groupby('Flaw_ID')]
    
    #  list for storing each data frame subsampled by flaw ID
    flaw_df = [g for k,g in df.groupby('Flaw_ID')]
    
    # make a dictionary with flaw ID as key and df subset as the value
    flaw_df_dict = dict(zip(flaw_list, flaw_df))
    
    # sort the df lengths to find the shortest one
    # i.e., with the fewest data points 
    flaw_df_len.sort()
    shortest_cat = flaw_df_len[0]
    
    # change dictionary values by randomly shuffling them, and then resizing based on the smallest one
    flaw_df_dict = {key:value.sample(frac=1, random_state=seed)[:shortest_cat] for key, value in flaw_df_dict.items()}
    
    # turn the dictionary back into a dataframe
    dict_list = [v for k,v in flaw_df_dict.items()]
    training_data = pd.concat(dict_list)
    
    # shuffle the data again
    training_data = training_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    return training_data

def split_tube_flaw_between_train_test(df, train_pct, seed=42):
    # splits data into training and test set so that 
    # the tube/flaw combinations are unique in the training/test sets
    # i.e., if AP01/A has multple angles, there won't be a mix of AP01/A
    # between the training and test set; all would be in one or the other
    
    # get a list of df's sorted by tube alias and flaw id
    df_list = [df_subset for tube_flaw, df_subset in df.groupby(['Tube_Alias', 'Flaw_ID'])]
    
    # shuffle the order of the list to randomize it
    random.Random(seed).shuffle(df_list)
    
    # select the train and test sets
    train_list = df_list[:round(len(df_list)*train_pct)]
    test_list = df_list[-round(len(df_list)*(1-train_pct)):]
    
    # concatenate the lists back into dataframes
    training = pd.concat(train_list).reset_index(drop=True)
    training = evenly_distribute(training, seed)
    test = pd.concat(test_list).reset_index(drop=True)
    return training, test

def use_train_test_split(df, train_pct, seed=42):
    # this simply uses the default train_test_split from sklearn to
    # create training and test sets
    training, test = train_test_split(df, 
                                      test_size=1-train_pct,
                                      random_state=seed)    
    training = evenly_distribute(training, seed).reset_index(drop=True)
    return training, test
    
def get_training_data(training, test, y_var):

    # selects the training data
    X_train = training.iloc[:, 3:-len(y_var)]
    y_train = training[y_var]
                        
    X_test = test.iloc[:, 3:-len(y_var)]
    y_test = test[y_var]
    # scale the data
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)   
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_one_data_set(training, y_var):

    # selects the training data
    X_train = training.iloc[:, 3:-len(y_var)]
    y_train = training[y_var]
                        
    # scale the data
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)   

    return X_train, y_train

# the following functions are simply different combinations of the functions above
# they are selected by the driver function get_scaled_training_test_data

def no_data_reduction_sklearn(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = subset_data_with_features(df, feature_list, y_var)
    training, test = use_train_test_split(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

def rand_angle_sklearn(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = pick_random_angle_rows(subset_data_with_features(df, feature_list, y_var), num, seed)
    training, test = use_train_test_split(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

def rand_angle_tube_sklearn(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = pick_random_angle_rows(exclude_random_tube(subset_data_with_features(df, feature_list, y_var), seed), num, seed)
    training, test = use_train_test_split(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

def no_data_reduction_custom(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = subset_data_with_features(df, feature_list, y_var)
    training, test = split_tube_flaw_between_train_test(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

def rand_angle_custom(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = pick_random_angle_rows(subset_data_with_features(df, feature_list, y_var), num, seed)
    training, test = split_tube_flaw_between_train_test(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

def rand_angle_tube_custom(df, feature_list=None, y_var=['Flaw_Depth'], train_pct=0.8, num=1, seed=42):
    df = pick_random_angle_rows(exclude_random_tube(subset_data_with_features(df, feature_list, y_var), seed), num, seed)
    training, test = split_tube_flaw_between_train_test(df, train_pct, seed)
    X_train, X_test, y_train, y_test = get_training_data(training, test, y_var)
    return X_train, X_test, y_train, y_test, training, test

# driver function for the above methods
# simply pass your feature list, method ('one', 'two', 'three', 'four' or 'five'),
# the train-test split you want (train_pct), and the number of angles you want if applicable
# returns scaled train and test data, and the appropriate y values

def get_scaled_training_test_data(df, feature_list=None, y_var=['Flaw_Depth'], method='one', train_pct=0.8, num=1, seed=42):
    method_dict = {
        'one':no_data_reduction_sklearn,
        'two':rand_angle_sklearn,
        'three': rand_angle_tube_sklearn,
        'four': no_data_reduction_custom,
        'five': rand_angle_custom,
        'six': rand_angle_tube_custom
        }
    X_train, X_test, y_train, y_test, training, test = method_dict[method](df, feature_list, y_var, train_pct, num, seed)
    
    
    return X_train, X_test, y_train, y_test, training, test

############### Functions for removing outliers
def remove_outliers(X_train, y_train, treatment = 'zscore_removal', threshold = None ):
    '''
    For zscore_capping, the threshold is z-score; z = 3 is default which restricts range to (-3*z, 3*z). Default is 3.
    For percentile_capping, the threshold is p, percentile score; p = 0.01 is default which restricts range to (0.01,0.99) in percentile score. Default is 0.01.
    For cooks_removal, the threshold is fraction of points to be removed. 4/n is default which will remove 4/n fraction of points with highest cook's distance
    For leverage_removal, the threshold is fraction of points to be removed. 0.04 is default which will remove 4% of points with highest leverage
    '''
    treatment_dict = {
                        'cooks_removal':remove_cooks_outliers,
                        'leverage_removal':remove_leverage_points,
                        'zscore_removal':remove_zscore_outliers
                        }    
    
    X_train, y_train = treatment_dict[treatment](X_train, y_train, threshold)
    return X_train, y_train

def remove_leverage_points(X_train, y_train, z_thres):
    '''
    Removes high leverage points from the dataset. Note, outliers are different from leverage points.
    To understand what are leverage points, see discussion here: https://online.stat.psu.edu/stat462/node/170/
    z_thres is the z-score for thresholding outliers.
    '''
    
    #Train a stats model to get leverage points
    import statsmodels.api as sm
    model = sm.OLS(y_train,sm.add_constant(X_train))
    results = model.fit()
    
    
    influence = results.get_influence() #create instance of influence
   
    leverage = influence.hat_matrix_diag  #leverage (matrix hat values)
    
    norm_residuals = influence.resid_studentized_internal # normalized residuals
    
    #calculate z-score for leverage values
    z_leverage = (leverage - leverage.mean())/leverage.std()
    
    # threshold for removing points
    if z_thres == None:
        z_thres = 3
    
    # determination of leverage values exceeding a certain z-score    
    outlier_idx = np.nonzero(z_leverage > z_thres)
    remain_idx = np.nonzero(z_leverage <= z_thres)
    
    # removal of outliers
    X_train, y_train = outlier_removal(X_train, y_train, outlier_idx)
    
    return X_train, y_train


def remove_zscore_outliers(X_train, y_train, z_thres):
    '''
    Accepts a dataframe 'df', and 'cols' which is a list of 
    features that are being used for training or need treatment.
    Input 'z_thres' is the z-score for removing outliers. Default is 3 which 
    covers 99% of the normally distributed data.
    The program calculates the l2-norm for standardized data and removes 
    outliers which exceed 'z_thres'
    '''
    # Perform standardization of features/X data
    X = X_train
    no_features = X_train.shape[1]
    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    # Calculate the L2-norm for each row and divide it by sqrt of column size
    l2_zscore = np.linalg.norm(X, axis=1)/np.sqrt(no_features)
    
    # If z_thres has 'None' input, use default value 3
    if z_thres == None:
        z_thres = 3
    
    # Find outliers that exceed 'z_thres'
    outlier_idx = np.nonzero(l2_zscore > z_thres)
    
    # Remove outliers
    X_train, y_train = outlier_removal(X_train, y_train, outlier_idx)
    return X_train, y_train


def remove_cooks_outliers(X_train, y_train, cooks_threshold):
    '''
    Accepts a matrix 'X_train', and 'cols' which is a list of 
    features that are being used for training or need treatment.
    Input 'cooks_distance' is the threshold for removing outliers. Default is 4/num_rows. 
    '''
    from yellowbrick.regressor import CooksDistance
    
    # Fit to a linear model to calculate Cooks Distance
    cooks = CooksDistance()
    cooks.fit(X_train, y_train)
    cooks.show()
    
    # Assign a Cooks distance threshold
    if cooks_threshold == None:
        cooks_threshold = cooks.influence_threshold_
    
    outlier_idx = np.nonzero(cooks.distance_ > cooks_threshold)
    
    X_train, y_train = outlier_removal(X_train, y_train, outlier_idx)
    return X_train, y_train

def outlier_removal(X_train, y_train, outlier_idx):
    # Remove outliers by indices listed in 'outlier_idx'
    # X_train has to be a pd Dataframe
    X_train = np.delete(X_train,outlier_idx,0)
    y_train = np.delete(np.array(y_train),outlier_idx,0)
    print('Number of rows removed: ' + str(len(outlier_idx[0])) + '/' + str(len(X_train)))
    print('Indices of rows removed: ' + str(outlier_idx[0]))
    return X_train, y_train

######### Functions for capping outliers
    
def cap_outliers(X_train, feature_list, treatment = 'zscore_capping', threshold = None ):
    '''
    For zscore_capping, the threshold is z-score; z = 3 is default which restricts range to (-3*z, 3*z). Default is 3.
    For percentile_capping, the threshold is p, percentile score; p = 0.01 is default which restricts range to (0.01,0.99) in percentile score. Default is 0.01.
    For cooks_removal, the threshold is fraction of points to be removed. 4/n is default which will remove 4/n fraction of points with highest cook's distance
    For leverage_removal, the threshold is fraction of points to be removed. 0.04 is default which will remove 4% of points with highest leverage
    '''
    treatment_dict = {
                        'zscore_capping':cap_zscore_outliers,
                        'percentile_capping':cap_percentile_outliers,
                        }    
    
    X_train = treatment_dict[treatment](X_train, feature_list, threshold)
    return X_train
def cap_percentile_outliers(X_train, cols, p_thres):
    '''
    Accepts a matrix 'X_train', and 'cols' which is a list of 
    features that are being treated corresponding to columns in X_train
    Input 'p_thres' should be between 0 and 0.5. Default is 0.01 -> [0.01 - 0.99] bounds
    '''
    df = pd.DataFrame(X_train, columns = cols)
    # Calculate the percentile bounds for capping the data
    if p_thres == None:
        p_thres = 0.01
    p_low = p_thres
    p_high = 1-p_thres
    
    # A simple for loop to cap values below and above percentile bounds
    for col in cols:
        percentiles = df[col].quantile([p_low, p_high]).values
        df[col][df[col] <= percentiles[0]] = percentiles[0]
        df[col][df[col] >= percentiles[1]] = percentiles[1]
        X_train = np.array(df)
        return X_train
    
def cap_zscore_outliers(X_train, cols, z_thres):
    '''
    Accepts a matrix 'X_train', and 'cols' which is a list of 
    features that are being treated corresponding to columns in X_train
    Input 'z_thres' is the z-score for capping outliers. Default is 3 which 
    covers 99% of the normally distributed data.
    '''
    df = pd.DataFrame(X_train, columns = cols)
    # If z_thres has 'None' input, use default value 3
    if z_thres == None:
        z_thres = 3

    # A simple for loop to cap values below and above z-score bounds
    for col in cols:
        z_low = df[col].mean() - z_thres * df[col].std()
        z_high = df[col].mean() + z_thres * df[col].std()
        df[col][df[col] <= z_low] = z_low
        df[col][df[col] >= z_high] = z_high
        X_train = np.array(df)
        return X_train

######### Apply a power transform to the specified columns

def change_power(df, cols, exp):
    '''
    Raise specific columns 'col' to an exponent 'exp' in the dataframe 'df'
    '''
    for col in cols:
        df[col] = np.power(df[col], exp)
    return df

def tangent(cols):
    '''
    Apply a tan function to specific columns 'col' to an exponent 'exp' in the dataframe 'df'
    '''
    for col in cols:
        df[col] = np.tan(df[col] * np.pi/180)
    return df
    
##################### Get a list of column names
def find_features(df, list_prefix):
    '''
    Get a list of feature names in the dataframe 'df' that share a common prefix
    'list_prefix' can be ['Amp','Phas'] etc
    '''
    cols = df.columns
    list_features = []
    for col in cols:
        for prefix in list_prefix:
            if col.startswith(prefix):
                list_features.append(col)
    return list_features
                
    
    for col in feature_prefix:
        df[col] = np.power(df[col])
    return df

################ Remove correlated variables based on correlatin coefficient
def remove_corr_coef(df, feature_list, threshold = 0.95, plot=True):
    #Some help from https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    '''
    Remove features in the 'feature_list' that share correlation values that are greater than 'threshold'
    '''
    corrmat = df[feature_list].corr().abs()
    
    if plot == True:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(12, 9))
        
        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=.8, square=True)
        
        f.tight_layout()    
    
    # Select upper triangle of correlation matrix
    upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = {column for column in upper.columns if any(upper[column] > threshold)}
    feature_list = {column for column in feature_list}
    remaining_features = list(feature_list - to_drop)
    
    return remaining_features

############# Remove collinearity based on variance inflation factor

def remove_coll_vif(df, feature_list, thresh=5.0):
    #Adapted from https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
    '''
    Remove features while the variance inflation factor is greater than 'threshold'
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor    
    X = df[feature_list]
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    remaining_features = list(X.columns[variables])
    return remaining_features