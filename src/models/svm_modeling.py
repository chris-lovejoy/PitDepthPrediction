import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def get_feature_list(feats, df):
    if feats == 'all':
        feats = ['AB', 'A_', 'B_','Amp', 'Phase']
    else:
        feats = feats
    cols = list(df.columns)
    feats_list = [col for col in cols if any(substring in col for substring in feats)]
    return feats_list

def get_scaler(df, feats_list):
    x = df[feats_list]
    sc = StandardScaler().fit(x)
    return sc

def scale_the_data(sc, df, feats_list, y):
    x_scaled = sc.transform(df[feats_list])
    y = df[y]
    return x_scaled, y

def get_group_labels(df):
    df_list = []
    gp = 0
    for k, g in df.groupby(['Tube_Alias', 'Flaw_ID']):
        gp += 1
        sub_frame = g.copy()
        sub_frame['group'] = gp
        df_list.append(sub_frame)
        
    df_with_group_labels = pd.concat(df_list)
    
    group_labels = list(df_with_group_labels['group'])
    return group_labels

def svm_model(df, model, features, features_provided=False, y='Flaw_Depth'):
    
    model_dict = {
        'svm': svm.SVR(),
        'svm_lin': svm.SVR(kernel='linear'),
        'svm_rbf': svm.SVR(kernel='rbf'),
        'lin_reg': LinearRegression(),
        'lasso': Lasso(),
        'ridge': Ridge(),
        'elastic': ElasticNet()
        }
    
    parameter_dict = {
        'svm': {
            'kernel': ('rbf', 'linear'),
            'C': (1e-1, 1e0, 1e1),
            'gamma':(1e-4,1e-3,1e-2,1e-1)
                  },
        'svm_lin':{
            'C': (1e-1, 1e0, 1e1)
                  },
        'svm_rbf':{
            'C': (1e-1, 1e0, 1e1),
            'gamma':(1e-4,1e-3,1e-2,1e-1)
                  },
        'lin_reg': {
            "fit_intercept": (True, False),
            "normalize": (True,False),
            "copy_X": (True, False),
            "n_jobs": (1,2,3,4,5)
            },
        'lasso': {
            "alpha":(1e-2,1e-1,1, 1e1,1e2),
            "tol":(1e-5,1e-4,1e-2,1e-1),
            "max_iter": (1e2,1e3,1e4,1e5,1e6)
            },
        'ridge': {
            "alpha":(1e-2,1e-1,1, 1e1,1e2),
            "tol":(1e-5,1e-4,1e-2,1e-1),
            "max_iter": (1e2,1e3,1e4,1e5,1e6)
            },
        'elastic':{
            "alpha":(1e-2,1e-1,1, 1e1,1e2),
            "l1_ratio":(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),
            "fit_intercept":(True,False),
            "normalize":(True,False),
            
            }
        }
    
    # get a list of features
    if features_provided == True:
        feats_list = features
    else:
        feats_list = get_feature_list(features, df)
    
    # get the scaler
    sc = get_scaler(df, feats_list)
    
    # get the scaled X_train data and the y values
    X, y = scale_the_data(sc, df, feats_list, y)
    
    # get group labels for the training data
    group_labels = get_group_labels(df)
    
    # specify our CV type; Group K fold in this case, with 5 folds.
    gkf = GroupKFold(n_splits=5)

    # the model function
    selected_model = model_dict[model]
  
    # used Randomized Search to optimize parameters
    # we set 'cv' to 'gfk' from above.
    # model scoring is based on RMSE
    rs_cv_model = GridSearchCV(selected_model, 
                             parameter_dict[model],
                             scoring='neg_root_mean_squared_error', 
                             cv=gkf,
                             verbose=1)
    
    # fit the model, using our group labels from before.
    #print('Fitting models. Might take a minutes....')
    rs_cv_model.fit(X, y.values.ravel(),groups=group_labels)
    print('Fitting complete!')
    return rs_cv_model, sc

def make_predictions_on_test(test_data, scaler, features, model):
    # scale data
    X_scaled = scaler.transform(test_data[features])
    y_pred = model.predict(X_scaled)
    return y_pred
