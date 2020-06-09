
import random
import pandas as pd
import os
import sys

# import the pipeline module using whatever path you need
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'epri_Mar20','src'))
import models.modeling_pipeline as mp

from math import sqrt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statistics import mean

# check the path
df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'epri_Mar20', 'data','interim', 'complete_feature_cg_data.csv'))

def svm_model(df, feature_list=None, method='one', train_pct=0.8, num=1):
    X_train, X_test, y_train, y_test, train, test = mp.get_scaled_training_test_data(df, feature_list, method, train_pct, num)
    
    # model parameters to optimize
    parameters = {
        'gamma':(0.001, 0.01, 0.1)
                  }
    # the model function
    lin_svm = svm.SVR()
    
    # used Grid Search to optimize parameters
    lin = GridSearchCV(lin_svm, parameters, cv=5)
    
    # fit the model
    lin.fit(X_train, y_train)
    
    # make predictions
    y_pred = lin.predict(X_test)
    
    # calculate rmse
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    #print(lin.get_params)
    
    return rmse

methods = ['one', 'two', 'three', 'four', 'five', 'six']

method_dict = {}

for method in methods:
    rmse_list = []
    print("Calculating SVM models using method {}...".format(method))
    for i in range(1,21):
        rmse = svm_model(df, feature_list=None, method=method, train_pct=0.8, num=1)
        rmse_list.append(rmse)
    method_dict[method] = mean(rmse_list)
    
print(method_dict)