import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import json
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


import datetime
from datetime import datetime

from sklearn.metrics import mean_squared_error

def evaluate_models(Estimators, X_train, y_train, X_test, y_test, save_png = False, filepath = None):
    no_of_models = len(Estimators)
    fig, ax = plt.subplots(nrows=len(Estimators), ncols=2,figsize=(20 ,8*no_of_models))
    j=-1
    for est in Estimators:
        j = j + 1
        y_train_pred = np.array(est.predict(X_train))
        y_test_pred = np.array(est.predict(X_test))
        
        train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
        # Calculation for R-squared
        corr_mat = np.corrcoef(y_train_pred, y_train)
        corr_xy = corr_mat[0,1]
        rsq_train = corr_xy**2 
        corr_mat = np.corrcoef(y_test_pred, y_test)
        corr_xy = corr_mat[0,1]
        rsq_test = corr_xy**2
        
        mae_train = mean_absolute_error(y_train_pred, y_train)
        mae_test = mean_absolute_error(y_test_pred, y_test)
        model_name = type(est).__name__
        
        
        # Preparation of plot for training data
        ax[j,0].plot([y_train.min(), y_train.max()],
                [y_train.min(), y_train.max()],
                '--r', linewidth=2)
        ax[j,0].scatter(y_train, y_train_pred, alpha=0.2)
        ax[j,0].spines['top'].set_visible(False)
        ax[j,0].spines['right'].set_visible(False)
        ax[j,0].get_xaxis().tick_bottom()
        ax[j,0].get_yaxis().tick_left()
        ax[j,0].spines['left'].set_position(('outward', 10))
        ax[j,0].spines['bottom'].set_position(('outward', 10))
        ax[j,0].set_xlim([y_train.min()*0.8, y_train.max()*1.2])
        ax[j,0].set_ylim([y_train.min()*0.7, y_train.max()*1.2])
        ax[j,0].set_xlabel('Measured')
        ax[j,0].set_ylabel('Predicted')
        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)
        ax[j,0].legend([extra],['R2 = ' + str(np.around(rsq_train,3))+ ', MAD = ' + str(np.around(mae_train,5)) + ', RMSE = ' + str(np.around(train_rmse,4))],loc='upper left')
        ax[j,0].set_title(model_name + ' : Training perfomance', FontSize = 15)
        
    
        # Preparation of plot for test data
        ax[j,1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                '--r', linewidth=2)
        ax[j,1].scatter(y_test, y_test_pred, alpha=0.2)
        ax[j,1].spines['top'].set_visible(False)
        ax[j,1].spines['right'].set_visible(False)
        ax[j,1].get_xaxis().tick_bottom()
        ax[j,1].get_yaxis().tick_left()
        ax[j,1].spines['left'].set_position(('outward', 10))
        ax[j,1].spines['bottom'].set_position(('outward', 10))
        ax[j,1].set_xlim([y_test.min()*0.8, y_test.max()*1.2])
        ax[j,1].set_ylim([y_test.min()*0.7, y_test.max()*1.2])
        ax[j,1].set_xlabel('Measured')
        ax[j,1].set_ylabel('Predicted')
        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)
        ax[j,1].legend([extra],['R2 = ' + str(np.around(rsq_test,3))+ ', MAD = ' + str(np.around(mae_test,5)) + ', RMSE = ' + str(np.around(test_rmse,4))],loc='upper left')
        ax[j,1].set_title(model_name + ' : Test perfomance', FontSize = 15)
     
    if save_png == True:
        fig.savefig(filepath)
        
def evaluate_model(est, X_train, y_train, X_test, y_test, save_png = False, filepath = None):
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20 ,8))
    
    y_train_pred = np.array(est.predict(X_train))
    y_test_pred = np.array(est.predict(X_test))
    
    train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_train_pred, y_train)
    corr_xy = corr_mat[0,1]
    rsq_train = corr_xy**2 
    corr_mat = np.corrcoef(y_test_pred, y_test)
    corr_xy = corr_mat[0,1]
    rsq_test = corr_xy**2
    
    mae_train = mean_absolute_error(y_train_pred, y_train)
    mae_test = mean_absolute_error(y_test_pred, y_test)
    model_name = type(est).__name__
    
    
    # Preparation of plot for training data
    ax[0].plot([y_train.min(), y_train.max()],
            [y_train.min(), y_train.max()],
            '--r', linewidth=2)
    ax[0].scatter(y_train, y_train_pred, alpha=0.2)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].get_xaxis().tick_bottom()
    ax[0].get_yaxis().tick_left()
    ax[0].spines['left'].set_position(('outward', 10))
    ax[0].spines['bottom'].set_position(('outward', 10))
    ax[0].set_xlim([y_train.min()*0.8, y_train.max()*1.2])
    ax[0].set_ylim([y_train.min()*0.7, y_train.max()*1.2])
    ax[0].set_xlabel('Measured')
    ax[0].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[0].legend([extra],['R2 = ' + str(np.around(rsq_train,3))+ ', MAD = ' + str(np.around(mae_train,5)) + ', RMSE = ' + str(np.around(train_rmse,4))],loc='upper left')
    ax[0].set_title(model_name + ' : Training perfomance', FontSize = 15)
    

    # Preparation of plot for test data
    ax[1].plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            '--r', linewidth=2)
    ax[1].scatter(y_test, y_test_pred, alpha=0.2)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].get_xaxis().tick_bottom()
    ax[1].get_yaxis().tick_left()
    ax[1].spines['left'].set_position(('outward', 10))
    ax[1].spines['bottom'].set_position(('outward', 10))
    ax[1].set_xlim([y_test.min()*0.8, y_test.max()*1.2])
    ax[1].set_ylim([y_test.min()*0.7, y_test.max()*1.2])
    ax[1].set_xlabel('Measured')
    ax[1].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[1].legend([extra],['R2 = ' + str(np.around(rsq_test,3))+ ', MAD = ' + str(np.around(mae_test,5)) + ', RMSE = ' + str(np.around(test_rmse,4))],loc='upper left')
    ax[1].set_title(model_name + ' : Test perfomance', FontSize = 15)
     
    if save_png == True:
        fig.savefig(filepath)
    


def run_model_store_JSON(fitted_model, X_train, y_train, X_test, y_test, dataset_used,
                         feature_columns, target_column, JSON_filename, 
                         optional_comment=None, task_type="regression", 
                         save_model = False, scaler=None):
    """
    This function takes in the following parameters:
        - fitted_model: the model that has been trained on the training set
        - X_train and y_train: the training datasets
        - X_test and Y-test: the test datasets
        - dataset_used: the .csv filenmae of the dataset used for training and testing
        - feature_columns: a list of the column names used for training
        - target_column: the output variable of interest
        - JSON_filename: the desired JSON file to record information to
        - optional_comment: takes a string adding any relevant information
            e.g. "PCA features were created using all phase recordings"
        - task_type: default is regression task, can specify 'classification' and different
            performance metris will be returned
        - save_model_as_pickle will save a .pkl file of the model in the
            'models' folder if True.
        - scaler: takes the scaler that was used to scale the training set and (1) save it as
            a pickle file and (2) records the directory for the file in the JSON

    The function stores the above in a JSON format, as well as the model performance
    and the time of training.
    """
    
    model_data = {}
    
    model_data['type_of_model'] = type(fitted_model).__name__
    model_data['parameters'] = fitted_model.get_params()
    model_data['dataset'] = dataset_used
    model_data['features'] = list(feature_columns)
    model_data['target_output'] = target_column
    model_data['time_run'] = date_model_run()
    if save_model == True:
        model_data['pickle_name'] = save_model_as_pickle(fitted_model, model_data['time_run'])
        if scaler is not None:
            model_data['pickle_scaler'] = save_scaler_as_pickle(scaler, model_data['pickle_name'])
    model_data['train_performance'] = evaluate_performance(fitted_model,
              X_train, y_train, train_test='train', task_type=task_type)
    model_data['test_performance'] = evaluate_performance(fitted_model,
              X_test, y_test, train_test='test', task_type=task_type)
    model_data['comment'] = optional_comment
    
    # Re-direct to 'models' folder, to save output into JSON in that location
    dirname = os.path.dirname(os.getcwd()) 
    new_path = os.path.join(dirname, 'models')
    save_location = os.path.join(new_path, JSON_filename)
    
    # If filename doesn't exist, create new model record
    if not os.path.isfile(save_location):
        model_record = {'models' : []}
    
    # Otherwise, open the existing one
    else:
        with open(save_location, 'r') as file:
            model_record = json.load(file)
    
    # Append information from current model to model record
    model_record['models'].append(model_data)
    updated_model_record = json.dumps(model_record, indent=2)    
    
    # Write updated record of all models to the file
    with open(save_location, 'w') as file:
        file.write(updated_model_record)
    
    print(f'Record of model and performance stored in {JSON_filename} within models folder.')
    
    

# def JSON_to_dataframe(   ):
    # TO DO: create function which takes values from JSON and puts into dataframe
    # or 


def save_model_as_pickle(fitted_model, date_model_run):
    
    type_of_model = type(fitted_model).__name__
    filename = f'{type_of_model}_{date_model_run}'
    dirname = os.path.dirname(os.getcwd()) 
    new_path = os.path.join(dirname, 'models')
    #pickle.dump(fitted_model, open(os.path.join(new_path, filename), 'wb'))
    with open(os.path.join(new_path, filename), 'wb') as file:
        pickle.dump(fitted_model, file)
    
    print(f'Model saved in models folder as {filename}.')
    
    return filename



def save_scaler_as_pickle(scaler, model_pickle_name):
    
    filename = f'{model_pickle_name}_scaler'    
    dirname = os.path.dirname(os.getcwd()) 
    new_path = os.path.join(dirname, 'models')
    #pickle.dump(fitted_model, open(os.path.join(new_path, filename), 'wb'))
    with open(os.path.join(new_path, filename), 'wb') as file:
        pickle.dump(scaler, file)
    
    print(f'Scaler saved in models folder as {filename}.')
    
    return filename




def evaluate_performance(fitted_model, X_test, y_test, train_test,
                         task_type="regression"):
    
    prediction = fitted_model.predict(X_test)        


    if train_test == 'train':
        print("\nMODEL PERFORMANCE ON TRAINING SET:")
    elif train_test == 'test':            
        print("\nMODEL PERFORMANCE ON TEST SET:")
            
    
    if task_type=="classification":
        accuracy = accuracy_score(y_test, prediction)
        AUC = roc_auc_score(y_test, prediction)
        F1 = f1_score(y_test, prediction)
        F1_weighted = f1_score(y_test, prediction, average='weighted')
        MCC = matthews_corrcoef(y_test, prediction)
        
    
        performance_metrics = {
            "Accuracy" : accuracy,
            "AUC" : AUC,
            "F1" : F1,
            "F1_weighted" : F1_weighted,
            "MCC" : MCC
            }

        print(f"""
              Accuracy: {accuracy}
              AUC: {AUC}
              F1 score: {F1}
              Weighted F1 score: {F1_weighted}
              Matthew's Correlation Coefficient: {MCC}
              """)
        
    
    elif task_type=="regression":
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, prediction)
        
        
        performance_metrics ={
            "Mean Absolute Error" : mae,
            "Mean Squared Error" : mse,
            "Root Mean Squared Error" : rmse,
            "R Squared" : r2
            # Could consider adding R2 adjusted
            }
        
        print(f"""
              Mean Absolute Error: {mae}
              Mean Squared Error: {mse}
              Root Mean Squared Error: {rmse}
              R Squared Score: {r2} 
              """)
          
    return performance_metrics


def date_model_run():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dt_string
