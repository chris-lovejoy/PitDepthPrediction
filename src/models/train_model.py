import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

def split_dataset(input_features, output_features, ratio):
    """
    This takes in dataframes of the input and output features and 
    returns training and valid/test subsets, divided by the  ratio 
    specified in the function.
    
    Note: the samples should already have been shuffled before this function is called
    
    Note 2: this is similar to the function above, but (1) returns as a dataframe and 
    (2) looks at data where input vs output split has already been made.
    """
    
    
    # Define the training and validation samples
    num_samples = len(input_features)
    num_valid = int(np.ceil(num_samples * ratio))
    
    X_train = input_features._slice(slice(num_valid,num_samples)).reset_index().drop(['index'],axis=1)
    y_train = output_features._slice(slice(num_valid,num_samples)).reset_index().drop(['index'],axis=1)
    
    X_valid = input_features._slice(slice(0,num_valid)).reset_index().drop(['index'],axis=1)
    y_valid = output_features._slice(slice(0,num_valid)).reset_index().drop(['index'],axis=1)

    return X_train, y_train, X_valid, y_valid


def generate_groups(df):
    labels = df.Tube_Alias + df.Flaw_ID
    labels_unique = labels.unique()
    groups_labels=np.zeros(len(labels))
    for i in range(len(labels)):
        boolean = labels[i] == labels_unique
        index = [j for j, val in enumerate(boolean) if val] 
        groups_labels[i] = index[0] 
    return groups_labels

def generate_groupcv_object(X, y, groups_labels, no_folds):
    group_kfold = GroupKFold(n_splits = no_folds)
    gkf = group_kfold.split(X, y, groups_labels)
    return gkf


def perform_lassoridge_cv(X, y, X_columns, X_test, y_test):
    ''' Performs lasso and ridge regression on training data consisting of 'X' and 'y'
    and uses the results of the modeling to test perfomance on 'X_test' and 'y_test'
    Also generates plots to assess weights of different predictors and cross-validation
    score as a function of penalty coefficient, alpha'''

    # Instanstiate a model class and perform fitting for lasso and ridge regression
    model_lasso = LassoCV(cv=10,verbose=0,normalize=True,eps=0.0008,n_alphas=1000, tol=0.0001,max_iter=10000, random_state = 0)
    model_lasso.fit(X,y)
    
    n_alphas = 200
    alphasr = np.logspace(-6, 2, n_alphas)
    
    #model_ridge = RidgeCV(cv=10,normalize=True, alphas = alphasr, store_cv_values = True, gcv_mode = 'eigen')
    model_ridge = RidgeCV(store_cv_values = True, normalize = True, alphas = alphasr)
    model_ridge.fit(X,y)
    
    # Perform linear regression for comparison later
    model_linear = LinearRegression()  
    model_linear.fit(X, y) 
    
    # Store training predictions
    y_lasso = np.array(model_lasso.predict(X))
    y_ridge = np.array(model_ridge.predict(X))
    
    # Store test predictions
    y_test_lasso = np.array(model_lasso.predict(X_test))
    y_test_ridge = np.array(model_ridge.predict(X_test))
    

    
    # Prepare model coefficients for outputs as a dataframe
    coeff_lasso = pd.DataFrame(model_lasso.coef_,index=X_columns, columns=['Coefficients'])
    df_intercept = pd.DataFrame(model_lasso.intercept_,index=['Intercept'], columns=['Coefficients'])
    coeff_lasso = coeff_lasso.append(df_intercept)
    
    coeff_ridge = pd.DataFrame(model_ridge.coef_,index=X_columns, columns=['Coefficients'])
    df_intercept = pd.DataFrame(model_ridge.intercept_,index=['Intercept'], columns=['Coefficients'])
    coeff_ridge = coeff_ridge.append(df_intercept)
    
    # Prepare plots
    fig, ax = plt.subplots(nrows=4, ncols=2,figsize=(20,32))
    
    # First row plots : strength of coefficients for lasso and ridge
    
    # Sort coefficients of the lasso and ridge model excluding the intercept for plotting
    bar_lasso = coeff_lasso.Coefficients[:-1]
    sort_index = np.argsort(-abs(bar_lasso))
    bar_lasso = bar_lasso[sort_index]
    
    bar_ridge= coeff_ridge.Coefficients[:-1]
    sort_index = np.argsort(-abs(bar_ridge))
    bar_ridge = bar_ridge[sort_index]
    
    bar_ylimit = max(max(np.abs(bar_ridge)),max(np.abs(bar_lasso)))*1.1
    
    # Plot lasso weights as bar plots
    ax[0,0].bar(range(len(bar_lasso)),bar_lasso)
    ax[0,0].set_xticks(range(len(bar_lasso)))
    ax[0,0].set_ylim([-bar_ylimit, bar_ylimit])
    ax[0,0].set_xticklabels(coeff_lasso.Coefficients.index[sort_index], rotation = 80)
    ax[0,0].set(ylabel = 'Lasso Coefficients')
    ax[1,0].set_title('Lasso : Weights of predictors', FontSize = 15)
    
    # Plot ridge weights as bar plots
    ax[0,1].bar(range(len(bar_ridge)),bar_ridge)
    ax[0,1].set_xticks(range(len(bar_lasso)))
    ax[0,1].set_ylim([-bar_ylimit, bar_ylimit])
    ax[0,1].set_xticklabels(coeff_ridge.Coefficients.index[sort_index], rotation = 80)
    ax[0,1].set(ylabel = 'Ridge Coefficients')
    ax[0,1].set_title('Ridge : Weights of predictors', FontSize = 15)
    
    # Second row plots : MSE as a function of penalty parameter 
    # With help from https://scikit-learn.org/stable/auto_examples/linear_model/
    # plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
    log_alphas_lasso = -np.log10(model_lasso.alphas_)

    ax[1,0].plot(log_alphas_lasso, model_lasso.mse_path_, ':')
    ax[1,0].plot(log_alphas_lasso, model_lasso.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    ax[1,0].axvline(-np.log10(model_lasso.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    ax[1,0].set(xlabel = '-log_10(alpha)', ylabel = 'Mean square error')
    ax[1,0].legend( loc = "upper center")
    ax[1,0].set_title('Lasso : CV-score vs regularization', FontSize = 15)
    
    cv_ridge = (np.mean(model_ridge.cv_values_, axis = 0))
    log_alphas_ridge = np.log10(alphasr)
    ax[1,1].plot(log_alphas_ridge, cv_ridge, 'k',
             label='LOOCV score', linewidth=2)
    ax[1,1].set(xlabel = 'log_10(alpha)', ylabel = 'Mean square error')
    ax[1,1].axvline(np.log10(model_ridge.alpha_), linestyle='--', color='k',
                label='alpha: LOOCV estimate')
    ax[1,1].legend( loc = "upper center")
    ax[1,1].set_title('Ridge : CV-score vs regularization', FontSize = 15)
    
    
    
    # Third row plots : deviation between true values vs predicted values for training data
    # With help from https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html
    # #sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
    
    # Preparation of plot for lasso
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_lasso, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2
    RMSE=np.sqrt(mean_squared_error(y, y_lasso))

    MAE = mean_absolute_error(y, y_lasso)
    
    ax[2,0].plot([y.min(), y.max()],
            [y.min(), y.max()],
            '--r', linewidth=2)
    ax[2,0].scatter(y, y_lasso, alpha=0.2)
    ax[2,0].spines['top'].set_visible(False)
    ax[2,0].spines['right'].set_visible(False)
    ax[2,0].get_xaxis().tick_bottom()
    ax[2,0].get_yaxis().tick_left()
    ax[2,0].spines['left'].set_position(('outward', 10))
    ax[2,0].spines['bottom'].set_position(('outward', 10))
    ax[2,0].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[2,0].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[2,0].set_xlabel('Measured')
    ax[2,0].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[2,0].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[2,0].set_title('Lasso : Training perfomance', FontSize = 15)
    
    # Preparation of plot for lasso
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_ridge, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2
    MAE = mean_absolute_error(y, y_ridge)
    RMSE=np.sqrt(mean_squared_error(y, y_ridge))
    
    ax[2,1].plot([y.min(), y.max()],
            [y.min(), y.max()],
            '--r', linewidth=2)
    ax[2,1].scatter(y, y_ridge, alpha=0.2)
    ax[2,1].spines['top'].set_visible(False)
    ax[2,1].spines['right'].set_visible(False)
    ax[2,1].get_xaxis().tick_bottom()
    ax[2,1].get_yaxis().tick_left()
    ax[2,1].spines['left'].set_position(('outward', 10))
    ax[2,1].spines['bottom'].set_position(('outward', 10))
    ax[2,1].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[2,1].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[2,1].set_xlabel('Measured')
    ax[2,1].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[2,1].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[2,1].set_title('Ridge : Training perfomance', FontSize = 15)
    
    # Fourth row plots : deviation between true values vs predicted values for test data
    # With help from https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html
    # #sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
    
    y_lasso = y_test_lasso
    y_ridge = y_test_ridge
    y = y_test
    
    # Preparation of plot for lasso
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_test_lasso, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2

    MAE = mean_absolute_error(y, y_test_lasso)
    RMSE = np.sqrt(mean_squared_error(y, y_test_lasso))
    
    ax[3,0].plot([y.min(), y.max()],
            [y.min(), y.max()],
            '--r', linewidth=2)
    ax[3,0].scatter(y, y_lasso, alpha=0.2)
    ax[3,0].spines['top'].set_visible(False)
    ax[3,0].spines['right'].set_visible(False)
    ax[3,0].get_xaxis().tick_bottom()
    ax[3,0].get_yaxis().tick_left()
    ax[3,0].spines['left'].set_position(('outward', 10))
    ax[3,0].spines['bottom'].set_position(('outward', 10))
    ax[3,0].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[3,0].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[3,0].set_xlabel('Measured')
    ax[3,0].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[3,0].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[3,0].set_title('Lasso : Test perfomance', FontSize = 15)
    
    # Preparation of plot for ridge
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_test_ridge, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2

    MAE = mean_absolute_error(y, y_test_ridge)
    RMSE=np.sqrt(mean_squared_error(y, y_test_ridge))
    
    ax[3,1].plot([y.min(), y.max()],
            [y.min(), y.max()],
            '--r', linewidth=2)
    ax[3,1].scatter(y, y_ridge, alpha=0.2)
    ax[3,1].spines['top'].set_visible(False)
    ax[3,1].spines['right'].set_visible(False)
    ax[3,1].get_xaxis().tick_bottom()
    ax[3,1].get_yaxis().tick_left()
    ax[3,1].spines['left'].set_position(('outward', 10))
    ax[3,1].spines['bottom'].set_position(('outward', 10))
    ax[3,1].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[3,1].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[3,1].set_xlabel('Measured')
    ax[3,1].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax[3,1].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[3,1].set_title('Ridge : Test perfomance', FontSize = 15)

    
    # Summary of compression
    L1_linear = sum(abs(model_linear.coef_))
    L1_lasso = sum(abs(model_lasso.coef_))
    L1_ridge = sum(abs(model_ridge.coef_))
    
    print("Compression achieved with Lasso Regression : " + str(np.around((1 - L1_lasso/L1_linear)*100,2)) + "% with a penalty coefficient of : " + str(model_lasso.alpha_))
    print("Compression achieved with Ridge Regression : " + str(np.around((1 - L1_ridge/L1_linear)*100,2)) + "% with a penalty coefficient of : " + str(model_ridge.alpha_))
    
    return model_lasso, model_ridge, coeff_lasso, coeff_ridge


def lasso_elasticnets_groupcv(df, X_columns, y_feature, groups_labels, no_folds):
    ''' Performs lasso and ridge regression on training data consisting of 'X' and 'y'
    Also generates plots to assess weights of different predictors and cross-validation
    score as a function of penalty coefficient, alpha'''
    
    
    X = df[X_columns]
    y = df[y_feature]
    gkf = generate_groupcv_object(X, y, groups_labels, no_folds)
    # Instanstiate a model class and perform fitting for lasso and ridge regression
    model_lasso = LassoCV(cv=gkf,verbose=0,normalize=True,eps=3e-4,n_alphas=1000, tol=0.0001,max_iter=10000, selection = 'random')
    model_lasso.fit(X,y)
    
    n_alphas = 200
    alphasr = np.logspace(-6, 2, n_alphas)
    
    gkf = generate_groupcv_object(X, y, groups_labels, no_folds)
    #model_ridge = RidgeCV(cv=10,normalize=True, alphas = alphasr, store_cv_values = True, gcv_mode = 'eigen')
    model_elasticnet = ElasticNetCV(cv=gkf,verbose=0,normalize=True,eps=3e-4,n_alphas=1000, tol=0.0001,max_iter=10000, selection = 'random')
    model_elasticnet.fit(X,y)
    
    # Perform linear regression for comparison later
    model_linear = LinearRegression()  
    model_linear.fit(X, y) 
    
    # Store training predictions
    y_lasso = np.array(model_lasso.predict(X))
    y_elasticnet = np.array(model_elasticnet.predict(X))

    
    # Prepare model coefficients for outputs as a dataframe
    coeff_lasso = pd.DataFrame(model_lasso.coef_,index=X_columns, columns=['Coefficients'])
    df_intercept = pd.DataFrame(model_lasso.intercept_,index=['Intercept'], columns=['Coefficients'])
    coeff_lasso = coeff_lasso.append(df_intercept)
    
    coeff_elasticnet = pd.DataFrame(model_elasticnet.coef_,index=X_columns, columns=['Coefficients'])
    df_intercept = pd.DataFrame(model_elasticnet.intercept_,index=['Intercept'], columns=['Coefficients'])
    coeff_elasticnet = coeff_elasticnet.append(df_intercept)
    
    # Prepare plots
    fig, ax = plt.subplots(nrows=3,  ncols=2,figsize=(20,24))
    
    # First row plots : strength of coefficients for lasso and elasticnet
    
    # Sort coefficients of the lasso and elasticnet model excluding the intercept for plotting
    bar_lasso = coeff_lasso.Coefficients[:-1]
    sort_index = np.argsort(-abs(bar_lasso))
    bar_lasso = bar_lasso[sort_index]
    
    bar_elasticnet= coeff_elasticnet.Coefficients[:-1]
    sort_index = np.argsort(-abs(bar_elasticnet))
    bar_elasticnet = bar_elasticnet[sort_index]
    
    bar_ylimit = max(max(np.abs(bar_elasticnet)),max(np.abs(bar_lasso)))*1.1
    
    # Plot lasso weights as bar plots
    ax[0,0].bar(range(len(bar_lasso)),bar_lasso)
    ax[0,0].set_xticks(range(len(bar_lasso)))
    ax[0,0].set_ylim([-bar_ylimit, bar_ylimit])
    ax[0,0].set_xticklabels(coeff_lasso.Coefficients.index[sort_index], rotation = 80)
    ax[0,0].set(ylabel = 'Lasso Coefficients')
    ax[1,0].set_title('Lasso : Weights of predictors', FontSize = 15)
    
     #Plot elasticnet weights as bar plots
    ax[0,1].bar(range(len(bar_elasticnet)),bar_elasticnet)
    ax[0,1].set_xticks(range(len(bar_lasso)))
    ax[0,1].set_ylim([-bar_ylimit, bar_ylimit])
    ax[0,1].set_xticklabels(coeff_elasticnet.Coefficients.index[sort_index], rotation = 80)
    ax[0,1].set(ylabel = 'Elasticnet Coefficients')
    ax[0,1].set_title('Elasticnet : Weights of predictors', FontSize = 15)
    
    # Second row plots : MSE as a function of penalty parameter 
    # With help from https://scikit-learn.org/stable/auto_examples/linear_model/
    # plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
    log_alphas_lasso = -np.log10(model_lasso.alphas_)
    log_alphas_elasticnet = -np.log10(model_elasticnet.alphas_)

    ax[1,0].plot(log_alphas_lasso, model_lasso.mse_path_, ':')
    ax[1,0].plot(log_alphas_lasso, model_lasso.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    ax[1,0].axvline(-np.log10(model_lasso.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    ax[1,0].set(xlabel = '-log_10(alpha)', ylabel = 'Mean square error')
    ax[1,0].legend( loc = "upper center")
    ax[1,0].set_title('Lasso : CV-score vs regularization', FontSize = 15)
    
    ax[1,1].plot(log_alphas_elasticnet, model_elasticnet.mse_path_, ':')
    ax[1,1].plot(log_alphas_elasticnet, model_elasticnet.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    ax[1,1].axvline(-np.log10(model_elasticnet.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    ax[1,1].set(xlabel = '-log_10(alpha)', ylabel = 'Mean square error')
    ax[1,1].legend( loc = "upper center")
    ax[1,1].set_title('elasticnet : CV-score vs regularization', FontSize = 15)
    
    
    
    # Third row plots : deviation between true values vs predicted values for training data
    # With help from https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html
    # #sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
    
    # Preparation of plot for lasso
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_lasso, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2
    RMSE=np.sqrt(mean_squared_error(y, y_lasso))

    MAE = mean_absolute_error(y, y_lasso)
    ax[2,0].plot([y.min(), y.max()],
             [y.min(), y.max()],
             '--r', linewidth=2)
    ax[2,0].scatter(y, y_lasso, alpha=0.2)
    ax[2,0].spines['top'].set_visible(False)
    ax[2,0].spines['right'].set_visible(False)
    ax[2,0].get_xaxis().tick_bottom()
    ax[2,0].get_yaxis().tick_left()
    ax[2,0].spines['left'].set_position(('outward', 10))
    ax[2,0].spines['bottom'].set_position(('outward', 10))
    ax[2,0].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[2,0].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[2,0].set_xlabel('Measured')
    ax[2,0].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                       edgecolor='none', linewidth=0)
    ax[2,0].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[2,0].set_title('Lasso : Perfomance', FontSize = 15)
    
    # Preparation of plot for lasso
    # Calculation for R-squared
    corr_mat = np.corrcoef(y_elasticnet, y)
    corr_xy = corr_mat[0,1]
    r_squared = corr_xy**2
    MAE = mean_absolute_error(y, y_elasticnet)
    RMSE=np.sqrt(mean_squared_error(y, y_elasticnet))
    
    ax[2,1].plot([y.min(), y.max()],
             [y.min(), y.max()],
             '--r', linewidth=2)
    ax[2,1].scatter(y, y_elasticnet, alpha=0.2)
    ax[2,1].spines['top'].set_visible(False)
    ax[2,1].spines['right'].set_visible(False)
    ax[2,1].get_xaxis().tick_bottom()
    ax[2,1].get_yaxis().tick_left()
    ax[2,1].spines['left'].set_position(('outward', 10))
    ax[2,1].spines['bottom'].set_position(('outward', 10))
    ax[2,1].set_xlim([y.min()*0.8, y.max()*1.2])
    ax[2,1].set_ylim([y.min()*0.7, y.max()*1.2])
    ax[2,1].set_xlabel('Measured')
    ax[2,1].set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                           edgecolor='none', linewidth=0)
    ax[2,1].legend([extra],['R2 = ' + str(np.around(r_squared,3))+ ', MAE = ' + str(np.around(MAE,5)) + ', RMSE = '+ str(np.around(RMSE,5))],loc='upper left')
    ax[2,1].set_title('Elasticnet : Perfomance', FontSize = 15)
    
       # Summary of compression
    L1_linear = sum(abs(model_linear.coef_))
    L1_lasso = sum(abs(model_lasso.coef_))
    L1_elasticnet = sum(abs(model_elasticnet.coef_))
    
    ind = np.where(model_lasso.alphas_ == model_lasso.alpha_)
    cv_mse_lasso = model_lasso.mse_path_.mean(axis=-1)
    bestcv_rmse_lasso = np.sqrt(cv_mse_lasso[ind[0][0]])   

    ind = np.where(model_elasticnet.alphas_ == model_elasticnet.alpha_)
    cv_mse_elasticnet = model_elasticnet.mse_path_.mean(axis=-1)
    bestcv_rmse_elasticnet = np.sqrt(cv_mse_elasticnet[ind[0][0]])    
    
    
    np.where(model_elasticnet.alphas_ == model_elasticnet.alpha_)
    print("Lasso optimization results : Compression = " + str(np.around((1 - L1_lasso/L1_linear)*100,2)) + "% , alpha : " + str(model_lasso.alpha_) + ", CV RMSE : " + str(np.around(bestcv_rmse_lasso,5)))
    print("Elastic Nets optimization results : Compression = " + str(np.around((1 - L1_elasticnet/L1_linear)*100,2)) + "% alpha : " + str(model_elasticnet.alpha_) + ", CV RMSE " + str(np.around(bestcv_rmse_elasticnet,5)))
    
    return model_lasso, model_elasticnet, coeff_lasso, coeff_elasticnet