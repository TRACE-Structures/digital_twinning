'''
Gradient Boosted Trees (GBT) model class for regression and classification tasks.
Supports multiple GBT implementations: scikit-learn, XGBoost, CatBoost, and LightGBM.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import plot_tree
from xgboost import XGBRegressor, XGBClassifier
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import kendalltau
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from digital_twinning.utils.gbt_plot_utils import *
from SALib.analyze import sobol


class GBTModel:
    '''
    Gradient Boosted Trees (GBT) model class for regression and classification tasks.
    Supports multiple GBT implementations: scikit-learn, XGBoost, CatBoost, and LightGBM.
    Includes methods for training, prediction, evaluation, and feature importance analysis.
    
    Attributes:
    -----------
    Q : ParameterSet
        Parameter set defining the input features.

    cols : list
        List of feature names.

    label : str
        Name of the quantity of interest (target variable).
    
    method : str
        GBT implementation method ('scikit', 'xgboost', 'catboost', 'lightgbm', 'ElasticNet', 'log_reg').
    
    model_type : str
        Type of model ('regression' or 'classification').
    
    threshold : float or None
        Threshold for classification tasks.
    
    important : list or None
        List of important features to highlight.
    
    model : object
        Trained GBT model.
    
    prediction : function
        Prediction function of the trained model.

    Methods:
    --------
    __init__(self, Q, QoI, ts=None, threshold=None, model_type="regression", gbt_method="xgboost", splitpercent=0.80, splittime=None, split_random=True, important=None)
        Initializes the GBTModel with specified parameters.

    model_dependent_data_preparation(self, d, cols, catcols, method=None)
        Prepares data based on the selected GBT method.

    split_test_and_train_data(self, d, cols, label, threshold, splitpercent, ts=None, splittime=None,
                    resampling=False, split_random=True, verbose_flag=False)
        Splits data into training and testing sets.

    train_model(self, d_X, d_y, n_est=150, max_d=3, learning_rate=0.15, k_fold=None, num_leaves=None, 
                plot_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True,
                task_type='GPU', verbose_flag=False)
        Trains the GBT model with optional k-fold cross-validation.

    train_and_validate(self, X_train, y_train, X_val, y_val, num_of_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, task_type='GPU', verbose_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True)
        Trains and validates the model on provided datasets.

    build_model(self, X_train, y_train, X_test, y_test, num_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, 
                    task_type='GPU', verbose_flag=False, ts=None, splitpercent=0.80, splittime=None, split_random=True)
        Builds and trains the GBT model.

    predict(self, X_test, **kwargs)
        Makes predictions using the trained model.

    evaluate_model(self, X_train, y_train, X_test, y_test, verbose=False, opt_thresh_comp="from_ROC")
        Evaluates the model's performance.

    compute_optimal_threshold_from_ROC(self, tpr, fpr, thresholds)
        Computes optimal classification threshold from ROC curve.

    compute_optimal_threshold_from_prec_recall(self, precision, recall, thresholds)
        Computes optimal classification threshold from precision-recall curve.

    get_global_feature_importances(self, verbose=True, n_imp_feats=30, feats_to_highlight=None)
        Retrieves global feature importances.

    get_most_important_cat_features(self, imp_feats, n_imp_feats=30, verbose=False)
        Identifies important categorical features.

    print_feature_importances(self, imp, title, feats_to_highlight=None)
        Displays feature importances in a formatted table.

    get_local_feature_importances(self, X_train, q, n=4, plot_flag=False, verbose=False, feats_to_highlight=None)
        Retrieves local feature importances using SHAP values.

    get_shap_values(self, predict_fn, q, forced=False, explainer_type="treeexplainer")
        Computes SHAP values for local interpretability.

    compute_partial_vars(self, model_obj, max_index)
        Computes partial variances for sensitivity analysis.

    get_average_shap_values(self, shap_values)
        Computes average SHAP values for feature importance.

    get_max_shap_values(self, shap_values)
        Computes maximum SHAP values for feature importance.

    get_feats_with_top_n_mixed_impacts(self, n, local_av_imp, local_max_imp)
        Identifies features with top n mixed importances from average and maximum SHAP values.
    
    plot_correlation_between_numeric_features(self, X_train, feats)
        Plots correlation matrix between numeric features.
    
    plot_trees(self, X_train, y_train, n_est)
        Plots the decision trees of the GBT model.
    
    scatterplot_gbt(self, X_train, y_train, feats, threshold, outlier=False, hue=None)
        Plots scatter plots for GBT model features.
    
    get_gbt_model_rules(self, gbt_model, k=None)
        Exports scikit-learn GBT model rules to function.
    
    get_gbt_feature_importance(self, X_train, importances)
        Computes GBT feature importances.
    
    build_lreg_model(self, splitpercent=0.66, splittime=None, verbose=True,
                         split_random=True, solver='newton-cg')
        Builds and evaluates a logistic regression model.
    
    get_lreg_coefficients(lreg_model, cols, regression=False)
        Retrieves coefficients from a logistic regression model.
    
    to_jsonld(self, model_id: str)
        Exports the GBT model metadata to JSON-LD format.
    '''

    def __init__(self, Q, QoI, ts=None, threshold=None, model_type="regression", gbt_method="xgboost", splitpercent=0.80, splittime=None, split_random=True, important=None):  # fmt: skip
        '''
        Initializes the GBTModel with specified parameters.

        Parameters:
        -----------
        Q : VariableSet
            Variable set defining the input features.
        QoI : str
            Name of the quantity of interest (target variable).
        ts : str, optional
            Name of the time series column (default is None).
        threshold : float or None, optional
            Threshold for classification tasks (default is None).
        model_type : str, optional
            Type of model ('regression' or 'classification', default is 'regression').
        gbt_method : str, optional
            GBT implementation method ('scikit', 'xgboost', 'catboost', 'lightgbm', 'ElasticNet', 'log_reg', default is 'xgboost').
        splitpercent : float, optional
            Percentage of data to use for training (default is 0.80).
        splittime : any, optional
            Time value to split data (default is None).
        split_random : bool, optional
            Whether to split data randomly (default is True).
        important : list or None, optional
            List of important features to highlight (default is None).
        '''

        self.Q = Q
        self.cols = self.Q.param_names()
        self.label = QoI
        self.method = gbt_method
        self.model_type = model_type
        self.threshold = threshold
        self.important = important

    def model_dependent_data_preparation(self, d, cols, catcols, method=None):
        '''
        Prepares data based on the selected GBT method.

        Parameters:
        -----------
        d : pd.DataFrame
            Input data.
        cols : list
            List of feature names.
        catcols : list
            List of categorical feature names.
        method : str, optional
            GBT implementation method (default is None, uses self.method).

        Returns:
        -------
        d : pd.DataFrame
            Prepared data.
        cols : list
            Updated list of feature names.
        '''

        if method == None:
            method = self.method

        if method == "scikit" or method == "xgboost":
            # One hot encoding
            # Features to be trained without any categorical ones
            extended_cols = list(set(cols) - set(catcols))
            # One-hot encoding
            if catcols != []:
                ohe = pd.get_dummies(d[catcols])
                extended_cols.extend(list(ohe.columns))
                # Join dataframes
                d = pd.concat([d, ohe], axis=1)
                cols = extended_cols
                # Fill Nans
                d[cols] = d[cols].fillna(0)
                # Change name of categorical features to ohe feature names
                self.catcols = list(ohe.columns)
        elif method == "catboost":
            # convert categories to strings
            for i_col in catcols:
                d[i_col] = d[i_col].astype(str).astype('category')

        return d, cols

    def split_test_and_train_data(self, d, cols, label, threshold, splitpercent, ts=None, splittime=None,
                                  resampling=False, split_random=True, verbose_flag=False):
        '''
        Splits data into training and testing sets.

        Parameters:
        -----------
        d : pd.DataFrame
            Input data.
        cols : list
            List of feature names.
        label : str
            Name of the quantity of interest (target variable).
        threshold : float or None
            Threshold for classification tasks.
        splitpercent : float
            Percentage of data to use for training.
        ts : str, optional
            Name of the time series column (default is None).
        splittime : any, optional
            Time value to split data (default is None).
        resampling : bool, optional
            Whether to apply resampling for class imbalance (default is False).
        split_random : bool, optional
            Whether to split data randomly (default is True).
        verbose_flag : bool, optional
            Whether to print verbose output (default is False).

        Returns:
        -------
        X_train : pd.DataFrame
            Training feature set.
        X_test : pd.DataFrame
            Testing feature set.
        y_train : pd.Series
            Training target variable.
        y_test : pd.Series
            Testing target variable.
        '''

        if split_random == False:
            if splittime == None:
                split_row = int(d.shape[0] * splitpercent)
                splittime = d[ts].iloc[split_row]
            X_train = d[(d[ts] < splittime) & (d[label] == d[label])][cols]
            X_test = d[(d[ts] >= splittime) & (d[label] == d[label])][cols]
            if threshold == None:  # regression
                y_train = d[(d[ts] < splittime) & (d[label] == d[label])][label]
                y_test = d[(d[ts] >= splittime) & (d[label] == d[label])][label]
            else:  # classification
                y_train = d[(d[ts] < splittime) & (d[label] == d[label])][label] > threshold
                y_test = d[(d[ts] >= splittime) & (d[label] == d[label])][label] > threshold
        else:
            X_train, X_test, y_train, y_test = train_test_split(d[cols], d[label], test_size=1 - splitpercent,
                                                                random_state=42)
        return X_train, X_test, y_train, y_test


    def train_model(self, d_X, d_y, n_est=150, max_d=3, learning_rate=0.15, k_fold=None, num_leaves=None, 
                    plot_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True,
                    task_type='GPU', verbose_flag=False):
        '''
        Trains the GBT model with optional k-fold cross-validation.

        Parameters:
        -----------
        d_X : pd.DataFrame
            Feature set.
        d_y : pd.DataFrame
            Target variable.
        n_est : int, optional
            Number of estimators (default is 150).
        max_d : int, optional
            Maximum depth of trees (default is 3).
        learning_rate : float, optional
            Learning rate (default is 0.15).
        k_fold : int or None, optional
            Number of folds for cross-validation (default is None).
        num_leaves : int or None, optional
            Number of leaves (default is None).
        plot_flag : bool, optional
            Whether to plot training progress (default is True).
        ts : str, optional
            Name of the time series column (default is None).
        splitpercent : float, optional
            Percentage of data to use for training (default is 0.80).
        splittime : any, optional
            Time value to split data (default is None).
        split_random : bool, optional
            Whether to split data randomly (default is True).
        task_type : str, optional
            Task type for CatBoost ('CPU' or 'GPU', default is 'GPU').
        verbose_flag : bool, optional
            Whether to print verbose output (default is False).

        Returns:
        -------
        model : object
            Trained GBT model.
        '''

        if type(d_X) != pd.DataFrame:
            d_X = pd.DataFrame(d_X)
        if type(d_y) != pd.DataFrame:
            d_y = pd.DataFrame(d_y)
        if k_fold:
            # Perform k-fold cross-validation
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            fold = 1
            fold_results = []
            for train_indices, val_indices in kf.split(d_X):
                print(f"Training Fold {fold}/{k_fold}...")
                X_train, X_val = d_X.iloc[train_indices], d_X.iloc[val_indices]
                y_train, y_val = d_y.iloc[train_indices], d_y.iloc[val_indices]
                
                # Call build_model for each fold
                model = self.build_model(X_train, y_train, n_est, max_d, learning_rate, 
                                         task_type, verbose_flag)
                
                # Validate the model on the validation set
                predictions = model.predict(X_val)
                if self.model_type == "regression":
                    mse = mean_squared_error(y_val, predictions)
                    print(f"Fold {fold} MSE: {mse:.4f}")
                    fold_results.append(mse)
                # Extend with classification evaluation if needed
                fold += 1
            
            # Report average performance
            print(f"Average MSE across folds: {np.mean(fold_results):.4f}")
        else:
            # Default 80-20 split
            split_index = int(len(d_X) * 0.8)
            X_train, X_val = d_X.iloc[:split_index], d_X.iloc[split_index:]
            y_train, y_val = d_y.iloc[:split_index], d_y.iloc[split_index:]
            
            print("Training with default 80-20 split...")
            model = self.build_model(X_train, y_train, n_est, max_d, learning_rate, num_leaves, plot_flag, 
                    task_type, verbose_flag, ts, splitpercent, splittime, split_random)
            
            predictions = model.predict(X_val)
            if self.model_type == "regression":
                mse = mean_squared_error(y_val, predictions)
                print(f"Validation MSE: {mse:.4f}")
                
        return model
      
    def train_and_validate(self, X_train, y_train, X_val, y_val, num_of_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, task_type='GPU', verbose_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True):
        '''
        Trains and validates the model on provided datasets.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.  
        y_train : pd.DataFrame
            Training target variable.
        X_val : pd.DataFrame
            Validation feature set.
        y_val : pd.DataFrame
            Validation target variable.
        num_of_iter : int, optional
            Number of iterations (default is 150).
        max_depth : int, optional
            Maximum depth of trees (default is 3).
        learning_rate : float, optional
            Learning rate (default is 0.15).
        num_leaves : int or None, optional
            Number of leaves (default is None).
        plot_flag : bool, optional
            Whether to plot training progress (default is True).
        task_type : str, optional
            Task type for CatBoost ('CPU' or 'GPU', default is 'GPU').
        verbose_flag : bool, optional
            Whether to print verbose output (default is True).
        ts : str, optional
            Name of the time series column (default is None).
        splitpercent : float, optional
            Percentage of data to use for training (default is 0.80).
        splittime : any, optional
            Time value to split data (default is None).
        split_random : bool, optional
            Whether to split data randomly (default is True).

        Returns:
        -------
        mse_tr : float
            Mean squared error on the training set.
        mse_vl : float
            Mean squared error on the validation set.
        '''

        max_d = max_depth
        n_est = num_of_iter
        
        self.hyper_params = {
            'num_iter': num_of_iter,
            'max_depth': max_d,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves
        }
        
        # Train model
        if self.method == 'scikit':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                  max_leaf_nodes=num_leaves)
            elif self.model_type == "classification":  # classification
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                   max_leaf_nodes=num_leaves)
            # fit model
            model.fit(X_train, y_train)

        elif self.method == 'xgboost':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_depth=max_d)
                else:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_leaves=num_leaves, grow_policy='lossguide')
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                          max_leaf_nodes=num_leaves)
                else:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_leaf_nodes=num_leaves,
                                          grow_policy='lossguide')
            # fit model
            model.fit(X_train, y_train, verbose=verbose_flag)

        elif self.method == 'catboost':
            cols = list(X_train.columns)
            self.catcols_ind = []
            pool_train = Pool(X_train, y_train, cat_features=self.catcols_ind, feature_names=cols)
            pool_test = Pool(X_val, y_val, cat_features=self.catcols_ind, feature_names=cols)
            # Initialize CatBoostClassifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = CatBoostRegressor(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                              loss_function="RMSE", eval_metric="MAE", use_best_model=True, verbose=0,
                                              task_type=task_type)
                else:
                    model = CatBoostRegressor(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                              n_estimators=n_est, max_leaves=num_leaves, loss_function="RMSE",
                                              eval_metric="MAE", use_best_model=True, verbose=0, task_type=task_type)
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = CatBoostClassifier(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                               loss_function="Logloss", use_best_model=True, verbose=0,
                                               task_type=task_type)
                else:
                    model = CatBoostClassifier(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                               n_estimators=n_est, max_leaves=num_leaves, loss_function="Logloss",
                                               use_best_model=True, verbose=0, task_type=task_type)
            # Fit model
            model.fit(pool_train, eval_set=pool_test, plot=plot_flag, verbose=verbose_flag)

        elif self.method == "lightgbm":
            assert not any(' ' in x for x in X_train.columns), 'No space allowed in column names with LightGBM.' 
            pool_train = lgb.Dataset(X_train, label=y_train)  # , categorical_feature=categorical)
            pool_test = lgb.Dataset(X_val, label=y_val)
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                hyper_params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'num_leaves': num_leaves,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 20,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
                self.hyper_params = hyper_params
            elif self.model_type == "classification":  # classification
                hyper_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': num_leaves,
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
                self.hyper_params = hyper_params
            else:
                raise ValueError
            model = lgb.train(hyper_params, pool_train, valid_sets=pool_test, num_boost_round=5000,
                              callbacks=[lgb.early_stopping(stopping_rounds=50)])
        elif self.method == 'ElasticNet':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = ElasticNet().fit(X_train, y_train)
            elif self.model_type == "classification":  # classification
                raise ('Use ElasticNet with regression or logistic regression!')
        elif self.method == "log_reg":
            model = self.build_lreg_model()
        else:
            raise ('Unknown method: ', self.method)
        self.model = model   
        self.prediction = model.predict
        y_tr_pred = self.predict(X_train)
        y_vl_pred = self.predict(X_val)
        mse_tr = mean_squared_error(y_train, y_tr_pred)
        mse_vl = mean_squared_error(y_val, y_vl_pred)
        return mse_tr, mse_vl
    
    
    def build_model(self, X_train, y_train, X_test, y_test, num_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, 
                    task_type='GPU', verbose_flag=False, ts=None, splitpercent=0.80, splittime=None, split_random=True):
        '''
        Builds and trains the GBT model.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame
            Training target variable.
        X_test : pd.DataFrame
            Testing feature set.
        y_test : pd.DataFrame
            Testing target variable.
        num_iter : int, optional
            Number of iterations (default is 150).
        max_depth : int, optional
            Maximum depth of trees (default is 3).
        learning_rate : float, optional
            Learning rate (default is 0.15).
        num_leaves : int or None, optional
            Number of leaves (default is None).
        plot_flag : bool, optional
            Whether to plot training progress (default is True).
        task_type : str, optional
            Task type for CatBoost ('CPU' or 'GPU', default is 'GPU').
        verbose_flag : bool, optional
            Whether to print verbose output (default is False).
        ts : str, optional
            Name of the time series column (default is None).
        splitpercent : float, optional
            Percentage of data to use for training (default is 0.80).
        splittime : any, optional
            Time value to split data (default is None).
        split_random : bool, optional
            Whether to split data randomly (default is True).

        Returns:
        -------
        model : object
            Trained GBT model.
        '''

        max_d = max_depth
        n_est = num_iter
        
        # Train model
        if self.method == 'scikit':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                  max_leaf_nodes=num_leaves)
            elif self.model_type == "classification":  # classification
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                   max_leaf_nodes=num_leaves)
            # fit model
            model.fit(X_train, y_train)

        elif self.method == 'xgboost':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_depth=max_d)
                else:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_leaves=num_leaves, grow_policy='lossguide')
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                          max_leaf_nodes=num_leaves)
                else:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_leaf_nodes=num_leaves,
                                          grow_policy='lossguide')
            # fit model
            #model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose_flag)
            model.fit(X_train, y_train, verbose=verbose_flag)

        elif self.method == 'catboost':
            cols = list(X_train.columns)
            self.catcols_ind = []
            pool_train = Pool(X_train, y_train, cat_features=self.catcols_ind, feature_names=cols)
            pool_test = Pool(X_test, y_test, cat_features=self.catcols_ind, feature_names=cols)
            # Initialize CatBoostClassifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = CatBoostRegressor(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                              loss_function="RMSE", eval_metric="MAE", use_best_model=True, verbose=0,
                                              task_type=task_type)
                else:
                    model = CatBoostRegressor(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                              n_estimators=n_est, max_leaves=num_leaves, loss_function="RMSE",
                                              eval_metric="MAE", use_best_model=True, verbose=0, task_type=task_type)
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = CatBoostClassifier(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                               loss_function="Logloss", use_best_model=True, verbose=0,
                                               task_type=task_type)
                else:
                    model = CatBoostClassifier(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                               n_estimators=n_est, max_leaves=num_leaves, loss_function="Logloss",
                                               use_best_model=True, verbose=0, task_type=task_type)
            # Fit model
            model.fit(pool_train, eval_set=pool_test, plot=plot_flag, verbose=verbose_flag)

        elif self.method == "lightgbm":
            assert not any(' ' in x for x in X_train.columns), 'No space allowed in column names with LightGBM.' 
            pool_train = lgb.Dataset(X_train, label=y_train)  # , categorical_feature=categorical)
            pool_test = lgb.Dataset(X_test, label=y_test)
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                hyper_params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'num_leaves': num_leaves,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 20,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            elif self.model_type == "classification":  # classification
                hyper_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': num_leaves,
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            else:
                raise ValueError
            model = lgb.train(hyper_params, pool_train, valid_sets=pool_test, num_boost_round=5000,
                              callbacks=[lgb.early_stopping(stopping_rounds=50)])
        elif self.method == 'ElasticNet':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = ElasticNet().fit(X_train, y_train)
            elif self.model_type == "classification":  # classification
                raise ('Use ElasticNet with regression or logistic regression!')
        elif self.method == "log_reg":
            model = self.build_lreg_model()
        else:
            raise ('Unknown method: ', self.method)
        self.model = model
        return model

    def predict(self, X_test, **kwargs):
        '''
        Makes predictions using the trained model.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing feature set.
        **kwargs : dict
            Additional keyword arguments for the prediction function.

        Returns:
        -------
        pred : np.ndarray
            Predicted values.
        '''

        pred = self.prediction(X_test)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred
    
    def evaluate_model(self, X_train, y_train, X_test, y_test, verbose=False, opt_thresh_comp="from_ROC"):
        '''
        Evaluates the model's performance.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame
            Training target variable.
        X_test : pd.DataFrame
            Testing feature set.
        y_test : pd.DataFrame
            Testing target variable.
        verbose : bool, optional
            Whether to print verbose output (default is False).
        opt_thresh_comp : str, optional
            Method to compute optimal threshold for classification ('from_ROC' or 'from_prec', default is 'from_ROC').

        Returns:
        -------
        results : list
            List containing a DataFrame of evaluation metrics and a dictionary of model evaluation results.
        '''

        if self.model_type == "classification":  # For classification problem

            if self.method == "lightgbm":
                pred = self.model.predict(X_test)
            else:
                pred = self.model.predict_proba(X_test)[:, 1]

            # Get optimal value of threshold
            fpr, tpr, thresh_ROC = roc_curve(y_test, pred)
            precision, recall, thresh_prec = precision_recall_curve(y_test, pred)

            if opt_thresh_comp == "from_ROC":
                opt_thresh = self.compute_optimal_threshold_from_ROC(tpr, fpr, thresh_ROC)
            elif opt_thresh_comp == "from_prec":
                opt_thresh = self.compute_optimal_threshold_from_prec_recall(precision, recall, thresh_prec)
            else:
                raise ("There is no optimal threshold method {}. Possible options are from_ROC or from_prec".format(
                    opt_thresh_comp))

            # Get predicted classes

            pred_class = pred > opt_thresh

            # if self.method == "catboost":
            #    pred_class = pred_class == "True"
            # evaluate model
            model_eval = {
                "auc": roc_auc_score(y_test, pred),
                "f1_score": f1_score(np.array(y_test), pred_class)
            }

            if verbose:
                plot_ROC_and_recall_curve = True
                plot_conf_matrix = True
                print(model_eval)
            else:
                plot_ROC_and_recall_curve = False
                plot_conf_matrix = False

            if plot_ROC_and_recall_curve:
                # Plot ROC CURVE
                plt.figure(figsize=(10, 3))
                plt.subplot(1, 2, 1)
                plt.plot(fpr, tpr)
                plt.title("ROC curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                # Plot recall curve
                plt.subplot(1, 2, 2)
                plt.plot(recall, precision)
                plt.title("Precision-recall curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.show()

            if plot_conf_matrix:
                # Print confusion matrix
                print("Confusion Matrix:")
                conf_matrix = pd.DataFrame(confusion_matrix(y_test, pred_class),
                                           columns=["predicted GOOD quality", "predicted BAD quality"],
                                           index=["actual GOOD quality", "actual BAD quality"])
                display(conf_matrix)


        elif self.model_type == "regression":  # for regression problem
            pred = self.model.predict(X_test)
            model_eval = {
                "Kendall_tau": kendalltau(y_test, pred)[0],
                "MSE": mean_squared_error(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "STD_of_label": self.y_train.std()
            }
            if verbose == True:
                print('Kendall tau correlation - measure of correspondance between two rankings: %.3f' % model_eval["Kendall_tau"])
                #print('Pearson correlation - measure of linear realationship (cov normalised): %.3f' % model_eval["Pearson"])
                #print('Spearman correaltion - cov(rank(y1), rank(y2)/stdv(rank(y1))): %.3f' % model_eval["Spearman"])
                print("mean squared error: ", model_eval["MSE"])
                print("Root mean squared error: ", model_eval["RMSE"]) 
                print("STD of label:", model_eval["STD_of_label"])
                print("MAE: ", model_eval["MAE"])
        # pdb.set_trace()
        df = pd.DataFrame(model_eval, index=[0])
        df['X_train_cols'] = [X_train.columns]
        df['label'] = [y_train.columns] #name
        return [df, model_eval]

    def compute_optimal_threshold_from_ROC(self, tpr, fpr, thresh):
        '''
        Computes the optimal threshold from ROC curve.

        Parameters:
        -----------
        tpr : np.ndarray
            True positive rates.
        fpr : np.ndarray
            False positive rates.
        thresh : np.ndarray
            Thresholds.

        Returns:
        -------
        ROC_thresh : float
            Optimal threshold based on ROC curve.    
        '''

        J = tpr - fpr
        ix = np.argmax(J)
        ROC_thresh = thresh[ix]
        return ROC_thresh

    def compute_optimal_threshold_from_prec_recall(self, precision, recall, thresh):
        '''
        Computes the optimal threshold from precision-recall curve.

        Parameters:
        -----------
        precision : np.ndarray
            Precision values.
        recall : np.ndarray
            Recall values.
        thresh : np.ndarray
            Thresholds.

        Returns:
        -------
        prec_thresh : float
            Optimal threshold based on precision-recall curve.
        '''

        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        prec_thresh = thresh[ix]
        return prec_thresh

    def get_global_feature_importances(self, verbose=True, n_imp_feats=30, feats_to_highlight=None):
        '''
        Computes and returns global feature importances.

        Parameters:
        -----------
        verbose : bool, optional
            Whether to print verbose output (default is True).
        n_imp_feats : int, optional
            Number of top important features to consider (default is 30).
        feats_to_highlight : list or str or None, optional
            Features to highlight in the output (default is None).

        Returns:
        -------
        global_imp : pd.Series
            Series containing global feature importances.
        imp_feats : list
            List of important features.
        imp_cat_feats : list
            List of important categorical features.
        '''

        # Pandas series with important features and their importances
        if self.method == "lightgbm":
            global_imp = self.get_gbt_feature_importance(self.model.feature_importance(importance_type='gain'))
        else:
            global_imp = self.get_gbt_feature_importance(self.model.feature_importances_)
        # Important features
        imp_feats = global_imp.index
        # Get most important categorical features
        imp_cat_feats = self.get_most_important_cat_features(imp_feats, n_imp_feats=n_imp_feats, verbose=verbose)
        # Print global importance list
        if verbose:
            self.print_feature_importances(global_imp, "Global feature importances",
                                           feats_to_highlight=feats_to_highlight)
        return global_imp, imp_feats, imp_cat_feats

    def get_most_important_cat_features(self, imp_feats, n_imp_feats=30, verbose=False):
        '''
        Identifies the most important categorical features.

        Parameters:
        -----------
        imp_feats : list
            List of important features.
        n_imp_feats : int, optional
            Number of top important features to consider (default is 30).
        verbose : bool, optional
            Whether to print verbose output (default is False).

        Returns:
        -------
        imp_cat_feats : list
            List of important categorical features.
        '''

        imp_cat_feats = list(set(imp_feats[:n_imp_feats]).intersection(set(self.catcols)))
        if verbose:
            if (len(imp_cat_feats) > 0):
                print("Categorical features in the top {} importance list:".format(n_imp_feats))
                print(imp_cat_feats)
            else:
                print("There is no categorical feature in the top {} importance list".format(n_imp_feats))

        return imp_cat_feats

    def print_feature_importances(self, imp, title, feats_to_highlight=None):
        '''
        Prints feature importances in a formatted table.

        Parameters:
        -----------
        imp : pd.Series
            Series containing feature importances.
        title : str
            Title for the importance table.
        feats_to_highlight : list or str or None, optional
            Features to highlight in the output (default is None).

        Returns:
        -------
        None
        '''
        print("")
        print(title)
        imp_df = pd.DataFrame()
        imp_df["feature"] = imp.index
        imp_df["importance"] = imp.values
        pd.options.display.max_colwidth = 100
        pd.options.display.max_rows = 200
        if feats_to_highlight == None:
            imp_form = imp_df.style.apply(
                lambda x: ['background-color: lightblue' if x.feature in self.catcols else '' for i in x], axis=1)
            display(imp_form)
        elif isinstance(feats_to_highlight, str):
            imp_form = imp_df.style.apply(lambda x: [
                'background-color: lightblue' if x.feature in self.catcols else 'background-color: orange' if (
                            x.feature in feats_to_highlight) else "" for i in x], axis=1)
            display(imp_form)
        else:  # if feats_to_highlight is a list
            imp_form = imp_df.style.apply(lambda x: ['background-color: orange' if any([x.feature in f for f in
                                                                                        feats_to_highlight]) else 'background-color: lightblue' if x.feature in self.catcols else ""
                                                     for i in x], axis=1)
            display(imp_form)

    def get_local_feature_importances(self, X_train, q, n=4, plot_flag=False, verbose=False, feats_to_highlight=None):
        '''
        Computes and returns local feature importances using SHAP values.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        q : pd.DataFrame
            Query points for which local feature importances are computed.
        n : int, optional
            Number of top important features to consider (default is 4).
        plot_flag : bool, optional
            Whether to plot SHAP summary plots (default is False).
        verbose : bool, optional
            Whether to print verbose output (default is False).
        feats_to_highlight : list or str or None, optional
            Features to highlight in the output (default is None).

        Returns:
        -------
        shap_values : np.ndarray
            Array of SHAP values.
        local_av_imp : pd.Series
            Series containing average local feature importances.
        local_max_imp : pd.Series
            Series containing maximum local feature importances.
        feats_with_top_n_mixed_imp : list
            List of features with top n mixed importances.                
        '''

        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(q)
        if isinstance(shap_values, list) and len(
                shap_values) == 2:  # if shap gives an array for both classes (True and False)
            shap_values = shap_values[1]
        local_av_imp = self.get_average_shap_values(shap_values)
        local_max_imp = self.get_max_shap_values(shap_values)
        feats_with_top_n_mixed_imp = self.get_feats_with_top_n_mixed_impacts(n, local_av_imp, local_max_imp)
        if plot_flag:
            # Plot summary of average effect of the features
            shap.summary_plot(shap_values, q, plot_type="bar", plot_flag=True, matplotlib=True)
            # Plot summary of local effect of the features
            shap.summary_plot(shap_values, X_train, plot_flag=True)
            plt.show()
        if verbose:
            self.print_feature_importances(local_av_imp, "Average local importances (average SHAP values)",
                                           feats_to_highlight=feats_to_highlight)
            self.print_feature_importances(local_max_imp, "Maximum local importances (max SHAP values)",
                                           feats_to_highlight=feats_to_highlight)
            print("Features with top {} highest local and average impact:".format(n))
            print(feats_with_top_n_mixed_imp)
        return shap_values, local_av_imp, local_max_imp, feats_with_top_n_mixed_imp
    
    def get_shap_values(self, q, forced=False, explainer_type="treeexplainer"):
        '''
        Computes SHAP values for the given query points.

        Parameters:
        -----------
        q : pd.DataFrame
            Query points for which SHAP values are computed.
        forced : bool, optional
            Whether to force re-computation of the explainer (default is False).
        explainer_type : str, optional
            Type of SHAP explainer to use (default is "treeexplainer").

        Returns:
        -------
        shap_values : np.ndarray
            Array of SHAP values.
        '''

        if explainer_type == "treeexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.TreeExplainer(self.model)
                self.explainer = explainer
        shap_values = self.explainer(q)
        return shap_values
    
    def compute_partial_vars(self, model_obj, max_index):
        '''
        Computes partial variances using Sobol sensitivity analysis.
        
        Parameters:
        -----------
        model_obj : object
            Model object containing parameters and prediction methods.
        max_index : int
            Maximum order of Sobol indices to compute (1 or 2).

        Returns:
        -------
        partial_var_df : pd.DataFrame
            DataFrame containing partial variances.
        sobol_index_df : pd.DataFrame
            DataFrame containing Sobol indices.
        y_var : np.ndarray
            Array of variances of the model outputs.
        '''

        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names
        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.get_dist_types(), 'bounds': paramset.get_dist_params()
            }
        q = paramset.sample(method='Sobol_saltelli', n=32768) # saltelli working only for uniform distribution
        # https://salib.readthedocs.io/en/latest/user_guide/advanced.html
        q_df = pd.DataFrame(q, columns=paramset.param_names())
        y = model_obj.predict(q_df)
        
        # Run model
        S1 = []
        S2 = []
        for i in range(y.shape[1]):
            y_i = y[:,i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
        
        y_var = np.broadcast_to(y.var(axis=0).reshape(-1,1), sobol_index.shape)
        partial_variance = np.multiply(sobol_index, y_var)
        
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)
        
        return partial_var_df, sobol_index_df, y_var

    def get_average_shap_values(self, shap_values):
        '''
        Computes average SHAP values for feature importance.

        Parameters:
        -----------
        shap_values : np.ndarray
            Array of SHAP values.

        Returns:
        -------
        local_av_imp : pd.Series
            Series containing average local feature importances.
        '''

        av_shap_values = np.mean(np.abs(shap_values), axis=0)
        local_av_imp = self.get_gbt_feature_importance(av_shap_values)
        return local_av_imp

    def get_max_shap_values(self, shap_values):
        '''
        Computes maximum SHAP values for feature importance.

        Parameters:
        -----------
        shap_values : np.ndarray
            Array of SHAP values.

        Returns:
        -------
        local_max_imp : pd.Series
            Series containing maximum local feature importances.
        '''

        av_shap_values = np.max(np.abs(shap_values), axis=0)
        local_max_imp = self.get_gbt_feature_importance(av_shap_values)
        return local_max_imp

    def get_feats_with_top_n_mixed_impacts(self, n, local_av_imp, local_max_imp):
        '''
        Identifies features with top n mixed importances from average and maximum SHAP values.

        Parameters:
        -----------
        n : int
            Number of top important features to consider.
        local_av_imp : pd.Series
            Series containing average local feature importances.
        local_max_imp : pd.Series
            Series containing maximum local feature importances.

        Returns:
        -------
        feats_with_top_n_mixed_imp : list
            List of features with top n mixed importances.
        '''

        top_n_av = list(local_av_imp.iloc[:n].index)
        top_n_max = list(local_max_imp.iloc[:n].index)
        feats_with_top_n_mixed_imp = list((set(top_n_av)).union(set(top_n_max)))
        return feats_with_top_n_mixed_imp

    def plot_correlation_between_numeric_features(self, X_train, feats):
        '''
        Plots correlation matrix between numeric features.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        feats : list
            List of features to consider for correlation plot.
        
        Returns:
        -------
        None
        '''

        feats = set(feats) - set(self.catcols)
        d = pd.concat([X_train[feats], self.y_train], axis=1)
        corrplot(df=d[feats])

    def plot_trees(self, X_train, y_train, n_est):
        '''
        Plots the decision trees of the GBT model.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame
            Training target variable.
        n_est : int
            Number of estimators (trees) to plot.

        Returns:
        -------
        None
        '''

        cols = list(X_train.columns)
        # n_est
        if self.method == 'catboost':
            pool_train = Pool(X_train, y_train, cat_features=self.catcols_ind, feature_names=cols)
            for i in range(n_est):
                print('Tree', i, ':')
                print(self.model.plot_tree(tree_idx=i, pool=pool_train))
        elif self.method == "scikit":
            self.get_gbt_model_rules(self.model)
        elif self.method == "lightgbm":
            # fig, ax = plt.subplots(n_est, figsize=(10, 30))
            for i in range(n_est):
                # lgb.create_tree_digraph(self.model, tree_index=i)
                lgb.plot_tree(self.model, tree_index=i)
        elif self.method =="xgboost":
            fig, ax = plt.subplots(n_est,figsize=(30, 30))
            for i in range(n_est):
                plot_tree(self.model, num_trees=i, ax=ax[i])
            plt.show()

    def scatterplot_gbt(self, X_train, y_train, feats, threshold, outlier=False):
        '''
        Plots scatter plots for GBT model features.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.DataFrame
            Training target variable.
        feats : list
            List of features to plot.
        threshold : float
            Threshold value for outlier detection.
        outlier : bool, optional
            Whether to highlight outliers (default is False).

        Returns:
        -------
        None
        '''

        d = pd.concat([X_train, y_train], axis=1)
        label = self.y_train.name
        for f in feats:
            scatterplot(d, label, f, outlier, threshold=threshold)
            scatterplot(d, label, f, outlier, threshold=threshold, regressionline=True)


    def get_gbt_model_rules(self, gbt_model, k=None):
        '''
        Exports scikit-learn GBT model rules to function.

        Parameters:
        -----------
        gbt_model : object
            Trained GBT model.
        k : int or None, optional
            Number of trees to consider (default is None, which means all trees).

        Returns:
        -------
        func_str : list
            List of strings representing the function to score using GBT rules.
        '''

        if len(self.cols) == 1:
            # duplication of the same feature name is needed if only 1 was used for GBT training!
            column_names = self.cols * 3
        INDENT = str("    ")
        num_of_trees = len(gbt_model.estimators_)
        if k != None:
            num_of_trees = min(k, num_of_trees)

        func_str = ["def score_gbt(eval_df, model_name):"]
        func_str += [INDENT + '"""Score segment by gbt rule"""']
        func_str += []
        func_str += [INDENT + "def score_by_gbt_tree_rule(row):"]
        func_str += [INDENT + '# GBT model generated by Scikit-Learn']
        func_str += [INDENT + INDENT + "score = 0.0", ]
        for i in range(num_of_trees):
            func_str += [INDENT + INDENT + '### tree_%i ###' % (i + 1)]
            func_str += print_tree_with_names(gbt_model.estimators_[i, 0].tree_, self.cols, INDENT)
            func_str += ['']
        func_str += [INDENT + INDENT + "return score"]
        func_str += []
        func_str += [INDENT + "eval_df['SCORE_'+model_name] = eval_df.apply(score_by_gbt_tree_rule, axis=1)"]
        func_str += [INDENT + "return eval_df"]
        for line in func_str:
            print(line)
        return func_str

    def get_gbt_feature_importance(self, X_train, importances):
        '''
        Computes GBT feature importances.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature set.
        importances : np.ndarray
            Array of feature importances.

        Returns:
        -------
        gbt_all_importances : pd.Series
            Series containing GBT feature importances.
        '''

        gbt_all_importances = pd.Series(importances, index=X_train.columns, name="feature importance").sort_values(
            ascending=False)
        return gbt_all_importances[gbt_all_importances > 0]

    def build_lreg_model(self, splitpercent=0.66, splittime=None, verbose=True,
                         split_random=True, solver='newton-cg'):
        '''
        Builds and evaluates a logistic regression model.

        Parameters:
        -----------
        splitpercent : float, optional
            Percentage of data to use for training (default is 0.66).
        splittime : any, optional
            Time value to split data (default is None).
        verbose : bool, optional
            Whether to print verbose output (default is True).
        split_random : bool, optional
            Whether to split data randomly (default is True).
        solver : str, optional
            Solver to use for logistic regression (default is 'newton-cg').
            
        Returns:
        -------
        auc : float
            Area Under the Curve (AUC) score of the logistic regression model.
        '''

        d = self.df[self.important + [self.label, 'ts']].fillna(0)

        if split_random == False:
            if splittime == None:
                split_row = int(d.shape[0] * splitpercent)
                splittime = d["ts"].iloc[split_row]
            features_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.important]
            features_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.important]
            if self.threshold == None:  # regression
                labels_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.label]
                labels_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.label]
            else:  # classification
                labels_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.label] > self.threshold
                labels_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.label] > self.threshold
        else:
            features_train, features_test, labels_train, labels_test = train_test_split(d[self.important], d[self.label],
                                                                                        test_size=1 - splitpercent,
                                                                                        random_state=42)

        logreg = LogisticRegression(solver=solver)

        logreg.fit(features_train, labels_train)
        pred = logreg.predict_proba(features_test)[:, 1]
        auc = roc_auc_score(labels_test, pred)
        print('Logreg model for', self.label, 'AUC =', auc)
        if verbose:
            pretty_print_df(df=self.get_lreg_coefficients(logreg, self.important))
        return auc

    @staticmethod
    def get_lreg_coefficients(lreg_model, cols, regression=False):
        '''
        Retrieves coefficients from a logistic regression model.

        Parameters:
        -----------
        lreg_model : object
            Trained logistic regression model.
        cols : list
            List of feature names.
        regression : bool, optional
            Whether the model is for regression (default is False).

        Returns:
        -------
        coef_df : pd.DataFrame
            DataFrame containing feature names and their coefficients.
        '''

        intercept_df = pd.DataFrame({"name": "(intercept)", "value": lreg_model.intercept_})
        if regression:
            coef_df = pd.DataFrame({"name": cols, "value": lreg_model.coef_}).sort_values(by='value')
        else:
            coef_df = pd.DataFrame({"name": cols, "value": lreg_model.coef_[0]}).sort_values(by='value')
        return intercept_df.append(coef_df, ignore_index=True)

    def to_jsonld(self, model_id: str):
        '''
        Exports the GBT model metadata to JSON-LD format.
        
        Parameters:
        -----------
        model_id : str
            Unique identifier for the model.

        Returns:
        -------
        jsonld : dict
            JSON-LD representation of the model metadata.
        '''

        match self.method:
            case 'scikit':
                algorithm = 'https://en.wikipedia.org/wiki/Gradient_boosting'
            case 'xgboost':
                algorithm = 'https://en.wikipedia.org/wiki/XGBoost'
            case 'catboost':
                algorithm = 'https://en.wikipedia.org/wiki/CatBoost'
            case 'lightgbm':
                algorithm = 'https://en.wikipedia.org/wiki/LightGBM'
            case 'ElasticNet':
                algorithm = 'https://en.wikipedia.org/wiki/Elastic_net_regularization'
            case 'log_reg':
                algorithm = 'https://en.wikipedia.org/wiki/Logistic_regression'

        jsonld = {

            "@context": {
                "mls": "https://ml-schema.github.io/documentation/mls.html",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },

            "@id": f"https://example.org/models/{model_id}",
            "@type": "mls:Model",
            "mls:implementsAlgorithm": {
                "@id": algorithm,
                "@type": "mls:Algorithm",
                "rdfs:label": self.method,
            },

            "mls:hasHyperParameter": [
                {
                    "@type": "mls:HyperParameterSetting",
                    "mls:hasParameterName": name,
                    "mls:hasParameterValue": str(value).lower() if isinstance(value, bool) else str(value)
                }
                for name, value in self.hyper_params.items()

            ],

            "mls:hasInput": [
                {
                    "@type": "mls:Feature",
                    "mls:featureName": name,
                    "mls:hasDistribution": {
                        "@type": "mls:Distribution",
                        "mls:distributionType": dist.get_type(),
                        "mls:params": str(dist.dist_params),
                    }
                }
                for (name, dist) in self.Q.params.items()
            ]
        }
    
        return jsonld