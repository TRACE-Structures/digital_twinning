from .PredictiveModel import PredictiveModel

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from .utils import get_scaler
import shap


class GBT(PredictiveModel):
    '''
    Gradient Boosted Trees (GBT) model class for regression and classification tasks.
    Supports multiple GBT implementations: scikit-learn, XGBoost, CatBoost, and LightGBM.
    Includes methods for training, prediction, evaluation, and feature importance analysis.
    
    Attributes:
    -----------
    Q : ParameterSet
        Parameter set defining the input features.

    QoI_names : str
        Names of the quantities of interest (target variables).
            
    model_type : str
        Type of model ('regression' or 'classification').
        
    model : object
        Trained GBT model.
    
    prediction : function
        Prediction function of the trained model. '''

    def __init__(self, Q, QoI_names, num_of_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None):
        '''
        Initializes the GBTModel with specified parameters.

        Parameters:
        -----------
        Q : VariableSet
            Variable set defining the input features.
        QoI_names : str
            Name of the quantity of interest (target variable).
        model_type : str, optional
            Type of model ('regression' or 'classification', default is 'regression'). '''

        self.Q = Q
        self.QoI_names = QoI_names

        if num_leaves == None:
            model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=num_of_iter,
                                    max_depth=max_depth)
        else:
            model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=num_of_iter,
                                    max_leaves=num_leaves, grow_policy='lossguide')

        self.model = model


    def train(self, q_train, y_train, k_fold=None):
        self.scaler = get_scaler("identity")

        xi = self.get_scaled(q_train)
        yt = self.get_scaled(y_train)

        print(f'----- Training started for \'GBTModel\' model -----')

        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold)
        else:
            # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = self.train_test_split(xi, yt, train_test_ratio=0.8, random_seed=1997, split_type='shuffle')
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for GBTModel model -----')



    def train_and_validate(self, X_train, y_train, X_val, y_val):

        X_train, X_val = pd.DataFrame(X_train, columns=self.Q.variable_names()), pd.DataFrame(X_val, columns=self.Q.variable_names())
        y_train, y_val = pd.DataFrame(y_train, columns=self.QoI_names), pd.DataFrame(y_val, columns=self.QoI_names)                 
        
        # fit model
        self.model.fit(X_train, y_train)

        # self.model = model   
        self.prediction = self.model.predict
        y_tr_pred = self.predict(X_train)
        y_vl_pred = self.predict(X_val)
        mse_tr = mean_squared_error(y_train, y_tr_pred)
        mse_vl = mean_squared_error(y_val, y_vl_pred)
        return mse_tr, mse_vl


    def predict(self, X):

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.Q.variable_names())

        X = self.get_scaled(X)
        X = pd.DataFrame(X, columns=self.Q.variable_names())

        pred = self.prediction(X)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        pred = self.get_orig(pred)
        return pred

    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="treeexplainer", silent=False):
        '''
        Computes SHAP values for the given query points.

        Parameters:
        -----------
        predict_fn : callable
            Prediction function for the model.
        q : pd.DataFrame
            Query points for which SHAP values are computed.
        forced : bool, optional
            Whether to force re-computation of the explainer (default is False).
        explainer_type : str, optional
            Type of SHAP explainer to use (default is "treeexplainer").
        silent : bool, optional
            Whether to suppress output (default is False).

        Returns:
        -------
        shap_values : np.ndarray
            Array of SHAP values.
        '''

        if isinstance(q, pd.DataFrame):
            q = q.apply(pd.to_numeric, errors='coerce').fillna(0)

        if explainer_type == "treeexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except (ValueError, Exception):
                    background = q.iloc[:min(50, len(q))]
                    self.explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = self.explainer(q, silent=silent)
        return shap_values

    def get_sobol_sensitivity(self, max_index=1):
        if max_index > 2:
            print(f'Warning: The maximum value of max_index for GBTModel is 2. It is automatically set to 2.')
            max_index = 2
        partial_var_df, sobol_index_df = super(GBT, self).get_sobol_sensitivity(max_index)
        return partial_var_df, sobol_index_df

    def plot_sobol_sensitivity(self, y_train, max_index=1, param_name=None):
        if max_index > 2:
            print(f'Warning: The maximum value of max_index for GBTModel is 2. It is automatically set to 2.')
            max_index = 2
        fig = super(GBT, self).plot_sobol_sensitivity(y_train, max_index, param_name)
        return fig

    def get_orig(self, scaled):
        original = self.scaler.inverse_transform(scaled)
        return original

    def get_scaled(self, original):
        scaled = self.scaler.transform(original)
        return scaled
