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
        """ Trains the GBT model with optional k-fold cross-validation.

            Parameters
            ----------
            q_train : array_like or pandas.DataFrame
                Training input features.
            y_train : array_like or pandas.DataFrame
                Training target features.
            k_fold : int, optional
                Number of folds for cross-validation. Default is None (no cross-validation).
        """
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
        """ Trains the GBT model on the training set and validates it on the validation set.

            Parameters
            ----------
            X_train : array_like or pandas.DataFrame
                Training input features.
            y_train : array_like or pandas.DataFrame
                Training target features.
            X_val : array_like or pandas.DataFrame
                Validation input features.
            y_val : array_like or pandas.DataFrame
                Validation target features.

            Returns
            -------
            train_loss : float
                Mean squared error on the training set.
            val_loss : float
                Mean squared error on the validation set.
        """

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
        """ Predicts the target features for the given input features.

            Parameters
            ----------
            X : array_like or pandas.DataFrame
                Input features for prediction.

            Returns
            -------
            pred : array_like
                Predicted target features.
        """

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
        """ Computes the Sobol sensitivity indices for the model.

            Parameters
            ----------
            max_index : int, optional
                Maximum order of Sobol indices to compute. Default is 1.

            Returns
            -------
            partial_var_df : pd.DataFrame
                DataFrame containing partial variances.
            sobol_index_df : pd.DataFrame
                DataFrame containing Sobol sensitivity indices.
        """
        if max_index > 2:
            print(f'Warning: The maximum value of max_index for GBTModel is 2. It is automatically set to 2.')
            max_index = 2
        partial_var_df, sobol_index_df = super(GBT, self).get_sobol_sensitivity(max_index)
        return partial_var_df, sobol_index_df

    def plot_sobol_sensitivity(self, y_train, max_index=1, param_name=None):
        """ Plots the Sobol sensitivity indices for the model.

            Parameters
            ----------
            y_train : array_like or pandas.DataFrame
                Training target features.
            max_index : int, optional
                Maximum order of Sobol indices to plot. Default is 1.
            param_name : str, optional
                Name of the parameter to highlight in the plot.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object containing the Sobol sensitivity plot.
        """
        if max_index > 2:
            print(f'Warning: The maximum value of max_index for GBTModel is 2. It is automatically set to 2.')
            max_index = 2
        fig = super(GBT, self).plot_sobol_sensitivity(y_train, max_index, param_name)
        return fig

    def get_orig(self, scaled):
        """ Inversely transforms the scaled features back to their original scale.

            Parameters
            ----------
            scaled : array_like
                Scaled features to be transformed.

            Returns
            -------
            original : array_like
                Original features.
        """
        original = self.scaler.inverse_transform(scaled)
        return original

    def get_scaled(self, original):
        """ Transforms the original features to their scaled representation.

            Parameters
            ----------
            original : array_like
                Original features to be scaled.

            Returns
            -------
            scaled : array_like
                Scaled features.
        """
        scaled = self.scaler.transform(original)
        return scaled
