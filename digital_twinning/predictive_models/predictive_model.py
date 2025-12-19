'''Module for Predictive Model including Surrogate Model and Data-Driven Model'''
import asyncio
import gPCE_model
from .gbt_model import GBTModel
from .dnn_model import DNNModel
from .linreg_model import LinRegModel

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import FunctionTransformer
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import digital_twinning.utils.utils as utils
import os
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PredictiveModel:
    '''
    Base class for Predictive Models including Surrogate Models and Data-Driven Models
    
    Attributes:
    -----------
    Q : VariableSet
        Parameter set
    QoI_names : list
        List of Quantity of Interest (QoI) names
    method : str
        Method type for predictive model ('DNN', 'gPCE', 'GBT', 'LinReg')
    model : object
        Instance of the specific predictive model class

    Methods:
    ----------
    get_model(self)
        Initializes and returns the predictive model based on the specified method
    
    train_test_split(self, X_data, y_data, train_test_ratio=0.2, random_seed=1997, split_type='shuffle')
        Splits the dataset into training and testing sets
    
    train(self, q_train, y_train, q_scaler=None, y_scaler=None, k_fold=None, **params)
        Trains the predictive model using the provided training data
    
    predict(self, q, **params)
        Predicts QoI values for given input parameters using the trained model
    
    save_model(self, name=None, path=None)
        Saves the predictive model to a file
    
    get_mean_and_var(self, n_sample=1000)
        Computes the mean and variance of the model predictions over a specified number of samples
    
    get_sobol_sensitivity(self, max_index=1)
        Computes Sobol sensitivity indices up to the specified maximum index
    
    get_shap_values(self, q, mean=False, sample_size_from_q=100)
        Computes SHAP values for the model predictions
    
    subtract_effects(self, q, QoI, subtracted_params)
        Subtracts the effects of specified parameters from the QoI using SHAP values
    
    calculate_error_ratios(self, errors, threshold_ratio)
        Calculates error ratios based on a threshold ratio
    
    plot_sobol_sensitivity(self, y_train, max_index=1, **kwargs)
        Plots Sobol sensitivity indices
    
    plot_shap_single_waterfall(self, X_test, **kwargs)
        Plots a single SHAP waterfall plot
    
    plot_shap_multiple_waterfalls(self, X_test, **kwargs)
        Plots multiple SHAP waterfall plots
    
    plot_shap_beeswarm(self, X_test, **kwargs)
        Plots a SHAP beeswarm plot
    
    plot_effects(self, effects, xticks=True)
        Plots the effects of parameters on QoI
    
    create_mx_from_tuple(self, pairs, measured_names)
        Creates a binary matrix from pairs of QoI names and measured variable names
    '''

    def __init__(self, Q, QoI_names, method, **kwargs):
        '''
        Initializes the PredictiveModel with the given parameters.

        Parameters:
        -----------
        Q : VariableSet
            Variable set
        QoI_names : list
            List of Quantity of Interest (QoI) names
        method : str
            Method type for predictive model ('DNN', 'gPCE', 'GBT', 'LinReg')
        **kwargs : dict
            Additional configuration parameters for the model
        '''

        self.Q = Q
        self.QoI_names = QoI_names
        self.method = method
        self.init_config = kwargs
        self.model = self.get_model()
        
    def get_model(self):
        '''
        Initializes and returns the predictive model based on the specified method.

        Returns:
        -------
        model : object
            Instance of the specific predictive model class
        '''

        match self.method:
            case "DNN":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}") # TODO use GPU in dnn_surrogate
                DNN = DNNModel(self.Q.num_params(), len(self.QoI_names), **self.init_config)
                DNN = DNN.double() # modify all datatype of model's parameters to torch.float64
                return DNN
            case "gPCE":
                gPCE = gPCE_model.GpcModel(self.Q, **self.init_config)
                return gPCE
            case "GBT":
                GBT = GBTModel(self.Q, self.QoI_names, **self.init_config)
                return GBT
            case "LinReg":
                LinReg = LinRegModel(self.Q, self.QoI_names)
                return LinReg
            case _:
                raise ValueError(f"There is no method type: {self.method}")
    
    def train_test_split(self, X_data, y_data, train_test_ratio=0.2, random_seed=1997, split_type='shuffle'):
        '''
        Splits the dataset into training and testing sets.

        Parameters:
        -----------
        X_data : pd.DataFrame
            Input features
        y_data : pd.DataFrame
            Target values
        train_test_ratio : float
            Ratio of training data to total data (default is 0.2)
        random_seed : int
            Random seed for reproducibility (default is 1997)
        split_type : str
            Type of split ('shuffle', 'no_shuffle', 'reverse') (default is 'shuffle')

        Returns:
        -------
        X_train, X_test, y_train, y_test : pd.DataFrame
            Split training and testing datasets
        '''

        self.samples = X_data
        self.QoI = y_data        
        if split_type == 'shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=1-train_test_ratio, random_state=random_seed, shuffle=True)
        elif split_type == 'no_shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=1-train_test_ratio, random_state=random_seed, shuffle=False)
        elif split_type == 'reverse':
            X_test, X_train, y_test, y_train = train_test_split(X_data, y_data, train_size=1-train_test_ratio, random_state=random_seed, shuffle=False)
        
        return X_train, X_test, y_train, y_test

    def train(self, q_train, y_train, q_scaler=None, y_scaler=None, k_fold=None, **params):
        '''
        Trains the predictive model using the provided training data.

        Parameters:
        -----------
        q_train : pd.DataFrame
            Training input features
        y_train : pd.DataFrame
            Training target values
        q_scaler : object
            Scaler for input features (default is None)
        y_scaler : object
            Scaler for target values (default is None)
        k_fold : int
            Number of folds for cross-validation (default is None)
        **params : dict
            Additional training parameters

        Returns:
        -------
        None
        '''

        self.train_config = params
        if self.method == "gPCE":
            xi = q_train.values
            yt = y_train.values
        else:
            self.get_q_scaler(q_train, q_scaler)
            self.get_y_scaler(y_train, y_scaler)
            xi = self.get_scaled_q(q_train)
            yt = self.get_scaled_y(y_train) # y transformed
        print(f'----- Training started for \'{self.method}\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold, **params)
        else:
            # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = train_test_split(xi, yt, test_size=0.2, random_state=12)
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for \'{self.method}\' model -----')
        
    def cross_validation(self, xi, yt, k_fold, **params):
        '''
        Performs k-fold cross-validation for model training.

        Parameters:
        -----------
        xi : np.ndarray
            Input features
        yt : np.ndarray
            Target values
        k_fold : int
            Number of folds for cross-validation
        **params : dict
            Additional training parameters

        Returns:
        -------
        train_losses : list
            List of training losses for each fold
        val_losses : list
            List of validation losses for each fold
        '''

        train_losses, val_losses = [], []
        kf = KFold(n_splits=k_fold, shuffle=True)
        for fold, (train_indices, val_indices) in enumerate(kf.split(xi)):
            print(f"Fold {fold + 1}/{k_fold}")
            xi_train, xi_val = xi[train_indices], xi[val_indices]
            yt_train, yt_val = yt[train_indices], yt[val_indices]
            tr_loss, vl_loss = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
            train_losses.append(tr_loss), val_losses.append(vl_loss)
        return train_losses, val_losses
    
    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        '''
        Trains the model on the training data and validates it on the validation data.

        Parameters:
        -----------
        xi_train : np.ndarray
            Training input features
        yt_train : np.ndarray
            Training target values
        xi_val : np.ndarray
            Validation input features
        yt_val : np.ndarray
            Validation target values
        **params : dict
            Additional training parameters

        Returns:
        -------
        tr_loss : float
            Training loss
        vl_loss : float
            Validation loss        
        '''

        match self.method:
            case "DNN":
                xi_tr,xi_vl = torch.tensor(xi_train, dtype=torch.float64), torch.tensor(xi_val, dtype=torch.float64)
                yt_tr,yt_vl = torch.tensor(yt_train, dtype=torch.float64), torch.tensor(yt_val, dtype=torch.float64)
                tr_loss, vl_loss = self.model.train_and_validate(xi_tr, yt_tr, xi_vl, yt_vl, **params)
            case "gPCE":
                tr_loss, vl_loss = self.model.train_and_evaluate(xi_train, yt_train, xi_val, yt_val)
            case "GBT":
                xi_tr, xi_vl = pd.DataFrame(xi_train, columns=self.Q.param_names()), pd.DataFrame(xi_val, columns=self.Q.param_names())
                yt_tr, yt_vl = pd.DataFrame(yt_train, columns=self.QoI_names), pd.DataFrame(yt_val, columns=self.QoI_names)                 
                tr_loss, vl_loss = self.model.train_and_validate(xi_tr, yt_tr, xi_vl, yt_vl, **params)
            case "LinReg":
                xi_tr, xi_vl = pd.DataFrame(xi_train, columns=self.Q.param_names()), pd.DataFrame(xi_val, columns=self.Q.param_names())
                yt_tr, yt_vl = pd.DataFrame(yt_train, columns=self.QoI_names), pd.DataFrame(yt_val, columns=self.QoI_names)
                tr_loss, vl_loss = self.model.train_and_evaluate(xi_tr, yt_tr, xi_vl, yt_vl)
            case _:
                raise ValueError(f"There is no method type: {self.method}")
        return tr_loss, vl_loss

    def __getstate__(self):
        '''
        Prepares the instance state for pickling by removing large training data attributes.

        Returns:
        -------
        state : dict
            The instance state dictionary without large training data attributes.
        '''

        # Create a copy of the instance dictionary
        state = self.__dict__.copy()
        
        # Remove large training data attributes before pickling
        for attr in ['X_train', 'y_train', 'X_test', 'y_test']:
            if attr in state:
                del state[attr]
        
        return state

    def __setstate__(self, state):
        '''
        Restores the instance state from the pickled state dictionary.

        Parameters:
        -----------
        state : dict
            The instance state dictionary to restore.
        '''

        # Restore the instance state
        self.__dict__.update(state)
    
    def save_model(self):
        '''
        Serializes the predictive model instance to a byte stream.

        Returns:
        -------
        bytes
            The serialized byte stream of the predictive model instance.
        '''
        return pickle.dumps(self)

    def save_shap_model(self, name, path=None):
        '''
        Saves the SHAP explainer model to a file.

        Parameters:
        -----------
        name : str
            The name of the file to save the SHAP explainer model.
        path : str, optional
            The directory path to save the file (default is None).

        Returns:
        -------
        None
        '''

        name = name + ".xm"
        if path is not None:
            name = os.path.join(path, name)
        with open(name, 'wb') as file:
            pickle.dump(self.model.explainer, file)
        print(f"Model saved to {name}")

    def predict(self, q, **params):
        '''
        Predicts QoI values for given input parameters using the trained model.

        Parameters:
        -----------
        q : pd.DataFrame
            Input features for prediction
        **params : dict
            Additional prediction parameters

        Returns:
        -------
        y : pd.DataFrame
            Predicted QoI values
        '''

        if type(q) != pd.DataFrame:
            q = pd.DataFrame(q, columns=self.Q.param_names())
        if self.method == "gPCE":
            xi = q.values
        else:
            xi = self.get_scaled_q(q)
        match self.method:
            case "DNN":
                xi = torch.tensor(xi, dtype=torch.float64)
            case "gPCE":
                pass
            case "GBT":
                xi = pd.DataFrame(xi, columns=self.Q.param_names())
            case "LinReg":
                xi = pd.DataFrame(xi, columns=self.Q.param_names())
            case _:
                raise ValueError(f"There is no method type: {self.method}")
        y_t = self.model.predict(xi, **params)
        if self.method == "gPCE":
            y = y_t
        else:
            y = self.get_orig_y(y_t)
        return y
    
    def get_mean_and_var(self, n_sample=1000):
        '''
        Computes the mean and variance of the model predictions over a specified number of samples.

        Parameters:
        -----------
        n_sample : int
            Number of samples to use for computing mean and variance (default is 1000)
        
        Returns:
        -------
        mean : np.ndarray
            Mean of the model predictions
        var : np.ndarray
            Variance of the model predictions
        '''

        if self.method == "DNN" or self.method == "GBT" or self.method == "LinReg":
            q = self.Q.sample(n_sample)
            q_df = pd.DataFrame(q, columns=self.Q.param_names())
            y_predict = self.predict(q_df)
            mean, var = y_predict.mean(axis=0), y_predict.var(axis=0)
        elif self.method == "gPCE":
            mean, var = self.model.mean(), self.model.variance()
        return mean, var

    def get_q_scaler(self, q, q_scaler, scale_method='default'):
        '''
        Initializes and returns the scaler for input features.

        Parameters:
        -----------
        q : pd.DataFrame
            Input features
        q_scaler : object
            Scaler for input features (default is None)
        scale_method : str
            Scaling method ('default', 'identity', 'minmax') (default is 'default')

        Returns:
        -------
        q_scaler : object
            Scaler for input features
        '''
        if scale_method == 'default':
            match self.method:
                case "DNN":
                    scale_method = 'minmax'
                case "gPCE":
                    scale_method = 'minmax'
                case "GBT":
                    scale_method = 'identity'
                case "LinReg":
                    scale_method = 'identity'
                    
        if scale_method == 'identity':
            self.q_scaler = FunctionTransformer(func=identity_func, inverse_func=identity_inverse_func)
        elif scale_method == 'minmax':
            self.q_scaler = MinMaxScaler((0.05, 0.95))
            self.q_scaler.fit(q)
        return self.q_scaler
    
    def get_scaled_q(self, q):
        '''
        Scales the input features using the initialized scaler.

        Parameters:
        -----------
        q : pd.DataFrame
            Input features to be scaled

        Returns:
        -------
        xi : np.ndarray
            Scaled input features
        '''

        xi = self.q_scaler.transform(q)
        return xi
    
    def get_orig_q(self, xi):
        '''
        Inversely transforms the scaled input features back to their original scale.

        Parameters:
        -----------
        xi : np.ndarray
            Scaled input features

        Returns:
        -------
        q : pd.DataFrame
            Original input features
        '''

        q = self.q_scaler.inverse_transform(xi)
        return q
    
    def get_y_scaler(self, y, y_scaler, scale_method='default'):
        '''
        Initializes and returns the scaler for target values.

        Parameters:
        -----------
        y : pd.DataFrame
            Target values
        y_scaler : object
            Scaler for target values (default is None)
        scale_method : str
            Scaling method ('default', 'identity', 'minmax') (default is 'default')

        Returns:
        -------
        y_scaler : object
            Scaler for target values
        '''

        if scale_method == 'default':
            match self.method:
                case "DNN":
                    scale_method = 'minmax'
                case "gPCE":
                    scale_method = 'minmax'
                case "GBT":
                    scale_method = 'identity'
                case "LinReg":
                    scale_method = 'identity'

        if scale_method == 'identity':
            self.y_scaler = FunctionTransformer(func=identity_func, inverse_func=identity_inverse_func)
        elif scale_method == 'minmax':
            self.y_scaler = MinMaxScaler((0.05, 0.95))
            self.y_scaler.fit(y)
        return self.y_scaler
    
    def get_scaled_y(self, y):
        '''
        Scales the target values using the initialized scaler.

        Parameters:
        -----------
        y : pd.DataFrame
            Target values to be scaled

        Returns:
        -------
        y_t : np.ndarray
            Scaled target values
        '''

        y_t = self.y_scaler.transform(y)
        return y_t
    
    def get_orig_y(self, y_t):
        '''
        Inversely transforms the scaled target values back to their original scale.

        Parameters:
        -----------
        y_t : np.ndarray
            Scaled target values

        Returns:
        -------
        y : pd.DataFrame
            Original target values
        '''

        y = self.y_scaler.inverse_transform(y_t)
        return y
    
    def evaluate_model(self, y_train, X_test, y_test, verbose=False):
        '''
        Evaluates the predictive model using various metrics.

        Parameters:
        -----------
        y_train : pd.DataFrame
            Training target values
        X_test : pd.DataFrame
            Testing input features
        y_test : pd.DataFrame
            Testing target values
        verbose : bool
            If True, prints the evaluation metrics (default is False)

        Returns:
        -------
        list:
            List containing a DataFrame of evaluation metrics and a dictionary of model evaluation results.
        '''

        pred = self.predict(X_test)

        # Ensure y_test is a DataFrame for consistent column-wise iteration
        if isinstance(y_test, pd.Series):
            y_test_df = y_test.to_frame()
        elif isinstance(y_test, np.ndarray) and y_test.ndim == 1:
            y_test_df = pd.DataFrame(y_test)
        else: # Already a DataFrame or multi-dim array
            y_test_df = y_test if isinstance(y_test, pd.DataFrame) else pd.DataFrame(y_test)

        # Ensure y_train is a DataFrame for consistent column-wise iteration
        if isinstance(y_train, pd.Series):
            y_train_df = y_train.to_frame()
        elif isinstance(y_train, np.ndarray) and y_train.ndim == 1:
            y_train_df = pd.DataFrame(y_train)
        else: # Already a DataFrame or multi-dim array
            y_train_df = y_train if isinstance(y_train, pd.DataFrame) else pd.DataFrame(y_train)

        # Ensure prediction is a DataFrame for consistent column-wise iteration
        if isinstance(pred, pd.Series):
            pred_df = pred.to_frame()
        elif isinstance(pred, np.ndarray) and pred.ndim == 1:
            pred_df = pd.DataFrame(pred)
        else: # Already a DataFrame or multi-dim array
            pred_df = pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred)

        num_outputs = y_test_df.shape[1]
        output_column_names = y_test_df.columns.tolist()
        if num_outputs == 1:
            output_column_names[0] = self.QoI_names[0]

        # List to store dictionaries for each output's metrics (for DataFrame creation)
        individual_metrics_list = []

        for i, col_name in enumerate(output_column_names):
            y_true_dim = y_test_df.iloc[:, i]
            y_pred_dim = pred_df.iloc[:, i]
            y_train_dim = y_train_df.iloc[:, i]

            # Check for constant inputs before calculating correlations
            is_y_true_constant = (y_true_dim.nunique() <= 1)
            is_y_pred_constant = (y_pred_dim.nunique() <= 1)

            if is_y_true_constant or is_y_pred_constant:
                # If either input is constant, correlation is undefined
                kendall_tau_val = np.nan
                pearson_val = np.nan
                spearman_val = np.nan
            else:
                try:
                    kendall_tau_val = kendalltau(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    kendall_tau_val = np.nan
                try:
                    pearson_val = pearsonr(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    pearson_val = np.nan
                try:
                    spearman_val = spearmanr(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    spearman_val = np.nan

            mse_val = mean_squared_error(y_true_dim, y_pred_dim)
            mae_val = mean_absolute_error(y_true_dim, y_pred_dim)
            rmse_val = np.sqrt(mse_val)
            norm_rmse = rmse_val / (max(y_train_dim) - min(y_train_dim))
            norm_mae = mae_val / (max(y_train_dim) - min(y_train_dim))

            # Get STD of label for the specific training output column
            if isinstance(y_train, pd.Series):
                std_of_label_val = y_train.std()
            elif isinstance(y_train, np.ndarray) and y_train.ndim == 1:
                std_of_label_val = pd.Series(y_train).std()
            else: # It's a DataFrame or multi-dim array
                y_train_df_temp = y_train if isinstance(y_train, pd.DataFrame) else pd.DataFrame(y_train)
                std_of_label_val = y_train_df_temp.iloc[:, i].std()

            current_output_metrics = {
                "Kendall_tau": kendall_tau_val,
                "Pearson": pearson_val,
                "Spearman": spearman_val,
                "MSE": mse_val,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "STD_of_label": std_of_label_val,
                "NRMSE": norm_rmse,
                "NMAE": norm_mae
            }
            # Add relative RMSE and summed metric for each output
            # Check for zero STD_of_label before division
            if std_of_label_val == 0 or np.isnan(std_of_label_val):
                current_output_metrics["rel_RMSE"] = np.nan
            else:
                current_output_metrics["rel_RMSE"] = (
                    current_output_metrics["RMSE"] /
                    current_output_metrics["STD_of_label"]
                )

            # Ensure correlation metrics are not NaN for summed metric calculation
            k_tau = 0 if np.isnan(kendall_tau_val) else kendall_tau_val
            p_corr = 0 if np.isnan(pearson_val) else pearson_val
            s_corr = 0 if np.isnan(spearman_val) else spearman_val
            rel_rmse = current_output_metrics["rel_RMSE"] if not np.isnan(current_output_metrics["rel_RMSE"]) else 0

            current_output_metrics["summed_metric"] = (
                k_tau + p_corr + s_corr - rel_rmse
            ) / 3

            # Add the output name/label to this dictionary
            current_output_metrics['Output_Name'] = col_name
            individual_metrics_list.append(current_output_metrics)

        # Create DataFrame for individual output metrics
        df_individual_metrics = pd.DataFrame(individual_metrics_list)
        df_individual_metrics = df_individual_metrics.set_index('Output_Name')

        # --- Conditional Aggregated Metrics ---
        model_eval_aggregated = {}
        if num_outputs > 1: # Only calculate aggregated metrics if there are multiple outputs
            for metric_name in ["Kendall_tau", "Pearson", "Spearman", "MSE", "MAE", "RMSE", "rel_RMSE", 'NRMSE', 'NMAE', "summed_metric"]:
                values = df_individual_metrics[metric_name].dropna().tolist()
                if values:
                    model_eval_aggregated[f"Aggregated_{metric_name}_Mean"] = np.mean(values)
                    model_eval_aggregated[f"Aggregated_{metric_name}_Min"] = np.min(values)
                    model_eval_aggregated[f"Aggregated_{metric_name}_Max"] = np.max(values)
                else:
                    model_eval_aggregated[f"Aggregated_{metric_name}_Mean"] = np.nan
                    model_eval_aggregated[f"Aggregated_{metric_name}_Min"] = np.nan
                    model_eval_aggregated[f"Aggregated_{metric_name}_Max"] = np.nan

        if verbose:
            print("\n--- Individual Output Metrics (Table) ---")
            print(df_individual_metrics.round(3))

            if num_outputs > 1:
                print("\n--- Aggregated Metrics ---")
                for key, value in model_eval_aggregated.items():
                    print(f'{key}: {value:.3f}')
            else:
                print("\n(Aggregated metrics not shown for single output.)")

        return [df_individual_metrics, model_eval_aggregated]
    
    def get_sobol_sensitivity(self, max_index=1):
        '''
        Computes Sobol sensitivity indices up to the specified maximum index.

        Parameters:
        -----------
        max_index : int
            Maximum index for Sobol sensitivity (default is 1)

        Returns:
        -------
        partial_var_df : pd.DataFrame
            DataFrame of partial variances
        sobol_index_df : pd.DataFrame
            DataFrame of Sobol sensitivity indices
        '''

        if (self.method == "DNN" or self.method == "GBT") and max_index > 2:
            print(f'Warning: The maximum value of max_index for {self.method} is 2. It is automatically set to 2.')
            max_index = 2
        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        partial_var_df, sobol_index_df, total_var = self.model.compute_partial_vars(self, max_index)
        self.partial_var_df, self.sobol_index_df, self.total_var = partial_var_df, sobol_index_df, total_var
        return partial_var_df, sobol_index_df
            
    async def get_shap_values(self, q, mean=False, sample_size_from_q=100):
        '''
        Computes SHAP values for the model predictions.

        Parameters:
        -----------
        q : pd.DataFrame
            Input features for SHAP value computation
        mean : bool
            If True, returns mean absolute SHAP values (default is False)
        sample_size_from_q : int or str
            Number of samples from q to use for SHAP computation or 'all' to use all samples (default is 100)

        Returns:
        -------
        shap_values : pd.DataFrame or np.ndarray
            SHAP values or mean absolute SHAP values
        '''

        if type(sample_size_from_q) is str and sample_size_from_q == 'all':
            print(f'Message: sample size for shap values is set to {q.shape[0]}, which is the number of samples.')
            sample_size_from_q = q.shape[0]
        elif type(sample_size_from_q) is int:
            if q.shape[0] < sample_size_from_q:
                print(f'Warning: sample_size_from_q is larger than the number of samples. It is automatically set to {q.shape[0]}, which is the number of samples.')
                sample_size_from_q = q.shape[0]
            else:
                print(f'Message: sample size for shap values is set to {sample_size_from_q}.')
            q = q.iloc[:sample_size_from_q]
        shap_values = await asyncio.to_thread(self.model.get_shap_values, self.predict, q)
        self.shap_values = shap_values
        if mean:
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            mean_shap_values_df = pd.DataFrame(mean_shap_values, index=q.columns, columns=self.QoI_names)
            return mean_shap_values_df
        return shap_values
    
    async def subtract_effects(self, q, QoI, subtracted_params):
        '''
        Subtracts the effects of specified parameters from the QoI using SHAP values.

        Parameters:
        -----------
        q : pd.DataFrame
            Input features
        QoI : pd.DataFrame
            Quantity of Interest values
        subtracted_params : list
            List of parameter names whose effects are to be subtracted

        Returns:
        -------
        remained_effects : pd.DataFrame
            QoI values with specified parameter effects subtracted
        '''

        QoI = pd.DataFrame(QoI, columns=self.QoI_names)
        shap_values = await self.get_shap_values(q, mean=False, sample_size_from_q='all')
        columns = q.columns.tolist()
        indices = []
        for x in subtracted_params:
            if x in columns:
                indices.append(columns.index(x))
            else:
                raise ValueError(f"There is no parameter called \'{x}\'")
        result = shap_values.values[:, indices, :].sum(axis=1)
        result_df = pd.DataFrame(result, columns=QoI.columns, index=QoI.index)
        remained_effects = QoI - result_df
        return remained_effects
    
    def calculate_error_ratios(self, errors, threshold_ratio):
        '''
        Calculates error ratios based on a threshold ratio.

        Parameters:
        -----------
        errors : pd.DataFrame
            DataFrame of errors
        threshold_ratio : float
            Threshold ratio for error calculation

        Returns:
        -------
        error_ratio : float
            Ratio of errors exceeding the threshold
        problematic_rows : pd.DataFrame
            DataFrame of rows with errors exceeding the threshold
        '''

        values = errors.values
        total_errors = len(values)
        for i in range(len(values)):
            for j in range(len(values[i])):
                if abs(values[i][j]) < 0.5:
                    values[i][j] = 0
                else:
                    values[i][j] = (abs(values[i][j]) - 0.5) * np.sign(values[i][j])
        indexes = []
        for i in range(len(values)):
            if (np.abs(values[i, :]) > threshold_ratio).any():
                indexes.append(i)
        error_ratio = len(indexes) / total_errors
        problematic_rows = pd.DataFrame(values[indexes], columns=errors.columns)
        return error_ratio, problematic_rows
    
    
    async def plot_subtract_effects_and_alert(self, X_train, y_train, q, QoI, subtracted_params, threshold_ratio=0.1, xticks=True):
        '''
        Plots the effects after subtracting specified parameter effects and checks for alerts.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training input features
        y_train : pd.DataFrame
            Training target values
        q : pd.DataFrame
            Input features for monitoring
        QoI : pd.DataFrame
            Quantity of Interest values for monitoring
        subtracted_params : list
            List of parameter names whose effects are to be subtracted
        threshold_ratio : float
            Threshold ratio for alerting (default is 0.1)
        xticks : bool
            If True, includes x-ticks in the plot (default is True)

        Returns:
        ------- 
        fig : matplotlib.figure.Figure
            Figure object of the plot
        alert : bool
            True if an alert is triggered, False otherwise
        remained_effects_df : pd.DataFrame
            DataFrame of remained effects after subtraction
        error_ratio : float
            Ratio of errors exceeding the threshold
        problematic_rows : pd.DataFrame
            DataFrame of rows with errors exceeding the threshold
        reference_statistics : pd.DataFrame
            Statistics of the remained training effects
        monitoring_statistics : pd.DataFrame
            Statistics of the remained measured effects        
        '''

        alert = False
        remained_train_effects = await self.subtract_effects(X_train, pd.DataFrame(y_train), subtracted_params)
        QoI_names = remained_train_effects.columns
        reference_statistics = round(remained_train_effects.describe().T, 4).reindex(columns=['25%', '50%', '75%', 'count', 'max', 'mean', 'min', 'std'])
        remained_train_effects = remained_train_effects.values

        higher_bound = np.max(remained_train_effects, axis=0)
        lower_bound = np.min(remained_train_effects, axis=0)
        magnitudes = (higher_bound - lower_bound)
        mean_train = higher_bound - magnitudes / 2
        margin = magnitudes * threshold_ratio

        remained_measured_effects = await self.subtract_effects(q, QoI, subtracted_params)
        monitoring_statistics = round(remained_measured_effects.describe().T, 4).reindex(columns=['25%', '50%', '75%', 'count', 'max', 'mean', 'min', 'std'])
        remained_measured_effects = remained_measured_effects.values

        higher_measured_bound = np.max(remained_measured_effects, axis=0)
        lower_measured_bound = np.min(remained_measured_effects, axis=0)

        remained_effects_df = pd.DataFrame(remained_measured_effects, columns=QoI_names)
        measured_errors_df = pd.DataFrame((remained_measured_effects - mean_train) / magnitudes, columns=QoI_names)

        error_ratio, problematic_rows = self.calculate_error_ratios(measured_errors_df, threshold_ratio)

        for i in range(len(higher_measured_bound)):
            if (higher_measured_bound[i] > higher_bound[i] + margin[i]) or (lower_measured_bound[i] < lower_bound[i] - margin[i]):
                alert = True
                break
        fig = utils.plot_remained_effects(higher_bound, lower_bound, margin, higher_measured_bound, lower_measured_bound, QoI_names, xticks)
        return fig, alert, remained_effects_df, error_ratio, problematic_rows, reference_statistics, monitoring_statistics

    
    def plot_sobol_sensitivity(self, y_train, max_index=1, **kwargs):
        '''
        Plots Sobol sensitivity indices up to the specified maximum index.

        Parameters:
        -----------
        y_train : pd.DataFrame
            Training target values
        max_index : int
            Maximum index for Sobol sensitivity (default is 1)
        **kwargs : dict
            Additional plotting parameters

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        '''

        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        if (self.method == "DNN" or self.method == "GBT") and max_index > 2:
            print(f'Warning: The maximum value of max_index for {self.method} is 2. It is automatically set to 2.')
            max_index = 2
        if self.max_index < max_index:
            self.get_sobol_sensitivity(max_index)
        fig = utils.plot_sobol_sensitivity(self.sobol_index_df, y_train, **kwargs)
        return fig
    
    async def plot_shap_single_waterfall(self, X_test, **kwargs):
        '''
        Plots a single SHAP waterfall plot for the model predictions.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing input features
        **kwargs : dict
            Additional plotting parameters

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        '''

        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_single_waterfall(self, **kwargs)
        return fig
    
    async def plot_shap_multiple_waterfalls(self, X_test, **kwargs):
        '''
        Plots multiple SHAP waterfall plots for the model predictions.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing input features
        **kwargs : dict
            Additional plotting parameters

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        '''

        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_multiple_waterfalls(self, **kwargs)
        return fig
    
    async def plot_shap_beeswarm(self, X_test, **kwargs):
        '''
        Plots a SHAP beeswarm plot for the model predictions.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing input features
        **kwargs : dict
            Additional plotting parameters

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        '''

        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_beeswarm(self, **kwargs)
        return fig
       
    async def plot_effects(self, effects, xticks=True):
        '''
        Plots the effects of parameters on the QoI.

        Parameters:
        -----------
        effects : pd.DataFrame
            DataFrame of effects
        xticks : bool
            If True, includes x-ticks in the plot (default is True)

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        '''

        fig = utils.plot_effects(effects, xticks)
        return fig
    
    def create_mx_from_tuple(self, pairs, measured_names):
        '''
        Creates a measurement matrix H based on provided pairs of QoI names and measured data names.

        Parameters:
        -----------
        pairs : list of tuples
            List of (QoI_name, measured_name) pairs indicating the mapping.
        measured_names : list
            List of names corresponding to the measured data columns.

        Returns:
        -------
        H : np.ndarray
            Measurement matrix H with shape (len(QoI_names), len(measured_names)).
        '''

        if len(measured_names) < len(self.QoI_names):
            raise ValueError("Measured data must have at least as many columns as QoI_names.")

        # Create index mappings for rows and columns
        row_idx = {name: i for i, name in enumerate(self.QoI_names)}
        col_idx = {name: j for j, name in enumerate(measured_names)}
        
        H = np.zeros((len(self.QoI_names), len(measured_names)), dtype=int)
        for a, b in pairs:
            H[row_idx[a], col_idx[b]] = 1
        
        return H

class SurrogateModel(PredictiveModel):
    '''
    SurrogateModel class for creating and managing surrogate models.
    Inherits from PredictiveModel and provides functionality to save the model.

    Attributes:
    -----------
    Q : VariableSet
        Set of input variables.
    QoI_names : list
        List of Quantity of Interest (QoI) names.
    method : str
        Method used for the surrogate model ('DNN', 'gPCE', 'GBT', 'LinReg').
    **kwargs : dict
        Additional keyword arguments for model configuration.

    Methods:
    --------
    save_model(self, name=None, path=None):
        Saves the surrogate model to a file or returns a byte stream.
    '''

    def __init__(self, Q, QoI_names, method, **kwargs):
        '''
        Initializes the SurrogateModel instance.

        Parameters:
        -----------
        Q : VariableSet
            Set of input variables.
        QoI_names : list
            List of Quantity of Interest (QoI) names.
        method : str
            Method used for the surrogate model (e.g., 'DNN', 'gPCE
            , 'GBT', 'LinReg').
        **kwargs : dict
            Additional keyword arguments for model configuration.
        '''
        super().__init__(Q, QoI_names, method, **kwargs)
    
    def save_model(self, name=None, path=None): # rossz
        '''
        Saves the surrogate model to a file or returns a byte stream.

        Parameters:
        -----------
        name : str, optional
            Name of the file to save the model (default is None).
        path : str, optional
            Directory path to save the file (default is None).

        Returns:
        -------
        None or bytes
            If name is provided, saves the model to a file. Otherwise, returns a byte stream.
        '''

        if name is not None:
            name = name + ".sm"
            if path is not None:
                name = os.path.join(path, name)
            with open(name, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {name}")
        else:
            return pickle.dumps(self)
    
class DataDrivenModel(PredictiveModel):
    '''
    DataDrivenModel class for creating and managing data-driven models.
    Inherits from PredictiveModel and provides functionality to save the model.

    Attributes:
    -----------
    Q : VariableSet
        Set of input variables.
    QoI_names : list
        List of Quantity of Interest (QoI) names.
    method : str
        Method used for the data-driven model ('DNN', 'gPCE', 'GBT', 'LinReg').
    **kwargs : dict
        Additional keyword arguments for model configuration.

    Methods:
    --------
    save_model(self, name=None, path=None):
        Saves the data-driven model to a file or returns a byte stream.
    '''

    def __init__(self, Q, QoI_names, method, **kwargs):
        '''
        Initializes the SurrogateModel instance.

        Parameters:
        -----------
        Q : VariableSet
            Set of input variables.
        QoI_names : list
            List of Quantity of Interest (QoI) names.
        method : str
            Method used for the surrogate model (e.g., 'DNN', 'gPCE
            , 'GBT', 'LinReg').
        **kwargs : dict
            Additional keyword arguments for model configuration.
        '''

        super().__init__(Q, QoI_names, method, **kwargs)
    
    def save_model(self, name=None, path=None):
        '''
        Saves the data-driven model to a file or returns a byte stream.

        Parameters:
        -----------
        name : str, optional
            Name of the file to save the model (default is None).
        path : str, optional   
            Directory path to save the file (default is None).

        Returns:
        -------
        None or bytes
            If name is provided, saves the model to a file. Otherwise, returns a byte stream.        
        '''

        if name is not None:
            name = name + ".ddm"
            if path is not None:
                name = os.path.join(path, name)
            with open(name, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {name}")
        else:
            return pickle.dumps(self)

def identity_func(x):
    '''
    Identity function that returns the input as a NumPy array.

    Parameters:
    -----------
    x : pd.DataFrame or np.ndarray
        Input data to be converted.

    Returns:
    -------
    np.ndarray
        Input data as a NumPy array.    
    '''

    if isinstance(x, np.ndarray):
        return x
    return x.to_numpy()

def identity_inverse_func(x):
    '''
    Inverse identity function that returns the input as is.

    Parameters:
    -----------
    x : np.ndarray
        Input data.
        
    Returns:
    -------
    np.ndarray
        Input data as is.
    '''

    return x