'''Module for Predictive Model including Surrogate Model and Data-Driven Model'''

import asyncio
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from .plotting_functions import plot_remained_effects, plot_shap_single_waterfall, plot_shap_multiple_waterfalls, plot_shap_beeswarm, plot_effects, plot_sobol_sensitivity
import os
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from SALib.analyze import sobol


class PredictiveModel(ABC):
    '''
    Abtract class for Predictive Models including Surrogate Models and Data-Driven Models '''

    @abstractmethod
    def train(self, q_train, y_train, k_fold=None, **params):
        """ Trains the predictive model using the provided training data.

            Parameters
            ----------
            q_train : array-like
                Input feature data for training.
            y_train : array-like
                Target data for training.
            k_fold : int, optional
                Number of folds for cross-validation. If None, no cross-validation is performed.
            **params : dict
                Additional training parameters.

            Returns
            -------
            None
        """
        pass

    @abstractmethod
    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        """ Trains the predictive model on the provided training and validation data and computes the training and validation loss.

            Parameters
            ----------
            xi_train : array-like
                Input feature data for training.
            yt_train : array-like
                Target data for training.
            xi_val : array-like
                Input feature data for validation.
            yt_val : array-like
                Target data for validation.
            **params : dict
                Additional training parameters.

            Returns
            -------
            tr_loss : float
                Training loss.
            vl_loss : float
                Validation loss.
        """
        pass

    @abstractmethod
    def predict(self, q):
        """ Makes predictions using the trained predictive model.

            Parameters
            ----------
            q : array-like
                Input feature data for prediction.

            Returns
            -------
            predictions : array-like
                Predicted target values.
        """
        pass

    def train_test_split(self, X_data, y_data, train_test_ratio=0.8, random_seed=1997, split_type='shuffle'):
        '''
        Splits the dataset into training and testing sets.

        Parameters
        ----------
        X_data : pd.DataFrame
            Input features
        y_data : pd.DataFrame
            Target values
        train_test_ratio : float
            Ratio of training data to total data (default is 0.8)
        random_seed : int
            Random seed for reproducibility (default is 1997)
        split_type : str
            Type of split ('shuffle', 'no_shuffle', 'reverse') (default is 'shuffle')

        Returns
        -------
        X_train, X_test, y_train, y_test : pd.DataFrame
            Split training and testing datasets
        '''

        self.samples = X_data
        self.QoI = y_data        
        if split_type == 'shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_test_ratio, random_state=random_seed, shuffle=True)
        elif split_type == 'no_shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_test_ratio, random_state=random_seed, shuffle=False)
        elif split_type == 'reverse':
            X_test, X_train, y_test, y_train = train_test_split(X_data, y_data, test_size=train_test_ratio, random_state=random_seed, shuffle=False)
        
        return X_train, X_test, y_train, y_test



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

    def get_mean_and_var(self, n_sample=1000):
        """ Computes the mean and variance of the model predictions for a given number of samples.

            Parameters
            ----------
            n_sample : int, optional
                The number of samples to use for prediction (default is 1000).

            Returns
            -------
            mean : array-like
                Mean of the predicted target values.
            var : array-like
                Variance of the predicted target values.
        """
        q = self.Q.sample(n_sample)
        q_df = pd.DataFrame(q, columns=self.Q.variable_names())
        y_predict = self.predict(q_df)
        mean, var = y_predict.mean(axis=0), y_predict.var(axis=0)
        return mean, var

    
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
        for attr in ['X_train', 'y_train', 'X_test', 'y_test', 'explainer']:
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

        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        partial_var_df, sobol_index_df, total_var = self.compute_partial_vars(max_index)
        self.partial_var_df, self.sobol_index_df, self.total_var = partial_var_df, sobol_index_df, total_var
        return partial_var_df, sobol_index_df

    def score(self, X, y):
        '''
        Computes the mean squared error of the model on given data.

        Parameters
        ----------
        X : array-like
            Input feature data.
        y : array-like
            True target values.

        Returns
        -------
        mse : float
            Mean squared error of the model.
        '''

        return mean_squared_error(y, self.predict(X))

                
    async def get_shap_values(self, q, mean=False, sample_size_from_q=100, silent=False):
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
        shap_values = await asyncio.to_thread(self.compute_shap_values, self.predict, q, silent=silent)
        self.shap_values = shap_values
        if mean:
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            mean_shap_values_df = pd.DataFrame(mean_shap_values, index=q.columns, columns=self.QoI_names)
            return mean_shap_values_df
        return shap_values
    
    async def subtract_effects(self, q, QoI, subtracted_params, silent=False):
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
        shap_values = await self.get_shap_values(q, mean=False, sample_size_from_q='all', silent=silent)
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
        problematic_rows = pd.DataFrame(values[indexes], columns=errors.columns, index=indexes)
        return error_ratio, problematic_rows
    
    
    async def plot_subtract_effects_and_alert(self, X_train, y_train, q, QoI, subtracted_params, threshold_ratio=0.1, xticks=True, silent=False):
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
        remained_train_effects = await self.subtract_effects(X_train, pd.DataFrame(y_train), subtracted_params, silent=silent)
        QoI_names = remained_train_effects.columns
        reference_statistics = round(remained_train_effects.describe().T, 4).reindex(columns=['25%', '50%', '75%', 'count', 'max', 'mean', 'min', 'std'])
        remained_train_effects = remained_train_effects.values

        higher_bound = np.max(remained_train_effects, axis=0)
        lower_bound = np.min(remained_train_effects, axis=0)
        magnitudes = (higher_bound - lower_bound)
        mean_train = higher_bound - magnitudes / 2
        margin = magnitudes * threshold_ratio

        remained_measured_effects = await self.subtract_effects(q, QoI, subtracted_params, silent=silent)
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
        fig = plot_remained_effects(higher_bound, lower_bound, margin, higher_measured_bound, lower_measured_bound, QoI_names, xticks)
        return fig, alert, remained_effects_df, error_ratio, problematic_rows, reference_statistics, monitoring_statistics

        
    async def plot_shap_single_waterfall(self, X_test, silent=True, **kwargs):
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
            self.shap_values = await self.get_shap_values(X_test, silent=silent)
            print('The SHAP values have been computed automatically!')
        fig = plot_shap_single_waterfall(self, silent, **kwargs)
        return fig
    
    async def plot_shap_multiple_waterfalls(self, X_test, silent=True, **kwargs):
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
            self.shap_values = await self.get_shap_values(X_test, silent=silent)
            print('The SHAP values have been computed automatically!')
        fig = plot_shap_multiple_waterfalls(self, silent, **kwargs)
        return fig
    
    async def plot_shap_beeswarm(self, X_test, silent=True, **kwargs):
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
            self.shap_values = await self.get_shap_values(X_test, silent=silent)
            print('The SHAP values have been computed automatically!')
        fig = plot_shap_beeswarm(self, silent, **kwargs)
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

        fig = plot_effects(effects, xticks)
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
            name = name + ".sm"
            if path is not None:
                name = os.path.join(path, name)
            with open(name, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {name}")
        else:
            return pickle.dumps(self)


    def compute_partial_vars_old(self, max_index):
        '''
        Computes partial variances using Sobol sensitivity analysis.

        Parameters
        ----------
        max_index : int
            Maximum index for Sobol analysis (1 or 2).

        Returns
        -------
        partial_var_df : pd.DataFrame
            DataFrame containing partial variances for each parameter and QoI.
        sobol_index_df : pd.DataFrame
            DataFrame containing Sobol indices for each parameter and QoI.
        y_var : np.ndarray
            Variance of the model outputs.
        '''

        variableset = self.Q
        QoI_names = self.QoI_names

        problem = {
            'num_vars': variableset.num_variables(), 'names': variableset.variable_names(), 'dists': variableset.get_dist_types(), 'bounds': variableset.get_dist_params()
            } 
        
        d = variableset.num_variables()
        q = variableset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
        y = self.predict(q)
        
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


        def _to_label(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return " ".join(_to_label(e) for e in x)
            return str(x)

        col_names = [_to_label(x) for x in col_names]
        QoI_names = [_to_label(x) for x in self.QoI_names]

                    
        # Compute partial variances
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var


    def compute_partial_vars(self, max_index, n=8192):
        """ Computes the partial variances and Sobol sensitivity indices for the predictive model.

            Parameters
            ----------
            max_index : int
                Maximum order of Sobol indices to compute (1 for first-order, 2 for second-order).
            n : int, optional
                Number of samples to use for the Sobol analysis (default is 8192).

            Returns
            -------
            partial_var_df : pandas.DataFrame
                DataFrame containing the partial variances for each input parameter and QoI.
            sobol_index_df : pandas.DataFrame
                DataFrame containing the Sobol sensitivity indices for each input parameter and QoI.
        """
        variableset = self.Q
        d = variableset.num_variables()

        def _to_label(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return " ".join(_to_label(e) for e in x)
            return str(x)

        names = [_to_label(x) for x in variableset.variable_names()]
        QoI_names = [_to_label(x) for x in self.QoI_names]

        q = variableset.sample(method='Sobol_saltelli', n=n)   # N * (2D + 2) rows
        y = np.asarray(self.predict(q), dtype=float)
        if y.ndim == 1:
            y = y[:, None]

        step = 2 * d + 2
        N = y.shape[0] // step
        if N * step != y.shape[0]:
            raise ValueError("Sample size does not match Saltelli layout N*(2D+2).")

        # Standardize each QoI, as SALib does inside analyze()
        std = y.std(axis=0)
        std[std == 0] = 1.0
        Y = (y - y.mean(axis=0)) / std

        # Saltelli layout per block: A, AB_1..AB_D, BA_1..BA_D, B
        Yr = Y.reshape(N, step, -1)            # (N, step, Q)
        A  = Yr[:, 0, :]                       # (N, Q)
        B  = Yr[:, -1, :]                      # (N, Q)
        AB = Yr[:, 1:d + 1, :]                 # (N, d, Q)
        BA = Yr[:, d + 1:2 * d + 1, :]         # (N, d, Q)

        V = np.concatenate([A, B]).var(axis=0) # (Q,)
        V[V == 0] = np.inf                     # constant output -> indices 0

        # First order (Saltelli 2010): mean(B * (AB_j - A)) / V
        S1 = np.einsum('nq,njq->qj', B, AB - A[:, None, :]) / N / V[:, None]   # (Q, d)

        sobol_index = S1
        col_names = names

        if max_index == 2:
            jj, kk = np.triu_indices(d, k=1)
            # Vjk = mean(BA_j * AB_k - A*B), all pairs at once via batched matmul
            BA_t = BA.transpose(2, 1, 0)       # (Q, d, N)
            AB_t = AB.transpose(2, 0, 1)       # (Q, N, d)
            Vjk = (BA_t @ AB_t) / N - (A * B).mean(axis=0)[:, None, None]   # (Q, d, d)
            S2 = Vjk[:, jj, kk] / V[:, None] - S1[:, jj] - S1[:, kk]         # (Q, n_pairs)

            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = names + [f"{names[j]} {names[k]}" for j, k in zip(jj, kk)]

        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var

        partial_var_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names)
        sobol_index_df = pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)
        return partial_var_df, sobol_index_df, y_var


    def plot_sobol_sensitivity(self, y_train, max_index=1, param_name=None):
        """ Plots the Sobol sensitivity indices for the predictive model.

            Parameters
            ----------
            y_train : array-like
                Training target data.
            max_index : int, optional
                Maximum order of Sobol indices to compute (default is 1).
            param_name : list of str, optional
                Names of the input parameters for labeling the plot.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object containing the Sobol sensitivity plot.
        """
        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        if self.max_index < max_index:
            self.get_sobol_sensitivity(max_index)
        fig = plot_sobol_sensitivity(self.sobol_index_df, y_train, param_name)
        return fig