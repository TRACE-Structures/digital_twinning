from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import numpy as np
from .plotting_functions import posterior_plot
import os
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


POINT_ESTIMATES = ('mean', 'median', 'branch')

class ProbabilisticModel(ABC):
    """ Abstract base class for probabilistic models.
        
        Attributes
        ----------
        samples : pd.DataFrame
            Input features of the dataset
        QoI : pd.DataFrame
            Target values of the dataset
        """
    @abstractmethod
    def train(self, q_train, y_train, k_fold=None, **params):
        """ Train the probabilistic model.

            Parameters
            ----------
            q_train : pd.DataFrame
                Training input features
            y_train : pd.DataFrame
                Training target values
            k_fold : int, optional
                Number of folds for cross-validation (default is None)
            **params : dict
                Additional training parameters

            Returns
            -------
            None
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
        """ Compute the mean and variance of the model predictions.

            Parameters
            ----------
            n_sample : int, optional
                Number of samples to draw from the model (default is 1000).

            Returns
            -------
            mean : np.ndarray
                Mean of the model predictions.
            var : np.ndarray
                Variance of the model predictions.
        """
        q = self.Q.sample(n_sample)
        q_df = pd.DataFrame(q, columns=self.Q.variable_names())
        y_predict = self.predict(q_df)
        mean, var = y_predict.mean(axis=0), y_predict.var(axis=0)
        return mean, var

    def get_MAP(self, samples, space='physical', refine=True, n_starts=5, logpdf=None,
                bw_method='scott', max_kde_samples=5000, return_logpdf=False, random_seed=None):
        '''
            Joint maximum a posteriori (MAP) estimate from posterior samples.
    
            Estimates the mode of the JOINT density of the samples (not the modes of the
            1-D marginals, which generally do not form a point of high joint density).
            The density is a multivariate Gaussian KDE of the samples, or the exact
            log-density if ``logpdf`` is given. The best samples serve as starting points
            of a multistart Nelder-Mead search for the maximum.
    
            Parameters
            ----------
            samples : np.ndarray or pd.DataFrame, shape (n_samples, n_variables)
                Posterior samples in PHYSICAL parameter values, columns ordered as
                self.Q.variable_names().
            space : {'physical', 'germ'}, optional
                Space in which the density is maximized (default 'physical'). The mode
                is not invariant under nonlinear reparametrization: 'physical' gives the
                mode of the density of q, 'germ' the mode of the density of the germ
                variables (mapped back to q). 'germ' requires self.Q.variable2germ.
            refine : bool, optional
                Refine the best samples with a local optimizer (default True).
            n_starts : int, optional
                Number of highest-density samples used as optimizer starts (default 5).
            logpdf : callable, optional
                Exact log-density of the posterior, ``logpdf(X) -> (n,)`` for X of shape
                (n, n_variables) in the space given by ``space``. Overrides the KDE.
            bw_method : str, float or callable, optional
                KDE bandwidth, passed to scipy.stats.gaussian_kde (default 'scott').
            max_kde_samples : int, optional
                Samples used to build the KDE; larger sets are randomly subsampled to
                keep the O(n^2) evaluation affordable (default 5000).
            return_logpdf : bool, optional
                Also return the (estimated) log-density at the MAP (default False).
            random_seed : int, optional
                Seed for the subsampling.
    
            Returns
            -------
            map_df : pd.DataFrame
                One-row DataFrame with the MAP estimate of the parameters.
    
            Notes
            -----
            - KDE-based mode estimation is reliable in low dimension (roughly up to
            5-6 variables) and needs enough samples; it is biased near hard prior
            bounds (e.g. uniform priors).
            - If the posterior is a ridge or has several equally probable modes, the
            MAP is not unique and should not be reported as a single point estimate.
        '''
 
        names = self.Q.variable_names()
        X = np.asarray(samples, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(names):
            raise ValueError(f"samples have {X.shape[1]} columns, Q has {len(names)} variables")
        X = X[np.all(np.isfinite(X), axis=1)]
        if X.shape[0] < 2:
            raise ValueError("need at least two finite samples")
        if space not in ('physical', 'germ'):
            raise ValueError("space must be 'physical' or 'germ'")
 
        if space == 'germ':
            if not hasattr(self.Q, 'variable2germ'):
                raise AttributeError("space='germ' needs Q.variable2germ")
            X = np.asarray(self.Q.variable2germ(X), dtype=float)
 
        # Variables without spread (fixed / degenerate) make the KDE covariance
        # singular: keep them at their constant value and estimate the rest.
        spread = X.std(axis=0)
        active = spread > 1e-12 * np.maximum(1.0, np.abs(X).max(axis=0))
        x_fixed = np.median(X, axis=0)
 
        def embed(z):
            x = x_fixed.copy()
            x[active] = z
            return x
 
        if logpdf is not None:
            def log_density(Z):
                Xf = np.tile(x_fixed, (Z.shape[0], 1))
                Xf[:, active] = Z
                return np.asarray(logpdf(Xf), dtype=float).ravel()
        else:
            if not np.any(active):
                log_density = None
            else:
                Z_kde = X[:, active]
                if Z_kde.shape[0] > max_kde_samples:
                    rng = np.random.default_rng(random_seed)
                    Z_kde = Z_kde[rng.choice(Z_kde.shape[0], max_kde_samples, replace=False)]
                try:
                    kde = gaussian_kde(Z_kde.T, bw_method=bw_method)
                except np.linalg.LinAlgError as err:
                    raise np.linalg.LinAlgError(
                        "KDE covariance is singular: the samples lie on a lower-dimensional "
                        "set (linearly dependent parameters). The MAP is not well defined; "
                        "consider reparametrizing.") from err
                log_density = lambda Z: kde.logpdf(np.atleast_2d(Z).T)
 
        if log_density is None:            # all variables constant
            best_x, best_lp = x_fixed, np.nan
        else:
            Z = X[:, active]
            # candidate starts: highest-density samples (evaluated on a subset for speed)
            rng = np.random.default_rng(random_seed)
            idx = np.arange(Z.shape[0]) if Z.shape[0] <= max_kde_samples else \
                rng.choice(Z.shape[0], max_kde_samples, replace=False)
            lp = log_density(Z[idx])
            lp[~np.isfinite(lp)] = -np.inf
            order = idx[np.argsort(lp)[::-1]]
            lp_sorted = np.sort(lp)[::-1]
 
            starts, start_lp = [], []
            for i, l in zip(order, lp_sorted):
                if not any(np.allclose(Z[i], s) for s in starts):
                    starts.append(Z[i]); start_lp.append(float(l))
                if len(starts) >= max(n_starts, 1):
                    break
            best_z, best_lp = starts[0], start_lp[0]
 
            if refine:
                lo, hi = Z.min(axis=0), Z.max(axis=0)
                def neg_lp(z):
                    if np.any(z < lo) or np.any(z > hi):    # no extrapolation beyond the samples
                        return np.inf
                    v = float(np.asarray(log_density(np.atleast_2d(z))).ravel()[0])
                    return -v if np.isfinite(v) else np.inf
                for s in starts:
                    res = minimize(neg_lp, s, method='Nelder-Mead',
                                   options={'xatol': 1e-10, 'fatol': 1e-12,
                                            'maxiter': 2000 * Z.shape[1]})
                    if np.isfinite(res.fun) and -res.fun > best_lp:
                        best_z, best_lp = res.x, float(-res.fun)
            best_x = embed(best_z)
 
        map_estimate = best_x.reshape(1, -1)
        if space == 'germ':
            map_estimate = np.asarray(self.Q.germ2variable(map_estimate), dtype=float)
 
        map_df = pd.DataFrame(map_estimate, columns=names)
        return (map_df, best_lp) if return_logpdf else map_df

    def plot_posterior(self, X_train, samples, extra_point=None, extra_point_name='Extra point', map_point=None):
        """ Plot the posterior distribution of the model.

            Parameters
            ----------
            X_train : torch.Tensor
                The training input data.
            samples : torch.Tensor
                Samples from the posterior distribution.
            extra_point : torch.Tensor, optional
                An extra point to highlight on the plot (default is None).
            extra_point_name : str, optional
                The name of the extra point (default is 'Extra point').
            map_point : torch.Tensor, optional
                The maximum a posteriori (MAP) point to highlight on the plot (default is None).

            Returns
            -------
            fig : matplotlib.figure.Figure
                The figure object containing the posterior plot.
        """
        fig = posterior_plot(self, X_train, samples, extra_point=extra_point, extra_point_name=extra_point_name, map_point=map_point)
        return fig

    def _evaluate_model(self, y_train, X_test, y_test, verbose=False):
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
            name = name + ".pm"
            if path is not None:
                name = os.path.join(path, name)
            with open(name, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {name}")
        else:
            return pickle.dumps(self)

