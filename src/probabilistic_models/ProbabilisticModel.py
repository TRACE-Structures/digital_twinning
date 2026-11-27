import asyncio
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import io
import torch
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.special import ndtr
import numpy as np
from .plotting_functions import posterior_plot
from ..utils.weather_effect_subtraction import plot_effect_alert, plot_remained_effects_over_time
from ..utils.reference_mapping import outlier_alert
import os
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


POINT_ESTIMATES = ('mean', 'median', 'branch')

CHECKPOINT_FORMAT = 'ProbabilisticModel-torch'
CHECKPOINT_VERSION = 1


def _plain(obj):
    """ Convert to types that torch.load(weights_only=True) accepts (no arbitrary objects). """
    if isinstance(obj, np.generic):        # before float: np.float64 is a float subclass
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return str(obj)       # e.g. torch.device -> 'cuda:1'


def _tensor_from_bytes_on_cpu(data):
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)


class _CPUUnpickler(pickle.Unpickler):
    """ Unpickler that loads tensors pickled on a GPU onto the CPU. """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _tensor_from_bytes_on_cpu
        return super().find_class(module, name)


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
    def train(self, q_train, y_train, k_fold=None, device='cpu', **params):
        """ Train the probabilistic model.

            Parameters
            ----------
            q_train : pd.DataFrame
                Training input features
            y_train : pd.DataFrame
                Training target values
            k_fold : int, optional
                Number of folds for cross-validation (default is None)
            device : str, optional
                'cpu' (default) or an explicit GPU such as 'cuda:1'
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

    @staticmethod
    def _check_finite(q, y):
        """ Raise a clear error if the training data contain NaN or inf. """
        for name, data in (('q', q), ('y', y)):
            values = np.asarray(data, dtype=float)
            if not np.isfinite(values).all():
                rows = np.where(~np.isfinite(values.reshape(len(values), -1)).all(axis=1))[0]
                raise ValueError(f"{name} contains NaN or inf in {len(rows)} row(s), e.g. rows {rows[:5].tolist()}. "
                                 f"Remove or impute them before training.")

    def cross_validation(self, xi, yt, k_fold, random_seed=1997, **params):
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
            random_seed : int or None
                Seed of the fold shuffling, for reproducible folds (default 1997)
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
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
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
        if getattr(self, 'inverse', False):
            raise NotImplementedError("get_mean_and_var samples the parameters Q, but an inverse model "
                                      "takes QoIs as input. Use it on a forward model.")
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

    def training_parameters(self):
        """ Parameter values the model was trained on (the training part of the split;
            files saved before this change may hold all rows given to train()).

            Forward model: the inputs q; inverse model: the outputs. Used as the
            prior samples in plot_posterior.

            Returns
            -------
            pd.DataFrame
                Columns ordered as self.Q.variable_names().
        """
        attr = 'QoI' if getattr(self, 'inverse', False) else 'samples'
        data = getattr(self, attr, None)
        if data is None:
            raise AttributeError("The model holds no training data (not trained, or loaded from a '.pt' file, "
                                 "which stores no data). Pass X_train explicitly.")
        names = list(self.Q.variable_names())
        if isinstance(data, pd.DataFrame) and set(names) <= set(data.columns):
            return data[names].copy()
        return pd.DataFrame(np.asarray(data), columns=names)

    def plot_posterior(self, samples, extra_point=None, extra_point_name='Extra point', map_point=None,
                       X_train=None):
        """ Plot the posterior distribution of the model.

            Parameters
            ----------
            samples : array-like, shape (n_samples, n_variables)
                Samples from the posterior distribution.
            extra_point : array-like or pd.DataFrame, optional
                An extra point to highlight on the plot (default is None).
            extra_point_name : str, optional
                The name of the extra point (default is 'Extra point').
            map_point : pd.DataFrame, optional
                The maximum a posteriori (MAP) point to highlight on the plot (default is None).
            X_train : array-like, optional
                Prior samples. Defaults to the parameter values the model was trained
                on (see training_parameters).

            Returns
            -------
            fig : seaborn.PairGrid
                The figure object containing the posterior plot.
        """
        return posterior_plot(self, samples, extra_point=extra_point, extra_point_name=extra_point_name,
                              map_point=map_point, X_train=X_train)

    # ------------------------------------------------------------------ #
    #  Removing input effects (e.g. environment) from measured outputs     #
    # ------------------------------------------------------------------ #
    # Forward torch models (MDN, CRPS) implement
    #     _reference_map(q_obs, z_obs, q_ref, **options) -> (z_ref, joint_scores, marginal_scores)
    # on outputs scaled with self.y_scaler. StandardScaler is a per-output
    # increasing affine map, so percentiles (also the conditional ones of the
    # Rosenblatt transform) are the same in scaled and in physical units.

    def _reference_map(self, q_obs, z_obs, q_ref, **options):
        raise NotImplementedError(f"{type(self).__name__} does not support effect removal.")

    def reference_inputs(self, q, subtracted_params, reference=None):
        """ Inputs of the reference condition for every row of q.

            The columns in subtracted_params are set to their reference values; all
            other inputs keep their measured values.

            Parameters
            ----------
            q : pd.DataFrame or array-like, shape (n, n_inputs)
                Measured inputs (e.g. temperature, humidity), columns as self.input_names.
            subtracted_params : str or list of str
                Inputs whose effect is removed.
            reference : dict or pd.Series, optional
                Reference value of every subtracted input, e.g. {'T': 15, 'H': 60}.
                Default: mean of the training inputs (self.q_scaler.mean_).

            Returns
            -------
            q_obs, q_ref : pd.DataFrame
                Measured and reference inputs, columns ordered as self.input_names.
            reference : dict
                Reference values used.
        """
        if getattr(self, 'inverse', False):
            raise NotImplementedError("Effect removal needs a forward model (inverse=False): it maps outputs "
                                      "between the distributions predicted for two input conditions.")
        if not hasattr(self, 'q_scaler'):
            raise RuntimeError("The model is not trained yet.")
        names = list(self.input_names)
        if isinstance(q, pd.DataFrame):
            missing = [c for c in names if c not in q.columns]
            if missing:
                raise ValueError(f"q has no column(s) {missing}")
            q_obs = q[names].astype(float)
        else:
            values = np.asarray(q, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[1] != len(names):
                raise ValueError(f"q has {values.shape[1]} columns, the model has {len(names)} inputs {names}")
            q_obs = pd.DataFrame(values, columns=names)

        if isinstance(subtracted_params, str):
            subtracted_params = [subtracted_params]
        for x in subtracted_params:
            if x not in names:
                raise ValueError(f"There is no parameter called \'{x}\'")

        if reference is None:
            training_mean = dict(zip(names, np.asarray(self.q_scaler.mean_, dtype=float)))
            reference = {x: training_mean[x] for x in subtracted_params}
        else:
            reference = dict(reference)
            missing = [x for x in subtracted_params if x not in reference]
            if missing:
                raise ValueError(f"reference gives no value for {missing}")
            reference = {x: float(reference[x]) for x in subtracted_params}

        q_ref = q_obs.copy()
        for x, value in reference.items():
            q_ref[x] = value
        return q_obs, q_ref, reference

    def map_to_reference(self, q, QoI, subtracted_params, reference=None, silent=False, normal_scores=False,
                         marginal=False, **options):
        """ Remove the effect of some inputs from measured outputs by distribution mapping.

            Every measured output vector is moved to the same position (percentile)
            of the distribution predicted at the reference condition:

                f_ref = F^-1( F(f_obs | q_obs) | q_ref )

            where q_ref equals q_obs with the subtracted inputs set to their reference
            values. MDN: exact Rosenblatt transform of the predicted Gaussian mixture.
            CRPS: empirical marginals from samples and a Gaussian copula.

            Parameters
            ----------
            q : pd.DataFrame, shape (n, n_inputs)
                Measured inputs (e.g. environmental conditions).
            QoI : pd.DataFrame or array-like, shape (n, n_outputs)
                Measured outputs (e.g. frequencies and mode shape components), columns
                as self.output_names. The column order fixes the order of the
                Rosenblatt transform (it must be the training order for the MDN).
            subtracted_params : str or list of str
                Inputs whose effect is removed.
            reference : dict or pd.Series, optional
                Reference values of the subtracted inputs (default: training mean).
                It should be a realistic, well covered input combination; derived
                inputs (e.g. lagged temperature) need consistent values.
            silent : bool, optional
                Suppress messages (default False).
            normal_scores : bool, optional
                Return the normal scores Phi^-1(u) instead of the percentiles u
                (default False). They are N(0, 1) on healthy data and stay exact far
                in the tails, where u rounds to 0 or 1.
            marginal : bool, optional
                Return the marginal percentiles (each output in its own predicted
                distribution) instead of the joint Rosenblatt / copula percentiles
                (default False). Marginal ones show which output deviates; joint ones
                are independent and used for the alarm distance.
            **options :
                Model-specific settings, see MDN._reference_map / CRPS._reference_map
                (e.g. device, n_samples, seed, copula).

            Returns
            -------
            remained_effects : pd.DataFrame
                Outputs mapped to the reference condition, same index and columns as QoI.
            percentiles : pd.DataFrame
                Percentiles u in (0, 1) of the measurements (Rosenblatt / copula
                percentiles), or their normal scores with normal_scores=True. On healthy data and for a calibrated model they are
                independent U(0, 1); they are a natural basis for alarms.
        """
        remained, joint, marg = self._map_with_scores(q, QoI, subtracted_params, reference, silent, **options)
        scores = marg if marginal else joint
        return remained, (scores if normal_scores else scores.apply(ndtr))

    def _map_with_scores(self, q, QoI, subtracted_params, reference=None, silent=False, **options):
        """ map_to_reference returning (remained, joint scores, marginal scores) DataFrames. """
        q_obs, q_ref, reference = self.reference_inputs(q, subtracted_params, reference)

        names = list(self.output_names)
        if isinstance(QoI, pd.Series):
            QoI = QoI.to_frame()
        if isinstance(QoI, pd.DataFrame):
            index = QoI.index
            if set(names) <= set(QoI.columns):
                f_obs = QoI[names].to_numpy(dtype=float)
                columns = names
            elif QoI.shape[1] == len(names):
                f_obs, columns = QoI.to_numpy(dtype=float), list(QoI.columns)
            else:
                raise ValueError(f"QoI must have the columns {names}")
        else:
            f_obs = np.asarray(QoI, dtype=float)
            f_obs = f_obs.reshape(-1, 1) if f_obs.ndim == 1 else f_obs
            index, columns = q_obs.index if isinstance(q, pd.DataFrame) else None, names
        if f_obs.shape != (len(q_obs), len(names)):
            raise ValueError(f"QoI has shape {f_obs.shape}, expected {(len(q_obs), len(names))}")
        self._check_finite(q_obs, f_obs)

        if not silent:
            print(f"Message: removing the effect of {list(reference)} (reference {reference}) "
                  f"from {len(names)} output(s) of {len(q_obs)} measurement(s).")

        mean, scale = np.asarray(self.y_scaler.mean_), np.asarray(self.y_scaler.scale_)
        z_ref, joint, marg = self._reference_map(q_obs, (f_obs - mean) / scale, q_ref, **options)
        frame = lambda a: pd.DataFrame(a, columns=columns, index=index)
        return frame(z_ref * scale + mean), frame(joint), frame(marg)

    async def subtract_effects(self, q, QoI, subtracted_params, reference=None, return_percentiles=False,
                               silent=False, **options):
        """ Removes the effects of specified inputs from the QoI by distribution mapping.

            Same call as PredictiveModel.subtract_effects, but instead of subtracting
            SHAP contributions every measurement is mapped to the reference condition
            (see map_to_reference, which is the synchronous version).

            Parameters
            ----------
            q : pd.DataFrame
                Measured inputs.
            QoI : pd.DataFrame
                Measured Quantity of Interest values.
            subtracted_params : list
                Inputs whose effects are removed.
            reference : dict, optional
                Reference values of the subtracted inputs (default: training mean).
            return_percentiles : bool, optional
                Also return the percentiles of the measurements (default False).
            silent : bool, optional
                Suppress messages.
            **options :
                Model-specific settings, see map_to_reference.

            Returns
            -------
            remained_effects : pd.DataFrame
                QoI values at the reference condition.
            percentiles : pd.DataFrame
                Only with return_percentiles=True.
        """
        remained, percentiles = await asyncio.to_thread(self.map_to_reference, q, QoI, subtracted_params,
                                                        reference, silent, **options)
        return (remained, percentiles) if return_percentiles else remained

    async def plot_subtract_effects_and_alert(self, X_train, y_train, q, QoI, subtracted_params, threshold_ratio=0.1,
                                              xticks=True, silent=False, reference=None, alpha=0.01, **options):
        '''
        Removes the effects of specified inputs, plots the result and checks for alerts.

        Same call and return values as PredictiveModel.plot_subtract_effects_and_alert
        and the same figure (plot_effect_alert). The scores come from the predicted
        distributions at the measured conditions:
        - joint scores (Rosenblatt / copula): independent N(0, 1) on healthy data, so
          d2 = sum_j w_j^2 ~ chi2(D). A measurement is an outlier if d2 exceeds the
          chi2(D) quantile 1 - alpha.
        - marginal scores: each output in its own predicted distribution; shown per
          output and used for the per-output outlier ratios.
        The alert is raised if more than threshold_ratio of the monitoring
        measurements are outliers. The reference (training) data serve as a
        calibration check: their scores should also be N(0, 1); use held-out healthy
        data for an honest check.

        Parameters:
        -----------
        X_train, y_train : pd.DataFrame
            Healthy reference inputs and outputs (training or held-out data).
        q : pd.DataFrame
            Inputs for monitoring.
        QoI : pd.DataFrame
            Measured outputs for monitoring. A DatetimeIndex is used as time axis.
        subtracted_params : list
            Inputs whose effects are removed.
        threshold_ratio : float
            Alert if more than this fraction of the monitoring measurements are
            outliers (default 0.1; healthy data give about alpha).
        xticks : bool
            Show the output names on the per-output panel (default True).
        silent : bool
            Suppress messages.
        reference : dict, optional
            Reference values of the subtracted inputs (default: training mean).
        alpha : float
            Outlier level of a single measurement (default 0.01).
        **options :
            Model-specific mapping settings, see map_to_reference.

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Figure object of the plot
        alert : bool
            True if an alert is triggered, False otherwise
        remained_effects_df : pd.DataFrame
            Monitoring outputs mapped to the reference condition
        error_ratio : float
            Fraction of monitoring measurements that are outliers
        problematic_rows : pd.DataFrame
            Marginal scores of the outlying measurements and their distance 'd2'
        reference_statistics : pd.DataFrame
            Statistics of the mapped reference outputs (with the outlier ratio)
        monitoring_statistics : pd.DataFrame
            Statistics of the mapped monitoring outputs (with the outlier ratio)
        '''
        remained_ref, joint_ref, marg_ref = await asyncio.to_thread(
            self._map_with_scores, X_train, y_train, subtracted_params, reference, silent, **options)
        remained_mon, joint_mon, marg_mon = await asyncio.to_thread(
            self._map_with_scores, q, QoI, subtracted_params, reference, silent, **options)
        res = outlier_alert(joint_ref, joint_mon, marg_ref, marg_mon, remained_ref, remained_mon, alpha,
                            threshold_ratio)
        if not silent:
            print(f"Message: {res['error_ratio']:.1%} of the monitoring measurements are outliers at level "
                  f"alpha={alpha} (reference data: {res['reference_ratio']:.1%}); alert threshold "
                  f"{threshold_ratio:.1%} -> {'ALERT' if res['alert'] else 'no alert'}.")
        fig = plot_effect_alert(res['d2_ref'], res['d2_mon'], marg_ref, marg_mon, res['limit'], alpha,
                                res['z_single'], res['alert'], threshold_ratio, xticks=xticks,
                                reference_is_check=True, method=f'{type(self).__name__} distribution mapping')
        return (fig, res['alert'], remained_mon, res['error_ratio'], res['problematic_rows'],
                res['reference_statistics'], res['monitoring_statistics'])

    async def plot_remained_effects_over_time(self, X_train, y_train, q, QoI, subtracted_params, outputs=None,
                                              band=(0.01, 0.99), ratio=0.05, silent=False, reference=None, **options):
        '''
        Mapped outputs of the monitoring data over time, in physical units, against the
        band of the mapped reference data (see plotting_functions.plot_remained_effects_over_time).

        Parameters:
        -----------
        X_train, y_train, q, QoI, subtracted_params, reference, **options :
            As in plot_subtract_effects_and_alert.
        outputs : list of str, optional
            Outputs to show (default: the first 6).
        band : tuple of float
            Quantiles of the reference band (default 1% - 99%).

        Returns:
        -------
        fig : matplotlib.figure.Figure
        '''
        remained_ref, _ = await self.subtract_effects(X_train, y_train, subtracted_params, reference=reference,
                                                      return_percentiles=True, silent=silent, **options)
        remained_mon, _ = await self.subtract_effects(q, QoI, subtracted_params, reference=reference,
                                                      return_percentiles=True, silent=silent, **options)
        return plot_remained_effects_over_time(remained_ref, remained_mon, outputs=outputs, band=band, ratio=ratio)

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
            Prepares the instance state for pickling. The training data stored by
            train_test_split (samples, QoI) are kept.

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
        # Torch models are kept on the CPU between calls, also when an old file
        # pickled them on a GPU (see load_model).
        model = state.get('model')
        if isinstance(model, torch.nn.Module):
            model.to('cpu')


    # ------------------------------------------------------------------ #
    #  Saving and loading                                                  #
    # ------------------------------------------------------------------ #
    # Torch-based subclasses implement _config() (JSON-like constructor arguments
    # except Q) and list further plain attributes to keep in _SAVED_ATTRS. They can
    # then also be saved without pickle (format='pt'): weights, scalers and
    # settings only, no training data.
    _SAVED_ATTRS = ()

    def _config(self):
        raise NotImplementedError

    @staticmethod
    def _scaler_state(scaler):
        state = {'with_mean': bool(scaler.with_mean), 'with_std': bool(scaler.with_std),
                 'n_samples_seen': np.asarray(scaler.n_samples_seen_).tolist()}
        for attr in ('mean_', 'scale_', 'var_'):
            value = getattr(scaler, attr, None)
            state[attr.rstrip('_')] = None if value is None else np.asarray(value, dtype=float).tolist()
        if hasattr(scaler, 'feature_names_in_'):
            state['feature_names'] = [str(c) for c in scaler.feature_names_in_]
        return state

    @staticmethod
    def _scaler_from_state(state):
        scaler = StandardScaler(with_mean=state['with_mean'], with_std=state['with_std'])
        for attr in ('mean', 'scale', 'var'):
            value = state[attr]
            setattr(scaler, attr + '_', None if value is None else np.asarray(value, dtype=float))
        n_seen = np.asarray(state['n_samples_seen'], dtype=np.int64)
        scaler.n_samples_seen_ = n_seen if n_seen.ndim else n_seen.item()
        reference = next(state[a] for a in ('mean', 'scale', 'var') if state[a] is not None)
        scaler.n_features_in_ = len(reference)
        if 'feature_names' in state:
            scaler.feature_names_in_ = np.asarray(state['feature_names'], dtype=object)
        return scaler

    def _to_checkpoint(self):
        """ Plain dict with everything needed to rebuild the model (no pickled objects). """
        if not hasattr(self, 'q_scaler'):
            raise RuntimeError("The model is not trained yet; nothing to save.")
        return {
            'format': CHECKPOINT_FORMAT,
            'version': CHECKPOINT_VERSION,
            'class': type(self).__name__,
            'config': _plain(self._config()),
            'input_names': _plain(list(self.input_names)),
            'output_names': _plain(list(self.output_names)),
            'q_scaler': self._scaler_state(self.q_scaler),
            'y_scaler': self._scaler_state(self.y_scaler),
            'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            'extra': {a: _plain(getattr(self, a)) for a in self._SAVED_ATTRS if hasattr(self, a)},
        }

    @classmethod
    def _subclass_named(cls, name):
        if cls.__name__ == name:
            return cls
        for sub in cls.__subclasses__():
            found = sub._subclass_named(name)
            if found is not None:
                return found
        return None

    @classmethod
    def load_model(cls, source, Q=None, trusted=False):
        '''
        Loads a model written by save_model, on the CPU. The file format is detected.

        - Pickled file ('.sm', also older '.pm'): the full object with all its data,
          including Q and the training data. Unpickling can run arbitrary code, so
          only files you created yourself should be opened: pass trusted=True.
          Tensors pickled on a GPU are mapped to the CPU.
        - '.pt' file: weights, scalers and settings only; read with
          weights_only=True, so a tampered file cannot run code. Needs Q.

        Parameters:
        -----------
        source : str, os.PathLike or bytes
            File path, or the bytes returned by save_model() without a name.
        Q : object, optional
            Variable set of the model parameters. Required for '.pt' files (its
            variable names must match the saved model); ignored for pickled files.
        trusted : bool, optional
            Allow loading a pickled file (default False).

        Returns:
        -------
        ProbabilisticModel
        '''
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            with open(source, 'rb') as file:
                raw = file.read()

        if raw[:2] != b'PK':                 # torch.save writes a zip archive; anything else is a pickle
            if not trusted:
                raise ValueError("This is a pickled model file. Unpickling can run arbitrary code, so only load "
                                 "files you created yourself: pass trusted=True.")
            obj = _CPUUnpickler(io.BytesIO(raw)).load()
            if not isinstance(obj, cls):
                raise ValueError(f"The file holds a {type(obj).__name__}, not a {cls.__name__}.")
            return obj

        ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if not isinstance(ckpt, dict) or ckpt.get('format') != CHECKPOINT_FORMAT:
            raise ValueError("Not a model file written by save_model().")
        if ckpt['version'] > CHECKPOINT_VERSION:
            raise ValueError(f"File format version {ckpt['version']} is newer than this code supports.")
        model_cls = cls._subclass_named(ckpt['class'])
        if model_cls is None:
            raise ValueError(f"The file holds a {ckpt['class']} model; load it with {ckpt['class']}.load_model.")
        if Q is None:
            raise ValueError("A '.pt' file does not store Q; pass the variable set as Q.")

        obj = model_cls(Q, **ckpt['config'])
        if list(obj.input_names) != ckpt['input_names'] or list(obj.output_names) != ckpt['output_names']:
            raise ValueError(f"Q does not match the saved model: expected inputs {ckpt['input_names']} "
                             f"and outputs {ckpt['output_names']}, got {list(obj.input_names)} and "
                             f"{list(obj.output_names)}.")
        obj.q_scaler = cls._scaler_from_state(ckpt['q_scaler'])
        obj.y_scaler = cls._scaler_from_state(ckpt['y_scaler'])
        obj.model.load_state_dict(ckpt['state_dict'])
        obj.model.eval()
        for attr, value in ckpt['extra'].items():
            setattr(obj, attr, value)
        return obj

    def save_model(self, name=None, path=None, format='pickle'):
        '''
        Saves the data-driven model to a file or returns a byte stream.

        Parameters:
        -----------
        name : str, optional
            Name of the file to save the model, without extension (default is None).
        path : str, optional
            Directory path to save the file (default is None).
        format : {'pickle', 'pt'}, optional
            'pickle' (default): the whole object with all its data (training data,
                Q, history) as a '.sm' file. Load with load_model(file, trusted=True).
            'pt': torch models (MDN, CRPS) only. Weights, scalers and settings, no
                training data and no Q; safe to load from untrusted sources.
                Load with load_model(file, Q).

        Returns:
        -------
        None or bytes
            If name is provided, saves the model to a file. Otherwise, returns a byte stream.
        '''
        if format == 'pickle':
            data, extension = pickle.dumps(self), '.sm'
        elif format == 'pt':
            try:
                self._config()
            except NotImplementedError:
                raise ValueError(f"format='pt' is only available for torch models, not {type(self).__name__}.")
            buffer = io.BytesIO()
            torch.save(self._to_checkpoint(), buffer)
            data, extension = buffer.getvalue(), '.pt'
        else:
            raise ValueError(f"format must be 'pickle' or 'pt', got {format!r}")

        if name is None:
            return data
        name = name + extension
        if path is not None:
            name = os.path.join(path, name)
        with open(name, 'wb') as file:
            file.write(data)
        print(f"Model saved to {name}")
