from mosaictools import Mosaic
import re
import numpy as np
from sklearn.metrics import mean_squared_error
from SALib.analyze import sobol
import pandas as pd
import shap
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


FREQ_RE = re.compile(r"^f\.(\d+)$")
EIGVEC_RE = re.compile(r"^S\.(\d+)\.(\d+)\.([xyzXYZ])$")
_DIR_PRIORITY = {"x": 0, "y": 1, "z": 2}  # preferred display order when inferring

class MosaicModel:
    def __init__(self, Q, QoI_names, resolution=0.6, max_freq_degree=4, max_vect_degree=5, classification_method='svc', **class_kwargs):
        self.Q = Q
        self.resolution = resolution
        self.max_freq_degree = max_freq_degree
        self.max_vect_degree = max_vect_degree
        self.classification_method = classification_method
        self.class_kwargs = class_kwargs

        self.model = Mosaic(Q, resolution=resolution, max_freq_degree=max_freq_degree, max_vect_degree=max_vect_degree, classification_method=classification_method, **class_kwargs)
        self.Q = Q
        self.QoI_names = QoI_names

    def reorder_header(self, columns, m=None, n=None, directions=None):
        """
        Validate that `columns` contains exactly the expected set of
        'f.<i>' and 'S.<i>.<j>.<d>' columns for m modes, n nodes, and the
        given directions -- in ANY order -- and return them reordered into
        canonical order. Raises ValueError on any unrecognized, missing,
        or out-of-range column.
        """
        columns = list(columns)

        freq_found, eigvec_found, unrecognized = {}, {}, []
        for col in columns:
            fm = FREQ_RE.match(col)
            if fm:
                freq_found[int(fm.group(1))] = col
                continue
            em = EIGVEC_RE.match(col)
            if em:
                mode_id, node_id, dir_id = int(em.group(1)), int(em.group(2)), em.group(3).lower()
                eigvec_found[(mode_id, node_id, dir_id)] = col
                continue
            unrecognized.append(col)

        if unrecognized:
            raise ValueError(
                f"Header contains columns that don't match 'f.<mode>' or "
                f"'S.<mode>.<node>.<dir>': {unrecognized}"
            )

        if m is None:
            if not freq_found:
                raise ValueError("No 'f.<mode>' columns found; cannot infer number of modes.")
            m = max(freq_found)

        if directions is None:
            found_dirs = {d for (_, _, d) in eigvec_found}
            if not found_dirs:
                raise ValueError("No 'S.<mode>.<node>.<dir>' columns found; cannot infer directions.")
            directions = sorted(found_dirs, key=lambda d: (_DIR_PRIORITY.get(d, 99), d))
        else:
            directions = [d.lower() for d in directions]

        if n is None:
            found_nodes = {node for (_, node, _) in eigvec_found}
            if not found_nodes:
                raise ValueError("No 'S.<mode>.<node>.<dir>' columns found; cannot infer number of nodes.")
            n = max(found_nodes)

        expected_freq = [f"f.{i}" for i in range(1, m + 1)]
        expected_eigvec = [(i, j, d) for i in range(1, m + 1) for j in range(1, n + 1) for d in directions]

        missing_freq = [i for i in range(1, m + 1) if i not in freq_found]
        missing_eigvec = [key for key in expected_eigvec if key not in eigvec_found]
        if missing_freq or missing_eigvec:
            parts = []
            if missing_freq:
                parts.append("missing frequency columns: " + ", ".join(f"f.{i}" for i in missing_freq))
            if missing_eigvec:
                parts.append("missing eigenvector columns: " + ", ".join(f"S.{i}.{j}.{d}" for i, j, d in missing_eigvec))
            raise ValueError("Header is incomplete -- " + "; ".join(parts))

        extra_freq = [freq_found[i] for i in freq_found if i not in range(1, m + 1)]
        extra_eigvec = [eigvec_found[key] for key in eigvec_found if key not in expected_eigvec]
        if extra_freq or extra_eigvec:
            raise ValueError(
                f"Header contains unexpected entries outside the expected "
                f"m={m}, n={n}, directions={directions}: {extra_freq + extra_eigvec}"
            )

        return expected_freq + [f"S.{i}.{j}.{d}" for (i, j, d) in expected_eigvec]

    def reorder_dataframe(self, df, m=None, n=None, directions=None):
        """Returns df with columns reordered canonically. Raises ValueError if invalid."""
        order = self.reorder_header(df.columns, m=m, n=n, directions=directions)
        return df[order]

    def _infer_dims(self, columns, directions=None):
        """Infer (m, n, directions) from column names, without validation.
        Assumes the columns already form a complete, correctly named set
        (e.g. as returned by reorder_dataframe)."""
        columns = list(columns)
        m = 0
        for col in columns:
            fm = FREQ_RE.match(col)
            if fm:
                m = max(m, int(fm.group(1)))

        found_nodes, found_dirs = set(), set()
        for col in columns:
            em = EIGVEC_RE.match(col)
            if em:
                found_nodes.add(int(em.group(2)))
                found_dirs.add(em.group(3).lower())
        n = max(found_nodes) if found_nodes else 0

        if directions is None:
            directions = sorted(found_dirs, key=lambda d: (_DIR_PRIORITY.get(d, 99), d))
        else:
            directions = [d.lower() for d in directions]

        return m, n, directions

    def extract_arrays(self, df, m=None, n=None, directions=None):
        """
        Extract frequencies and eigenvectors from a DataFrame whose header is
        already known-valid and in canonical order (e.g. the output of
        reorder_dataframe). Does NOT re-validate the header.

        Returns
        -------
        frequencies : ndarray, shape (n_samples, n_modes)
        eigenvectors: ndarray, shape (n_samples, n_modes, n_nodes * n_directions)
            Last axis ordered as: all nodes for direction[0], then all nodes for
            direction[1], ... i.e. [d0_node1, ..., d0_nodeN, d1_node1, ..., dD_nodeN]
        """
        m, n, directions = self._infer_dims(df.columns, directions=directions) if (m is None or n is None or directions is None) \
            else (m, n, [d.lower() for d in directions])

        freq_cols = [f"f.{i}" for i in range(1, m + 1)]
        frequencies = df[freq_cols].to_numpy(dtype=np.float64)

        n_samples = len(df)
        eigenvectors = np.empty((n_samples, m, n * len(directions)), dtype=np.float64)
        for mode_idx, i in enumerate(range(1, m + 1)):
            cols = [f"S.{i}.{j}.{d}" for d in directions for j in range(1, n + 1)]
            eigenvectors[:, mode_idx, :] = df[cols].to_numpy(dtype=np.float64)

        return frequencies, eigenvectors

    def train_and_validate(self, X_train, y_train, X_val, y_val):
        X_train = X_train.values
        y_train = self.reorder_dataframe(y_train)
        train_freq, train_eig = self.extract_arrays(y_train)

        self.model.fit(X_train, train_freq, train_eig)
        y_pred_tr = self.model.predict(X_train)
        tr_loss = mean_squared_error(y_train.values, y_pred_tr)

        X_val = X_val.values
        y_val = self.reorder_dataframe(y_val)

        y_pred_vl = self.model.predict(X_val)
        vl_loss = mean_squared_error(y_val.values, y_pred_vl)
        return tr_loss, vl_loss


    def predict(self, X):
        '''
        Makes predictions using the trained Linear Regression model.

        Parameters
        ----------
        X : array-like
            Input feature data for prediction.

        Returns
        -------
        predictions : array-like
            Predicted target values.
        '''

        X = X.values
        return self.model.predict(X)

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

    def evaluate_model(self, y_train, X_test, y_test, verbose=False):
        '''
        Evaluates the model using various statistical metrics.

        Parameters
        ----------
        y_train : array-like
            Training target data.
        X_test : array-like
            Test feature data.
        y_test : array-like
            Test target data.
        verbose : bool, optional
            If True, prints the evaluation metrics. Default is False.

        Returns
        -------
        results : list
            A list containing a DataFrame of evaluation metrics and a dictionary of the same metrics.
        '''

        pred = self.model.predict(X_test)
        model_eval = {
            "Kendall_tau": kendalltau(y_test, pred)[0],
            "Pearson":  pearsonr(y_test.squeeze(), pred.squeeze())[0],
            "Spearman": spearmanr(y_test, pred)[0],
            "MSE":  mean_squared_error(y_test, pred),
            "MAE":  mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "STD_of_label": y_train.std()
            }
        if verbose==True:
            print('Kendall tau correlation - measure of correspondance between two rankings: %.3f' %model_eval["Kendall_tau"])
            print('Pearson correlation - measure of linear realationship (cov normalised): %.3f' %model_eval["Pearson"])
            print('Spearman correaltion - cov(rank(y1), rank(y2)/stdv(rank(y1))): %.3f' %model_eval["Spearman"])
            print("mean squared error:", model_eval["RMSE"])
            print("STD of label:", model_eval["STD_of_label"])
        df = pd.DataFrame(model_eval, index=[0])
        df['label'] = y_train.name if hasattr(y_train, 'name') else 'label'
        df['rel_RMSE'] = df['RMSE'] / df['STD_of_label']
        df['summed_metric'] = (df['Kendall_tau'] + df['Pearson'] + df['Spearman'] - df['rel_RMSE']) / 3
        
        return [df, model_eval]

    def compute_partial_vars(self, model_obj, max_index):
        '''
        Computes partial variances using Sobol sensitivity analysis.

        Parameters
        ----------
        model_obj : LinRegModel
            The linear regression model object.
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

        variableset = model_obj.Q
        QoI_names = model_obj.QoI_names

        problem = {
            'num_vars': variableset.num_variables(), 'names': variableset.variable_names(), 'dists': variableset.get_dist_types(), 'bounds': variableset.get_dist_params()
            } 
        
        d = variableset.num_variables()
        q = variableset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
        y = model_obj.predict(q)
        
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
                    
        # Compute partial variances
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var
        

    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
        '''
        Computes SHAP values for model interpretability.
        
        Parameters
        ----------
        predict_fn : function
            The prediction function of the model.
        q : array-like
            Input data for SHAP value computation.
        forced : bool, optional
            If True, forces re-computation of the SHAP explainer. Default is False.
        explainer_type : str, optional
            Type of SHAP explainer to use. Default is "kernelexplainer".
        silent : bool, optional
            If True, suppresses output during SHAP value computation. Default is False.
        Returns
        -------
        shap_values : array-like
            Computed SHAP values.
        '''

        if explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
        shap_values = self.explainer(q, silent=silent)
        return shap_values


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
        for attr in ['explainer']:
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