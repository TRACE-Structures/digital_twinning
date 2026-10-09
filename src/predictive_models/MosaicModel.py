import mosaictools as mosaic
from mosaictools import Mosaic
import re
import numpy as np
from sklearn.metrics import mean_squared_error
import shap
from .PredictiveModel import PredictiveModel


FREQ_RE = re.compile(r"^f\.(\d+)$")
EIGVEC_RE = re.compile(r"^S\.(\d+)\.(\d+)\.([xyzXYZ])$")
_DIR_PRIORITY = {"x": 0, "y": 1, "z": 2}  # preferred display order when inferring

class MosaicModel(PredictiveModel):
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

    def train(self, q_train, y_train, k_fold=None):

        xi = q_train
        yt = y_train

        print(f'----- Training started for \'MosaicModel\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold)
        else:
            # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = self.train_test_split(xi, yt, train_test_ratio=0.8, random_seed=1997, split_type='shuffle')
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for MosaicModel model -----')


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
        return self.model.predict(X)        

    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
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

    def plot_variable_space_exploration(self, mode, n_samples=1000):
        """ Plot the variable space exploration for a Mosaic model. Returns three figures:
                1. Space division plot
                2. Reference eigenvectors plot
                3. Reference correlation matrix plot
            
            Parameters
            ----------
            model : MosaicModel
                The Mosaic model instance to be visualized.
            mode : int
                The index of mode for plotting.
            n_samples : int, optional
                The number of samples to use for the space division plot (default is 1000). """
        list_of_subdomains = self.model.get_number_of_subdomains()
        fig1 = mosaic.plot_reference_eigenvectors(self.model, mode)
        if list_of_subdomains[mode-1] == 1:
            return fig1, None, None
        else:
            fig2 = mosaic.plot_space_division(self.model, mode, n_samples)
            fig3 = mosaic.plot_reference_correlation_matrix(self.model, mode)
        return fig1, fig2, fig3

    def get_number_of_subdomains(self):
        """ Get the number of subdomains for a Mosaic model. 
        
            Parameters
            ----------
            model : MosaicModel
                The Mosaic model instance.

            Returns
            -------
            list_of_subdomains : list of int
                The number of subdomains for the Mosaic model. """
        
        list_of_subdomains = self.model.get_number_of_subdomains()
        return list_of_subdomains