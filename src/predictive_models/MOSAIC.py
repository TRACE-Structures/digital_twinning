import mosaictools as mosaic
from mosaictools import Mosaic
import numpy as np
from sklearn.metrics import mean_squared_error
import shap
from .PredictiveModel import PredictiveModel
import pandas as pd
from .utils import reorder_dataframe, extract_arrays

class MOSAIC(PredictiveModel):
    """ MOSAIC predictive model class for handling frequency and eigenvector data.
    
        Attributes
        ----------
        Q : object
            Input feature data.
        QoI_names : list
            Names of the quantities of interest.
        resolution : float
            Resolution parameter for the MOSAIC model.
        max_freq_degree : int
            Maximum degree for frequency features.
        max_vect_degree : int
            Maximum degree for eigenvector features.
        classification_method : str
            Classification method used in the MOSAIC model.
        class_kwargs : dict
            Additional keyword arguments for the classification method.
        model : Mosaic
            The underlying MOSAIC model instance.
    """
    def __init__(self, Q, QoI_names, resolution=0.6, max_freq_degree=4, max_vect_degree=5, classification_method='svc', **class_kwargs):
        """
        Initializes the MOSAIC predictive model with the given parameters.

        Parameters
        ----------
        Q : object
            Input feature data.
        QoI_names : list
            Names of the quantities of interest.
        resolution : float, optional
            Resolution parameter for the MOSAIC model. Default is 0.6.
        max_freq_degree : int, optional
            Maximum degree for frequency features. Default is 4.
        max_vect_degree : int, optional
            Maximum degree for eigenvector features. Default is 5.
        classification_method : str, optional
            Classification method used in the MOSAIC model. Default is 'svc'.
        **class_kwargs : dict, optional
            Additional keyword arguments for the classification method.
        """
        self.Q = Q
        self.resolution = resolution
        self.max_freq_degree = max_freq_degree
        self.max_vect_degree = max_vect_degree
        self.classification_method = classification_method
        self.class_kwargs = class_kwargs

        self.model = Mosaic(Q, resolution=resolution, max_freq_degree=max_freq_degree, max_vect_degree=max_vect_degree, classification_method=classification_method, **class_kwargs)
        self.Q = Q
        self.QoI_names = QoI_names


    def train(self, q_train, y_train, k_fold=None):
        """ Trains the Mosaic model using the provided training data.

            Parameters
            ----------
            q_train : array-like
                Input feature data for training.
            y_train : array-like
                Target data for training.
            k_fold : int, optional
                Number of folds for cross-validation. If None, no cross-validation is performed.

            Returns
            -------
            None
        """

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
        """ Trains the model on the provided training and validation data and computes the training and validation loss.

            Parameters
            ----------
            X_train : array-like
                Input feature data for training.
            y_train : array-like
                Target data for training.
            X_val : array-like
                Input feature data for validation.
            y_val : array-like
                Target data for validation.

            Returns
            -------
            tr_loss : float
                Training loss.
            vl_loss : float
                Validation loss.
        """
        # if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
        y_train = reorder_dataframe(y_train)
        train_freq, train_eig = extract_arrays(y_train)

        self.model.fit(X_train, train_freq, train_eig)
        y_pred_tr = self.model.predict(X_train)
        tr_loss = mean_squared_error(y_train.values, y_pred_tr)

        X_val = X_val.values
        y_val = reorder_dataframe(y_val)

        y_pred_vl = self.model.predict(X_val)
        vl_loss = mean_squared_error(y_val.values, y_pred_vl)
        return tr_loss, vl_loss

    def modal_validation(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """ Performs modal validation using the test data.

            Parameters
            ----------
            X_test : pd.DataFrame
                Input feature data for testing.
            y_test : pd.DataFrame
                Target data for testing.

            Returns
            -------
            freq_errors : array-like
                Frequency errors.
            mac_errors : array-like
                MAC errors.
        """
        test_freq, test_eig = extract_arrays(y_test)
        freq_errors, mac_errors = self.model.validate(X_test.values, test_freq, test_eig)
        return freq_errors, mac_errors

    def numpy_train(self, X_train, train_freq, train_eig):
        """ Trains the model using the provided training data in structured NumPy arrays.

            Parameters
            ----------
            X_train : array-like
                Input feature data for training.
            train_freq : array-like
                Frequency data for training.
            train_eig : array-like
                Eigenvector data for training.

            Returns
            -------
            None
        """
        self.model.fit(X_train, train_freq, train_eig)

    def predict(self, X):
        ''' Makes predictions using the trained Linear Regression model.

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
        ''' Computes SHAP values for model interpretability.
        
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

    def plot_variable_space_exploration(self, mode):
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
            fig2, _ = mosaic.plot_classification_regions(self.model, mode)
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