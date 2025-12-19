'''Linear Regression Model for Predictive Modelling'''

import numpy as np
import pandas as pd
import shap
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LinRegModel:
    '''
    Linear Regression Model for Predictive Modelling
    
    Attributes
    ----------
    model : LinearRegression
        The linear regression model instance.

    Q : VariableSet
        The variable set used for the model.

    QoI_names : list
        List of Quantity of Interest names.

    Methods
    -------
    __init__(self, Q, QoI_names)
        Initializes the LinRegModel with a parameter set and QoI names.

    train_and_evaluate(self, X_train, y_train, X_val, y_val)
        Trains the model and evaluates it on training and validation data.

    predict(self, X)
        Makes predictions using the trained model.

    score(self, X, y)
        Computes the mean squared error of the model on given data.

    evaluate_model(self, y_train, X_test, y_test, verbose=False)
        Evaluates the model using various statistical metrics.

    compute_partial_vars(self, model_obj, max_index)
        Computes partial variances using Sobol sensitivity analysis.

    get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer")
        Computes SHAP values for model interpretability.

    to_jsonld(self, model_id: str)
        Serializes the model to JSON-LD format.
    '''

    def __init__(self, Q, QoI_names):
        '''
        LinRegModel Constructor

        Parameters
        ----------
        Q : VariableSet
            The variable set for the model.
        QoI_names : list
            List of Quantity of Interest names.
        '''
        
        self.model = LinearRegression()
        self.Q = Q
        self.QoI_names = QoI_names

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        '''
        Trains the Linear Regression model and evaluates it on training and validation data.

        Parameters
        ----------
        X_train : array-like
            Training feature data.
        y_train : array-like
            Training target data.
        X_val : array-like
            Validation feature data.
        y_val : array-like
            Validation target data.

        Returns
        -------
        tr_loss : float
            Mean squared error on the training data.
        vl_loss : float
            Mean squared error on the validation data.
        '''

        self.model.fit(X_train, y_train)
        y_pred_tr = self.model.predict(X_train)
        tr_loss = mean_squared_error(y_train, y_pred_tr)
        y_pred_vl = self.model.predict(X_val)
        vl_loss = mean_squared_error(y_val, y_pred_vl)
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

        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names

        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.get_dist_types(), 'bounds': paramset.get_dist_params()
            } 
        
        d = paramset.num_params()
        q = paramset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
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
    
    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer"):
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
        Returns
        -------
        shap_values : array-like
            Computed SHAP values.
        '''

        if explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
        shap_values = self.explainer(q)
        return shap_values

    def to_jsonld(self, model_id: str):
        '''
        Serializes the model to JSON-LD format.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        jsonld : dict
            JSON-LD representation of the model.
        '''

        jsonld = {
            "@context": {
                "mls": "https://ml-schema.github.io/documentation/mls.html",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },

            "@id": f"https://example.org/models/{model_id}",
            "@type": "mls:Model",
            "mls:implementsAlgorithm": {
                "@id": "https://en.wikipedia.org/wiki/Linear_regression",
                "@type": "mls:Algorithm",
                "rdfs:label": "Linear Regression"
            },

            "mls:hasHyperParameter": [],

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