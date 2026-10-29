from .PredictiveModel import PredictiveModel
from gPCE_model import GpcModel
import numpy as np
import pandas as pd
import shap
from scipy.sparse import csr_matrix
import copy
from gPCE_model import gpc_multiindex2param_names
import datetime

class gPCE(PredictiveModel):
    """ gPCE model class for predictive modeling.

        Attributes
        ----------
        Q : VariableSet
            Variable set defining the input features.
        QoI_names : list
            List of Quantity of Interest (QoI) names.
        p : int
            Polynomial order for the gPCE model.
        model : GpcModel
            Instance of the gPCE model.
    """

    def __init__(self, Q, QoI_names, p=4):
        '''
        Initializes the PredictiveModel with the given parameters.

        Parameters
        ----------
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
        self.p = p
        self.model = GpcModel(Q, p)

    def train(self, q_train, y_train, k_fold=None):
        """ Trains the gPCE model with optional k-fold cross-validation.

            Parameters
            ----------
            q_train : array_like or pandas.DataFrame
                Training input features.
            y_train : array_like or pandas.DataFrame
                Training target features.
            k_fold : int, optional
                Number of folds for cross-validation. Default is None (no cross-validation).
        """

        xi = q_train.values
        yt = y_train.values

        print(f'----- Training started for \'gPCEModel\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold)
        else:
            # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = self.train_test_split(xi, yt, train_test_ratio=0.8, random_seed=1997, split_type='shuffle')
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for gPCEModel model -----')


    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val):
        """ Trains the gPCE model on the training set and validates it on the validation set.

            Parameters
            ----------
            xi_train : array_like
                Training input features.
            yt_train : array_like
                Training target features.
            xi_val : array_like
                Validation input features.
            yt_val : array_like
                Validation target features.

            Returns
            -------
            tr_loss : float
                Training loss.
            vl_loss : float
                Validation loss.
        """
        tr_loss, vl_loss = self.model.train_and_evaluate(xi_train, yt_train, xi_val, yt_val)
        return tr_loss, vl_loss


    def predict(self, q):
        """ Predicts the target features for the given input features using the trained gPCE model.

            Parameters
            ----------
            q : array_like or pandas.DataFrame
                Input features for prediction.

            Returns
            -------
            y : array_like
                Predicted target features.
        """
        if type(q) != pd.DataFrame:
            q = pd.DataFrame(q, columns=self.Q.variable_names())
        xi = q.values
        y = self.model.predict(xi)
        return y

    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
        """ Compute SHAP values for the gPCE model predictions

            Parameters
            ----------
            predict_fn : function
                Prediction function of the gPCE model

            q : array_like
                Input parameter samples

            forced : bool, optional
                If True, force re-computation of SHAP values, by default False

            explainer_type : str, optional
                Type of SHAP explainer to use, by default "kernelexplainer"

            Returns
            -------
            shap_values : numpy.array
                SHAP values for the gPCE model predictions """
        
        if explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
            shap_values = self.explainer(q, silent=silent)
        return shap_values

    def compute_partial_vars(self, max_index=1):
        """ Compute partial variances and Sobol indices up to specified order

            Parameters
            ----------
            model_obj : GpcModel
                gPCE model object

            max_index : int, optional
                Maximum Sobol order to consider, by default 1

            Returns
            -------
            partial_var_df : pandas.DataFrame
                DataFrame of partial variances for each QoI and Sobol index

            sobol_index_df : pandas.DataFrame
                DataFrame of Sobol indices for each QoI and Sobol index

            total_var : numpy.array
                Total variance of each QoI """
        
        a_i_alpha = self.model.u_alpha_i.T
        V_a_orig = self.model.basis
        QoI_names = self.QoI_names
        
        # Get the multiindex set and convert to a logical array
        V_a = copy.deepcopy(V_a_orig)
        I_a = V_a.I #210x6
        I_s = I_a.astype(bool) #210x6
        
        # Identify the alpha=0 term (mean) to exclude from total variance
        is_alpha0 = np.all(I_a == 0, axis=1)
        
        # Calculate total variance excluding alpha=0
        a_i_alpha_nonzero = a_i_alpha[:, ~is_alpha0]
        sqr_norm_nonzero = V_a_orig.norm(do_sqrt=False)[~is_alpha0]
        var_row_nonzero = np.multiply(a_i_alpha_nonzero**2, sqr_norm_nonzero.transpose())
        total_var = np.sum(var_row_nonzero, axis=1)[:, np.newaxis]

        # Get rid of indices with too high an order and the mean
        sobol_order = np.sum(I_s, axis=1)
        ind = (sobol_order > 0) & (sobol_order <= max_index)

        I_a = I_a[ind, :]
        I_s = I_s[ind, :]
        a_i_alpha = a_i_alpha[:, ind]
        V_a.I = I_a

        # Calculate variance of the a_i_alpha * F_alpha polynomials
        sqr_norm = V_a.norm(do_sqrt=False)
        var_row = np.multiply(a_i_alpha**2, sqr_norm.transpose())

        # Get unique rows from Sobol indices
        I_s, _, ind2 = np.unique(I_s, axis=0, return_index=True, return_inverse=True)
        M = V_a.size()[0]
        U = csr_matrix((np.ones(M), (np.arange(M), ind2)), shape=(M, len(I_s)))
        
        # Compute the partial variance
        partial_var = var_row @ U.toarray()

        # Sort partial variances by Sobol order
        order_criterion = np.column_stack([np.sum(I_s, axis=1), np.flip(I_s, axis=1)])
        sortind = np.lexsort([order_criterion[:, i] for i in reversed(range(order_criterion.shape[1]))])
        I_s = I_s[sortind]
        partial_var = partial_var[:, sortind]
        
        indexed_param_names = gpc_multiindex2param_names(I_s, self.Q.variable_names())
        
        # Compute the Sobol indices
        sobol_index = np.divide(partial_var, total_var)
        
        partial_var_df = pd.DataFrame(partial_var, columns=indexed_param_names, index=QoI_names)
        sobol_index_df = pd.DataFrame(sobol_index, columns=indexed_param_names, index=QoI_names)
                
        return partial_var_df, sobol_index_df, total_var


    def to_jsonld(self, model_id=None, building_did=None, building_ual=None, name=None, doc_type=None, description=None):
        """ Converts the gPCE model to JSON-LD format.

            Parameters
            ----------
            model_id : str, optional
                Identifier for the model.
            building_did : str, optional
                Decentralized identifier for the building.
            building_ual : str, optional
                Uniform Asset Locator for the building.
            name : str, optional
                Name of the model.
            doc_type : str, optional
                Document type.
            description : str, optional
                Description of the model.

            Returns
            -------
            jsonld : dict
                JSON-LD representation of the gPCE model.
        """

        hyperparameters = [
            {
                "@id": "#p",
                "@type": "mls:HyperParameter",
                "rdfs:label": "p"
            }
        ]

        hyperparameter_settings = [
            {
                "@id": "#p_setting",
                "@type": "mls:HyperParameterSetting",
                "mls:specifiedBy": {
                    "@id": "#p"
                },
                "mls:hasValue": self.p
            }
        ]

        mls_spec = {
            "@id": f"urn:implementation:{model_id}",
            "@type": "mls:Implementation",

            "mls:implements": {
                "@id": "urn:algorithm:generalized-polynomial-chaos",
                "@type": "mls:Algorithm",
                "rdfs:label": "Generalized Polynomial Chaos"
            },

            "mls:hasHyperParameter": hyperparameters
        }

        mls_inputs = [
            {
                "@type": "mls:Feature",

                "rdfs:label": feat_name,

                "mls:hasQuality": {
                    "@type": "mls:FeatureCharacteristic",

                    "distributionType": dist.get_dist_type(),
                    "distributionParameters": dist.get_dist_params()
                }
            }
            for feat_name, dist in self.Q.get_variables().items()
        ]

        if building_did is not None:
            jsonld = {
                "@context": {
                    "schema": "https://schema.org/",
                    "dcterms": "http://purl.org/dc/terms/",
                    "dkg": "https://origintrail.io/ontology/",
                    "mls": "http://www.w3.org/ns/mls#",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "prov": "http://www.w3.org/ns/prov#"
                },

                "@id": model_id,
                "@type": "schema:Dataset",

                "schema:name": name,
                "dcterms:type": doc_type,
                "schema:description": description,
                "schema:hasPart": {"@id": building_did, "dkg:ual": building_ual},
                "dcterms:created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "prov:wasGeneratedBy": {
                    "@type": "prov:Activity",
                    "schema:name": "Trace Structures API",
                    "schema:url": "https://buildchain.ilab.sztaki.hu/"
                },

                "mls:specifiedBy": mls_spec,
                "mls:hasPart": hyperparameter_settings,
                "mls:hasInput": mls_inputs
            }
        else:
            jsonld = {
                "@context": {
                    "mls": "http://www.w3.org/ns/mls#",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
                },

                "@id": f"urn:model:{model_id}",
                "@type": "mls:Model",

                "mls:specifiedBy": mls_spec,
                "mls:hasPart": hyperparameter_settings,
                "mls:hasInput": mls_inputs
            }

        return jsonld
